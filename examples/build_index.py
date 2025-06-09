import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from datasets import Dataset, Value, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bergson import build_index, fit_normalizers
from bergson.data import MemmapDataset, compute_batches
from bergson.gradients import GradientProcessor
from bergson.utils import assert_type


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "run_name",
        type=str,
        help="Name of the run. Used to create a directory for the index.",
    )
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/SmolLM2-135M-10B",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model in 8-bit mode. Requires the bitsandbytes library.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=16,
        help="Dimension of the random projection for the index, or 0 to disable it.",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=8192,
        help="Batch size in tokens for building the index.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="text",
        help="Column in the dataset that contains the prompts.",
    )
    parser.add_argument(
        "--completion-column",
        type=str,
        default="",
        help="Optional column in the dataset that contains the completions.",
    )
    parser.add_argument(
        "--conversation-column",
        type=str,
        default="",
        help="Optional column in the dataset that contains the conversation.",
    )
    parser.add_argument(
        "--stats-sample-size",
        type=int,
        default=10_000,
        help="Number of examples to use for the second moments",
    )
    parser.add_argument(
        "--drop-columns",
        type=int,
        default=10_000,
        help="Only return the new dataset columns",
    )
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group("nccl")

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank)

    dtype = None
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
        ),
        torch_dtype=dtype,
    )

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    def tokenize(batch):
        # We're dealing with a prompt-completion dataset
        if args.completion_column:
            return tokenizer.apply_chat_template(
                conversation=[
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                    ]
                    for prompt, resp in zip(
                        batch[args.prompt_column], batch[args.completion_column]
                    )
                ],
                return_dict=True,
                tokenizer_kwargs=dict(
                    return_attention_mask=False,
                    return_length=True,
                ),
                truncation=True,
            )
        elif args.conversation_column:
            return tokenizer.apply_chat_template(
                conversation=batch[args.conversation_column],
                return_dict=True,
                tokenizer_kwargs=dict(
                    return_attention_mask=False,
                    return_length=True,
                ),
                truncation=True,
            )
        # We're dealing with vanilla next-token prediction
        else:
            return tokenizer(
                batch[args.prompt_column],
                return_attention_mask=False,
                return_length=True,
                truncation=True,
            )
    
    if args.dataset.endswith(".bin"):
        # TODO: Make this configurable, right now this is just a hack to support
        # the Pythia preshuffled Pile dataset.
        MEMMAP_CTX_LEN = 2049

        # If the dataset is a memmap file, use MemmapDataset
        ds = MemmapDataset(args.dataset, MEMMAP_CTX_LEN)
        ds = ds.shard(world_size, rank)

        # Uniform batches
        batch_size = args.token_batch_size // MEMMAP_CTX_LEN
        batches = [
            slice(start, start + batch_size) for start in range(0, len(ds), batch_size)
        ]
    else:
        try:
            ds = assert_type(Dataset, load_dataset(args.dataset, split="train"))
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e
        
        assert "_original_idx" not in ds.column_names, (
            "The dataset already contains a column named '_original_idx'. "
        )

        metadata = {"length"}
        if args.drop_columns:
            metadata.update(set(ds.column_names))

        ds = ds.map(lambda x, idx: {**x, "_original_idx": idx}, with_indices=True)

        # Shuffle before sharding to make sure each rank gets a different subset
        ds = ds.shuffle(seed=42)
        ds = ds.shard(world_size, rank)

        # Tokenize
        ds = ds.map(tokenize, batched=True)
        ds = ds.sort("length", reverse=True)
        batches = compute_batches(ds["length"], args.token_batch_size)

        ds = ds.remove_columns(list(metadata))


    if os.path.exists(args.run_name):
        processor = GradientProcessor.load(args.run_name, map_location=f"cuda:{rank}")
    else:
        if rank == 0:
            print("Estimating normalizers...")

        processor = GradientProcessor(
            normalizers=fit_normalizers(
                model,
                ds,
                batches=batches,
                max_documents=args.stats_sample_size or None,
            ),
            projection_dim=args.projection_dim or None,
        )

    if not processor.preconditioners:
        if rank == 0:
            print("Estimating preconditioners...")

        # We need a lot of examples for the preconditioner
        processor.estimate_preconditioners(
            model,
            ds,
            batches=batches,
            max_documents=args.stats_sample_size or None,
        )
        processor.save(args.run_name)

    if rank == 0:
        print("Building index...")

    # Build the index
    build_index(model, ds, processor, args.run_name, batches=batches)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
