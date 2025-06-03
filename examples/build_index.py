import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import build_index, estimate_preconditioners, estimate_second_moments
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
        "--projection-dim",
        type=int,
        default=16,
        help="Dimension of the random projection for the index, or 0 to disable it.",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=8196,
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
        "--index-size",
        type=int,
        default=0,
        help="Maximum of examples to use for the index, or 0 to use the entire dataset",
    )
    parser.add_argument(
        "--precondition",
        action="store_true",
        help="Use the preconditioner to build the index",
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": f"cuda:{rank}"}
    )

    def tokenize(batch):
        prompts = tokenizer(
            batch[args.prompt_column],
            truncation=True,
        ).input_ids
        if args.completion_column:
            resps = tokenizer(
                batch[args.completion_column],
                truncation=True,
            ).input_ids
            inputs = [prompt + resp for prompt, resp in zip(prompts, resps)]
            labels = [
                [-100] * len(prompt) + resp for prompt, resp in zip(prompts, resps)
            ]
            return {
                "input_ids": inputs,
                "labels": labels,
                "length": [len(inp) for inp in inputs],
            }
        else:
            return {
                "input_ids": prompts,
                "length": [len(prompt) for prompt in prompts],
            }

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
        ds = assert_type(Dataset, load_dataset(args.dataset, split="train"))
        ds = ds.shard(world_size, rank)

        # Tokenize
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        ds = ds.sort("length", reverse=True)
        batches = compute_batches(ds["length"], args.token_batch_size)

    # Create the run directory
    os.makedirs(args.run_name, exist_ok=True)

    if not os.path.exists(args.run_name + "/normalizers.pth"):
        if rank == 0:
            print("Estimating second moments...")

        normalizers = estimate_second_moments(model, ds, num_examples=1000)
        torch.save(normalizers, args.run_name + "/normalizers.pth")
    else:
        if rank == 0:
            print("Loading second moments from disk.")

        normalizers = torch.load(
            args.run_name + "/normalizers.pth",
            map_location=f"cuda:{rank}",
            weights_only=False,
        )

    processor = GradientProcessor(
        normalizers,
        projection_dim=args.projection_dim or None,
    )

    if args.precondition:
        # TODO: Actually use the preconditioner
        if not os.path.exists(args.run_name + "/preconditioner.pth"):
            if rank == 0:
                print("Estimating preconditioner...")

            # We need a lot of examples for the preconditioner
            preconditioner = estimate_preconditioners(
                model, ds, processor, num_examples=10_000
            )
            torch.save(preconditioner, args.run_name + "/preconditioner.pth")
        else:
            if rank == 0:
                print("Loading preconditioner from disk.")

            preconditioner = torch.load(
                args.run_name + "/preconditioner.pth",
                map_location=f"cuda:{rank}",
                weights_only=True,
            )

    if rank == 0:
        print("Building index...")

    # Build the index
    build_index(model, ds, processor, args.run_name, batches=batches)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
