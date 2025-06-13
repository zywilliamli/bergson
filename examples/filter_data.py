from typing import Literal
import os
import heapq
from dataclasses import dataclass
from contextlib import nullcontext, redirect_stdout

from simple_parsing import parse
from tqdm import tqdm
from torch import Tensor
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, setup_chat_format

from bergson.data import load_index
from bergson.utils import assert_type


def select_topk(ds: Dataset, n: int, key: str, reverse: bool = False):
    heap = []

    for idx, s in enumerate(ds[key]):
        key = -s if reverse else s

        if len(heap) < n:
            heapq.heappush(heap, (key, idx))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, idx))
    return ds.select([i for _, i in heap])


def normalize_batch(batch):
    return {
        "gradient": F.normalize(batch["gradient"].cuda(), dim=1).cpu(),
    }


def add_index(
    dataset: Dataset,
    index: str | None = None,
) -> Dataset:
    assert index is not None, "Index must be provided for attribution filtering"

    gradient_ds = (
        load_index(index)
        .with_format("torch")
        .map(normalize_batch, batched=True, batch_size=512)
        .sort("row_number")
        .select_columns(["gradient", "row_number"])
    )

    def generator():
        for row, grad_row in zip(dataset, gradient_ds):
            yield {**dict(row), **dict(grad_row)}

    return assert_type(Dataset, Dataset.from_generator(generator))


def get_importance_scores(train: Dataset, test: Dataset, batch_size: int):
    """
    Assign training items influence scores for the test set.
    Does not normalize gradients with second moment estimates.
    """

    mean_test_grads = assert_type(Tensor, test["gradient"]).mean(0).cuda()

    scores = torch.empty(len(train))
    for i in tqdm(range(0, len(train), batch_size)):
        batch = assert_type(Tensor, train["gradient"])[i : i + batch_size].cuda()
        if len(batch) == 0:
            continue

        # Compute the influence scores for this batch
        batch_scores = batch @ mean_test_grads
        scores[i : i + batch_size] = batch_scores.cpu()

    return scores


def main(
    filter: Literal["classification", "attribution", "random"],
    n: int,
    model_name: str,
    dataset_name: str,
    index: str | None = None,
    name: str | None = None,
    reverse: bool = False,
):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    if name is None:
        run_name = (
            f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}-{filter}"
            f"{'-reverse' if reverse else ''}"
        )
    else:
        run_name = name

    with nullcontext() if rank == 0 else redirect_stdout(None):
        dataset = assert_type(Dataset, load_dataset(dataset_name, split="train"))

        if filter == "attribution":
            dataset = add_index(dataset, index)

        dataset.shuffle(42).with_format("torch")

        train, eval = dataset.train_test_split(
            test_size=0.05,
            seed=42,
            load_from_cache_file=True,
            train_indices_cache_file_name=f"cache/{run_name}/train.arrow",
            test_indices_cache_file_name=f"cache/{run_name}/test.arrow",
        ).values()

        train.set_format("torch")
        eval.set_format("torch")

        print("Filtering...")
        if filter == "attribution":
            eval_grad = assert_type(Tensor, eval["gradient"]).mean(0).cuda()

            def importance_score_generator():
                for row in train:
                    row = dict(row)
                    yield {
                        **row,
                        "importance_score": (row["gradient"].cuda() @ eval_grad).item(),
                    }

            train = assert_type(
                Dataset, Dataset.from_generator(importance_score_generator)
            )
            train = select_topk(train, n, "importance_score", reverse=reverse)
        elif filter == "classification":
            ranks = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "very poor": 0}

            def add_rank(ex):
                q = ex.get("quality")
                return {"_q": ranks.get(q, -1)}

            train = (
                train.map(add_rank)
                .filter(lambda x: x["_q"] >= 0)
                .sort("_q", reverse=True)
                .select(range(min(n, len(train))))
                .remove_columns("_q")
            )
        elif filter == "random":
            train = train.select(range(min(n, len(train))))
        else:
            raise ValueError(f"Invalid filter: {filter}")

        # TODO switch to data.tokenize util
        # https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face
        if "conversation" in train.column_names:
            train = train.rename_column("conversation", "messages")
            eval = eval.rename_column("conversation", "messages")
            metadata = set(train.column_names) - {"messages"}
            train = train.remove_columns(list(metadata))
            eval = eval.remove_columns(list(set(eval.column_names) - {"messages"}))
        elif "response" in train.column_names:

            def to_pc(x):
                return {"prompt": x["instruction"], "completion": x["response"]}

            train = train.map(to_pc, remove_columns=train.column_names)
            eval = eval.map(to_pc, remove_columns=eval.column_names)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=8192)
        model, tokenizer = setup_chat_format(model, tokenizer)

        print("Current chat format:")
        print(tokenizer.chat_template)

        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config.yaml
        trainer = SFTTrainer(
            model=model,
            train_dataset=train,
            eval_dataset=eval,
            args=SFTConfig(
                max_length=8192,
                max_seq_length=8192,
                output_dir=f"runs/{run_name}",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=32,
                learning_rate=3e-4,
                num_train_epochs=2,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=10,
                eval_steps=20,
                save_steps=100,
                save_total_limit=3,
                group_by_length=True,
                completion_only_loss=True,
                ddp_find_unused_parameters=False,
                seed=42,
            ),
            processing_class=tokenizer,
        )

        trainer.train()

        if rank == 0:
            trainer.save_model(f"examples/runs/{run_name}")
            tokenizer.save_pretrained(f"examples/runs/{run_name}")


@dataclass
class FilterConfig(SFTConfig):
    """Config for building the index and running the model/dataset pipeline."""

    model: str = "HuggingFaceTB/SmolLM2-1.7B"
    """Name of the model to load."""

    dataset: str = "argilla/magpie-ultra-v0.1"
    """Dataset identifier to finetune on."""

    filter: Literal["classification", "attribution", "random"] = "attribution"
    """Filter to apply to the training set before finetuning."""

    index: str = ""
    """Bergson index to use for attribution filtering."""

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    n: int = 30_000
    """Number of items to select from the training set."""

    reverse: bool = False
    """Reverse the scores."""


if __name__ == "__main__":
    args = parse(FilterConfig)

    main(
        args.filter,
        args.n,
        args.model,
        args.dataset,
        args.index,
        args.name,
        args.reverse,
    )
