import json
import random

import torch
import torch.distributed as dist
from datasets import Dataset
from tqdm.auto import tqdm, trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor
from .utils import assert_type
from .gradients import AdafactorNormalizer, GradientCollector, GradientProcessor


def build_index(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # Pre-compute first example to define dataset features
    # and names and shapes of the gradients for serialization
    first_sl, *rest = batches
    first_batch = data[first_sl]

    with GradientCollector(model.base_model, processor) as mgr:
        x, y = pad_and_tensor(
            first_batch["input_ids"], # type: ignore
            labels=first_batch.get("labels"), # type: ignore
            device=model.device,
        )
        model(x, labels=y).loss.backward()
        model.zero_grad()

    # Drop the batch dimension from the shape
    shapes = {n: g.shape[1:] for n, g in mgr.collected_grads.items()}

    first_grads = mgr.flattened_grads().cpu().float().numpy()

    def generator():
        nonlocal first_batch, first_grads

        cols = list(first_batch.keys())

        for i, g in enumerate(first_grads):
            row = {k: first_batch[k][i] for k in cols}
            row["gradient"] = g
            yield row

        for sl in tqdm(rest, position=rank):
            batch = data[sl]

            with GradientCollector(model.base_model, processor) as mgr:
                x, y = pad_and_tensor(
                    batch["input_ids"], # type: ignore
                    labels=batch.get("labels"), # type: ignore
                    device=model.device,
                )
                model(x, labels=y).loss.backward()
                model.zero_grad()

            gradient = mgr.flattened_grads().cpu().float().numpy() 

            for i, g in enumerate(gradient):
                row = {k: batch[k][i] for k in cols}
                row["gradient"] = g
                yield row

    index = assert_type(Dataset, Dataset.from_generator(generator))
    
    if isinstance(data, Dataset):
        index = (
            index.sort("_original_idx")
                .remove_columns("_original_idx")
        )

    idx_path = path + f"/rank_{rank}.idx"
    print(f"Saving index to {idx_path}")
    index.save_to_disk(idx_path)  # type: ignore

    # Save the shapes of the gradients for later use
    if rank == 0:
        shapes_path = path + "/shapes.json"

        with open(shapes_path, "w") as f:
            json.dump(shapes, f, indent=2)

    return index


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    *,
    batches: list[slice] | None = None,
    max_documents: int | None = None,
) -> dict[str, AdafactorNormalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    moments: dict[str, AdafactorNormalizer] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # If max_tokens is specified, randomly select a subset of batches
    elif max_documents is not None:
        batches = batches.copy()

        rng = random.Random(rank)
        rng.shuffle(batches)

    def callback(name: str, g: torch.Tensor):
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.float().square_().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if name not in moments:
            # initialize accumulators at zero
            moments[name] = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )

        # in‐place accumulate
        moments[name].row.add_(row_acc)
        moments[name].col.add_(col_acc)

    N = 0
    total = (max_documents or len(batches)) // world_size
    pbar = trange(total, position=rank)

    for sl in batches:
        batch = data[sl]

        # Update progress
        n = len(range(*sl.indices(len(data))))
        pbar.update(n)

        N += n
        if max_documents and N >= max_documents:
            break

        with GradientCollector(model.base_model, closure=callback):
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

    for moment in moments.values():
        # normalize by the number of examples
        moment.row.div_(N)
        moment.col.div_(N)

        if dist.is_initialized():
            dist.all_reduce(moment.row, op=dist.ReduceOp.AVG)
            dist.all_reduce(moment.col, op=dist.ReduceOp.AVG)

    return moments
