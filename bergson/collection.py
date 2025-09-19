import math
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .data import create_index, pad_and_tensor
from .gradients import GradientCollector, GradientProcessor, HeadConfig
from .peft import set_peft_enabled


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[list[int]] | None = None,
    kl_divergence: bool | None = None,
    loss_reduction: Literal["mean", "sum"] = "mean",
    skip_preconditioners: bool = False,
    target_modules: set[str] | None = None,
    head_cfgs: dict[str, HeadConfig] = {},
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [[idx] for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = {}
    preconditioners = {}

    # TODO: Handle this more elegantly
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    lo = torch.finfo(dtype).min
    hi = torch.finfo(dtype).max

    def callback(name: str, g: torch.Tensor):
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads[name] = g.to(device="cpu", dtype=dtype, non_blocking=True)

        # Compute the outer product of the flattened gradient
        if not skip_preconditioners:
            g = g.float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioner.addmm_(g.mT, g)

    collector = GradientCollector(
        model.base_model,
        callback,
        processor,
        target_modules=target_modules,
        head_cfgs=head_cfgs,
    )

    # Allocate space ahead of time for the gradients
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    # Allocate structured space ahead of time for the gradients
    grad_buffer = create_index(
        path, num_grads=len(data), grad_sizes=grad_sizes, dtype=np_dtype
    )

    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=dtype,
        fill_value=0.0,
    )

    for indices in tqdm(batches, disable=rank != 0, desc="Building index"):
        batch = data[indices]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )
        masks = y[:, 1:] != -100
        denoms = masks.sum(dim=1, dtype=dtype) if loss_reduction == "mean" else 1.0

        if kl_divergence:
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            with collector:
                ft_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)

                # Compute average KL across all unmasked tokens
                kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
                losses = torch.sum(kls * masks, dim=-1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()
        else:
            with collector:
                logits = model(x).logits[:, :-1]

                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:].flatten(),
                    reduction="none",
                ).reshape_as(y[:, 1:])
                losses = losses.sum(1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()

        # Weirdly you need to explicitly synchronize here in order to make sure that
        # the nonblocking copies actually finish before we call .numpy()
        model.zero_grad()
        torch.cuda.synchronize()

        # It turns out that it's very important for efficiency to write the gradients
        # sequentially instead of first concatenating them, then writing to one vector
        for module_name in mod_grads.keys():
            grad_buffer[module_name][indices] = mod_grads[module_name].numpy()

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

    process_preconditioners(processor, preconditioners, len(data))

    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16" if dtype == torch.float16 else "float32"),
            new_fingerprint="loss",
        )
        data.save_to_disk(path + "/data.hf")

        processor.save(path)

    # Make sure the gradients are written to disk
    grad_buffer.flush()


def process_preconditioners(
    processor: GradientProcessor,
    preconditioners: dict[str, torch.Tensor],
    len_data: int,
):
    """
    Aggregate preconditioners across ranks and compute their eigen decomposition
    distributed across all ranks.
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    preconditioners_eigen = {}
    if rank == 0:
        print("Saving preconditioners...")
    for name, prec in preconditioners.items():
        if dist.is_initialized():
            dist.all_reduce(prec)

        preconditioners[name] = prec / len_data

    processor.preconditioners = preconditioners

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")
    names = list(preconditioners.keys())
    names_per_rank = names[rank::world_size]

    for name in names_per_rank:
        original_dtype = preconditioners[name].dtype
        prec = preconditioners[name].to(dtype=torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        preconditioners_eigen[name] = (
            eigvals.to(dtype=original_dtype).contiguous(),
            eigvecs.to(dtype=original_dtype).contiguous(),
        )

    if rank == 0:
        print("Gathering and saving preconditioner eigen decompositions...")

    for name in names:
        prec = preconditioners[name]
        if name not in preconditioners_eigen:
            eigval = torch.zeros(prec.size(0), dtype=prec.dtype, device=prec.device)
            eigvec = torch.zeros_like(prec)
        else:
            eigval, eigvec = preconditioners_eigen[name]

        dist.all_reduce(eigval, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
        dist.all_reduce(eigvec, op=dist.ReduceOp.SUM) if dist.is_initialized() else None

        preconditioners_eigen[name] = (eigval, eigvec)
    if rank == 0:
        processor.preconditioners_eigen = preconditioners_eigen
