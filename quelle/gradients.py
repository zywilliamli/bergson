import hashlib
from contextlib import ContextDecorator
from dataclasses import dataclass

import faiss
import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from datasets import Dataset
from torch import Tensor
from tqdm.auto import trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor


def apply_second_moments(
    model: nn.Module,
    moments: dict[str, Tensor],
    eps: float = 1e-8,
):
    """Precondition the model's gradients using the second moments."""
    for name, param in model.named_parameters():
        if (g := param.grad) is None:
            continue

        # We don't have any second moment for this parameter
        if name not in moments:
            continue

        g /= moments[name].add(eps).sqrt()

    return moments


@torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported())
def build_index_old(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    path: str,
    *,
    batch_size: int = 32,
    num_examples: int = 0,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    from faiss import IndexFlat

    index = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Use the entire dataset
    if num_examples <= 0:
        num_examples = len(data)
    else:
        num_examples = min(num_examples, len(data))

    chunk_size = num_examples // world_size

    # Shuffle the order in which we visit the batches just so that the progress bar
    # shows a valid estimate for the total time
    num_batches = num_examples // (batch_size * world_size)
    indices = np.random.permutation(num_batches)

    for i in trange(num_batches, position=rank):
        j = indices[i]
        batch = data[j * batch_size : (j + 1) * batch_size]

        with AdafactorProjHookManager(model, moments, 16, 16) as mgr:
            x = pad_and_tensor(batch["input_ids"], device=model.device)
            model(x, labels=x).loss.backward()
            model.zero_grad()

        grads = mgr.flat_grads
        assert grads is not None, "No matrix-valued gradients found"

        if index is None:
            index = IndexFlat(grads.shape[1])

            # Figure out how much RAM we have
            ram = psutil.virtual_memory().available
            grad_size = grads[0].element_size() * grads[0].numel()

            # Check if we can fit the index in RAM
            chunk_size = min(chunk_size, ram // (grad_size * 2))

            if rank == 0:
                print(f"Total dataset size: {len(data):_}")
                if num_examples < len(data):
                    print(f"Using only {num_examples:_} examples for the index")

                print(f"RAM available: {ram / 2**30:.2f} GB")
                print(f"Grad dimension: {grads.shape[1]}")
                if chunk_size < num_examples:
                    print(f"Conservatively using chunk size of {chunk_size:_} examples")

        index.add(grads.cpu().float().numpy())  # type: ignore

    # Save the index to disk
    idx_path = path + f"/rank_{rank}.faiss"
    print(f"Saving index to {idx_path}")
    faiss.write_index(idx_path, index)


def estimate_preconditioner(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    moments: dict[str, Tensor],
    num_examples: int = 1000,
):
    """
    Estimate the second moment matrix of the projected gradients.
    """
    preconditioner = None
    rank = dist.get_rank() if dist.is_initialized() else 0

    for i in trange(num_examples, position=rank):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        apply_second_moments(model, moments)
        grad = project_grads(model)
        model.zero_grad()

        if preconditioner is None:
            preconditioner = torch.outer(grad, grad) / num_examples
        else:
            preconditioner.addmm_(grad[:, None], grad[None], alpha=1 / num_examples)

    # Sanity check
    assert preconditioner is not None, "num_examples must be > 0"

    if dist.is_initialized():
        dist.all_reduce(preconditioner)
        preconditioner /= dist.get_world_size()

    return preconditioner


def estimate_second_moments(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    num_examples: int = 1000,
) -> dict[str, tuple[Tensor, Tensor]]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    moments: dict[str, tuple[Tensor, Tensor]] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    for i in trange(num_examples, position=rank):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        for name, param in model.named_parameters():
            if (g := param.grad) is None:
                continue

            # Skip vector-valued parameters since they are negligible
            if g.ndim < 2:
                continue

            # squared grads, scaled by 1/num_examples
            sq = g.square().div_(num_examples)

            # reduce across processes if needed
            if dist.is_initialized():
                dist.all_reduce(sq, op=dist.ReduceOp.SUM)
                sq.div_(world_size)

            # accumulate row‐ and column‐sums
            # row: sum over columns, shape [O]
            row_acc = sq.mean(dim=1)
            # col: sum over rows,    shape [I]
            col_acc = sq.mean(dim=0)

            if name not in moments:
                # initialize accumulators at zero
                moments[name] = (
                    torch.zeros_like(row_acc),
                    torch.zeros_like(col_acc),
                )

            # in‐place accumulate
            moments[name][0].add_(row_acc)
            moments[name][1].add_(col_acc)

        model.zero_grad()

    return moments


@torch.inference_mode()
def project_grads(mod: nn.Module, target_dim: int = 16, *, seed: int = 0):
    """Randomly project gradients to a lower dimension.

    Only matrix-valued gradients are affected since vector-valued gradients contribute
    a negligible amount to the total parameter count.

    Args:
        mod: The model whose gradients to project.
        target_dim: Target dimension for each axis of the matrix gradients.
        seed: Random seed for reproducibility.
    """
    device = next(mod.parameters()).device
    grads: list[Tensor] = []
    prng = torch.Generator(device).manual_seed(seed)

    for name, param in mod.named_parameters():
        # Skip if the parameter has no gradient
        if (g := param.grad) is None:
            continue

        # Skip vector-valued parameters since they are negligible
        if g.ndim < 2:
            continue

        # Higher order tensors are not supported
        if g.ndim > 2:
            raise NotImplementedError(f"Unsupported shape {g.shape} for '{name}'")

        # Create two random projection matrices
        m, n = g.shape
        proj1 = torch.randn((m, target_dim), device=g.device, generator=prng)
        proj2 = torch.randn((n, target_dim), device=g.device, generator=prng)
        proj1 = proj1 / proj1.norm(dim=0, keepdim=True)
        proj2 = proj2 / proj2.norm(dim=0, keepdim=True)

        # Project the gradient
        grads.append(proj1.T @ g @ proj2)

    if not grads:
        raise ValueError("No matrix-valued gradients found")

    return torch.stack(grads).flatten()


def _stable_seed(name: str, base_seed: int):
    md5 = hashlib.md5(name.encode("utf-8")).hexdigest()
    h = int(md5, 16) % (2**31)
    return (base_seed ^ h) & 0x7FFFFFFF


class _FlatHook:
    def __init__(
        self,
        layer: nn.Linear,
        layer_name: str,
        p: int,
        q: int,
        base_seed: int,
        moments: dict[str, tuple[Tensor, Tensor]],
        out_list: list[Tensor],
        eps: float,
    ):
        self.layer = layer
        self.name = layer_name
        self.eps = eps
        self.out_list = out_list

        R, C = moments.get(layer_name, (None, None))
        o, i = layer.out_features, layer.in_features
        seed = _stable_seed(layer_name, base_seed)

        device = layer.weight.device
        gen = torch.Generator(device).manual_seed(seed)

        A0 = torch.randn(p, o, device=device, generator=gen)
        B0 = torch.randn(q, i, device=device, generator=gen)

        if R is not None and C is not None:
            row_scale = (R + eps).rsqrt()  # [O]
            col_scale = (C + eps).rsqrt()  # [I]
            self.A = A0 * row_scale.unsqueeze(0)
            self.B = B0 * col_scale.unsqueeze(0)
        else:
            self.A = A0
            self.B = B0

        self.fwd = layer.register_forward_hook(self._save_V)
        self.bwd = layer.register_full_backward_hook(self._project)

    def _save_V(self, module, inp, out):
        # inp: [N, S, I]
        module._tmp_V = inp[0].detach() @ self.B.T  # [N, S, q]

    def _project(self, module, grad_in, grad_out):
        G = grad_out[0]  # [N, S, O]
        V = module._tmp_V  # [N, S, q]
        U = G @ self.A.T  # [N, S, p]
        P = torch.mean(U[..., None] * V[..., None, :], dim=1)  # [N, S, p, q]

        # flatten to [N, p*q] and append
        self.out_list.append(P.reshape(P.size(0), -1))

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()


@dataclass
class AdafactorProjHookManager(ContextDecorator):
    """
    After backward, `self.flat_grads` is a single tensor of shape
      [batch_size, num_layers * p * q]
    by concatenating each layer’s projected [p×q] block.
    """

    model: nn.Module
    moments: dict[str, tuple[Tensor, Tensor]]
    p: int
    q: int
    seed: int = 42
    eps: float = 1e-8

    def __post_init__(self):
        self._hooks: list[_FlatHook] = []
        self._buffers: list[Tensor] = []
        self.flat_grads: Tensor | None = None  # will be set on exit

    def __enter__(self):
        # install a hook on every Linear
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                h = _FlatHook(
                    layer=layer,
                    layer_name=name,
                    p=self.p,
                    q=self.q,
                    base_seed=self.seed,
                    moments=self.moments,
                    out_list=self._buffers,
                    eps=self.eps,
                )
                self._hooks.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        # concatenate all the flattened [N, p*q] chunks → [N, total]
        if self._buffers:
            # assume all have same N
            self.flat_grads = torch.cat(self._buffers, dim=1)

        # clean up hooks
        for h in self._hooks:
            h.remove()

        return False
