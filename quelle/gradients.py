import faiss
import psutil
import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from datasets import Dataset
from torch import nn
from tqdm.auto import trange
from transformers import PreTrainedModel

from .data import MemmapDataset


def apply_second_moments(
    model: nn.Module,
    moments: dict[str, torch.Tensor],
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
def build_index(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    moments: dict[str, torch.Tensor],
    path: str,
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

    chunk_size = num_examples
    chunk_idx = 0

    for i in trange(num_examples // world_size, position=rank):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        apply_second_moments(model, moments)
        grad = project_grads(model).float()
        model.zero_grad()

        if index is None and rank == 0:
            index = IndexFlat(grad.shape[0])

            print(f"Total dataset size: {len(data):_}")
            if num_examples < len(data):
                print(f"Using only {num_examples:_} examples for the index")

            # Figure out how much RAM we have
            ram = psutil.virtual_memory().available
            print(f"RAM available: {ram / 2**30:.2f} GB")

            print(f"Grad dimension: {grad.shape[0]}")
            grad_size = grad.element_size() * grad.numel()

            # Check if we can fit the index in RAM
            chunk_size = min(chunk_size, ram // (grad_size * 2))
            if chunk_size < num_examples:
                print(f"Conservatively using chunk size of {chunk_size:_} examples")

        if dist.is_initialized():
            # Master rank needs to actually add the gradients to the index
            if rank == 0:
                bufs = [torch.empty_like(grad) for _ in range(world_size)]
                dist.gather(grad, bufs)

                vecs = torch.stack(bufs).cpu().numpy()
                index.add(vecs)  # type: ignore
            else:
                # Other ranks just send the gradients to the master rank
                dist.gather(grad, None)
        else:
            index.add(grad.cpu().unsqueeze(0).numpy())  # type: ignore

        # Check chunk size
        if index is not None and index.ntotal >= chunk_size:
            # Save the index to disk
            idx_path = path + f"/index_{chunk_idx}.faiss"
            print(f"Saving index to {idx_path}")
            faiss.write_index(idx_path, index)

            # Reset the index
            index = IndexFlat(grad.shape[0])
            chunk_idx += 1


def estimate_preconditioner(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    moments: dict[str, torch.Tensor],
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
) -> dict[str, torch.Tensor]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    moments: dict[str, torch.Tensor] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0

    for i in trange(num_examples, position=rank):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        for name, param in model.named_parameters():
            if (g := param.grad) is None:
                continue

            if name not in moments:
                moments[name] = torch.zeros_like(g)

            # Accumulate the second moment
            acc = g.square() / num_examples

            if dist.is_initialized():
                dist.all_reduce(acc)
                acc /= dist.get_world_size()

            moments[name] += acc

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
    grads: list[torch.Tensor] = []
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
