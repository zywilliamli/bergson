import torch
from accelerate.utils import send_to_device
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


def build_index(
    model: PreTrainedModel,
    data: MemmapDataset,
    moments: dict[str, torch.Tensor],
    num_examples: int = 1000,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    from faiss import IndexFlat

    index = None

    for i in trange(num_examples):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        apply_second_moments(model, moments)
        grad = project_grads(model)
        model.zero_grad()

        if index is None:
            index = IndexFlat(grad.shape[0])

        # Type signatures of the faiss library are completely broken
        index.add(grad.cpu().unsqueeze(0).numpy())  # type: ignore

    return index


def estimate_preconditioner(
    model: PreTrainedModel,
    data: MemmapDataset,
    moments: dict[str, torch.Tensor],
    num_examples: int = 1000,
):
    """
    Estimate the second moment matrix of the projected gradients.
    """
    preconditioner = None

    for i in trange(num_examples):
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

    return preconditioner


def estimate_second_moments(
    model: PreTrainedModel,
    data: MemmapDataset,
    num_examples: int = 1000,
) -> dict[str, torch.Tensor]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    moments: dict[str, torch.Tensor] = {}

    for i in range(num_examples):
        example = send_to_device(data[i], model.device)

        x = example["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        for name, param in model.named_parameters():
            if (g := param.grad) is None:
                continue

            if name not in moments:
                moments[name] = torch.zeros_like(g)

            # Accumulate the second moment
            moments[name] += g.square() / num_examples

        model.zero_grad()

    return moments


@torch.inference_mode()
def project_grads(mod: nn.Module, target_dim: int = 32, *, seed: int = 0):
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
