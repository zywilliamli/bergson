from contextlib import contextmanager

import torch
from torch import Tensor, nn

from .gradients import Normalizer


class FiniteDiff:
    def __init__(
        self,
        model: nn.Module,
        *,
        normalizers: dict[str, Normalizer] | None = None,
    ):
        self.model = model
        self.normalizers = normalizers or {}
        self.params: dict[str, Tensor] = {}

    def compute(self, step_size: float):
        """Compute and store finite differences for the model parameters.

        This method assumes that you've just called .backward() on some loss function,
        so the model's parameters have gradients. We reuse these gradient buffers and
        set the .grad attributes to None to avoid unnecessary memory usage.
        """
        num_normalized = 0

        for name, param in self.model.named_parameters():
            if (g := param.grad) is None:
                continue

            ## Normalize the gradient if needed
            mod_name = name.removesuffix(".weight")
            if (n := self.normalizers.get(mod_name)) is not None:
                n.normalize_(g)
                num_normalized += 1

            # Compute the finite difference in-place *in the grad buffer*
            torch.add(param.data, g, alpha=step_size, out=g)

            # Stash the grad buffer and set p.grad to None
            self.params[name] = g
            param.grad = None

        # Sanity check to make sure we actually use the normalizers
        if num_normalized != len(self.normalizers):
            raise ValueError(
                f"Expected {len(self.normalizers)} normalized gradients, "
                f"but got {num_normalized}."
            )

    def swap(self):
        """Swap the original and updated parameters."""
        for name, param in self.model.named_parameters():
            if (buf := self.params.get(name)) is None:
                continue

            # Perform efficient XOR swap to avoid copying
            assert buf.shape == param.shape
            a_i, b_i = buf.view(torch.uint8), param.data.view(torch.uint8)
            a_i.bitwise_xor_(b_i)  # a = a ^ b
            b_i.bitwise_xor_(a_i)  # b = (a ^ b) ^ b = a
            a_i.bitwise_xor_(b_i)  # a = (a ^ b) ^ a = b

    @contextmanager
    def apply(self):
        """Context manager to apply finite differences to the model."""
        self.swap()
        try:
            yield self
        finally:
            self.swap()

    def clear(self):
        """
        Clear the stored finite differences.
        """
        self.params.clear()
