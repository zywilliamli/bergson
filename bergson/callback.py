import os

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import DTypeLike
from torch import Tensor
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from bergson import GradientCollector, GradientProcessor
from bergson.data import create_index


class GradientCollectorCallback(TrainerCallback):
    """Callback that collects gradients from the model during training.
    Implements the same functionality as `collect_gradients`."""

    def __init__(
        self,
        collector: GradientCollector,
        processor: GradientProcessor,
        mod_grads: dict[str, Tensor],
        preconditioners: dict[str, Tensor],
        grad_sizes: dict[str, int],
        path: str,
        dtype: DTypeLike = np.float16,
        mean_gradients: bool = False,
        normalize_eval: bool = False,
    ):
        """
        Args:
            grad_sizes: The sizes of the module gradients
            processor: The gradient processor
            mod_grads: The module gradients
            preconditioners: The gradient preconditioners
            path: The path to the gradient store
            dtype: The dtype of the on-disk gradient store
            mean_gradients: Whether to take the mean of the gradients
                of the same example across epochs. If `False`, the
                gradients for each epoch are stored separately.
            normalize_eval: Whether to normalize the evaluation gradients
                using the training optimizer normalization.
        """
        super().__init__()
        self.collector = collector
        self.processor = processor
        self.mod_grads = mod_grads
        self.preconditioners = preconditioners
        self.grad_sizes = grad_sizes
        self.path = path
        self.dtype = dtype
        self.mean_gradients = mean_gradients
        self.normalize_eval = normalize_eval

    def write_grads(
        self,
        grad_buffer: np.memmap,
        mod_grads: dict[str, Tensor],
        indices: Tensor,
    ):
        for layer_name in mod_grads.keys():
            torch.cuda.synchronize()
            grad_buffer[layer_name][indices] = mod_grads[layer_name].numpy()
        self.mod_grads.clear()

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = int(state.epoch or 0)

        if self.mean_gradients and epoch > 0:
            return

        epoch_suffix = "" if self.mean_gradients else f"/epoch_{epoch}"

        # Set up the gradient buffers for the training dataset
        self.train_indices = [
            i.item() for batch in kwargs["train_dataloader"] for i in batch["_idx"]
        ]
        self.num_examples = len(self.train_indices)
        self.train_grad_buffer = create_index(
            os.path.join(self.path, "train" + epoch_suffix),
            num_grads=self.num_examples,
            grad_sizes=self.grad_sizes,
            dtype=self.dtype,
        )
        self.train_step_idx = 0

        # Set up the gradient buffers for the evaluation datasets
        if kwargs["eval_dataloader"] is None:
            return
        elif isinstance(kwargs["eval_dataloader"], dict):
            eval_datasets = kwargs["eval_dataloader"]
        else:
            eval_datasets = {"eval": kwargs["eval_dataloader"]}

        self.eval_grad_buffers = {}
        self.eval_step_idxs = {}
        self.eval_indices = {}

        for dataset_name, dataloader in eval_datasets.items():
            self.eval_grad_buffers[dataset_name] = create_index(
                os.path.join(self.path, dataset_name + epoch_suffix),
                num_grads=len(dataloader),
                grad_sizes=self.grad_sizes,
                dtype=self.dtype,
            )
            self.eval_step_idxs[dataset_name] = 0
            self.eval_indices[dataset_name] = [
                i.item() for batch in dataloader for i in batch["_idx"]
            ]

        if epoch > 0 and not self.normalize_eval:
            # Enable normalization for training
            raise NotImplementedError("Not normalizing is not supported yet.")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Save the preconditioners
        chols = {}
        for name, prec in self.preconditioners.items():
            if dist.is_initialized():
                dist.all_reduce(prec)

            L, info = torch.linalg.cholesky_ex(prec / self.num_examples)
            if info.any() and rank == 0:
                print(f"Warning: {name} has a singular second moment matrix.")

            chols[name] = L

        self.processor.preconditioners = chols

        if rank == 0:
            self.processor.save(self.path)

        # Ensure the gradients are written to disk
        self.train_grad_buffer.flush()
        for eval_grad_buffer in self.eval_grad_buffers.values():
            eval_grad_buffer.flush()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each training step.
        If using gradient accumulation, one training step might take several inputs."""
        indices = torch.tensor(
            self.train_indices[
                self.train_step_idx : self.train_step_idx
                + args.per_device_train_batch_size
            ]
        )
        self.train_step_idx += args.per_device_train_batch_size

        self.write_grads(self.train_grad_buffer, self.mod_grads, indices)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.normalize_eval:
            # Disable normalization in the gradient collector
            raise NotImplementedError("Not normalizing is not supported yet.")

    def on_prediction_step(self, args, state, control, **kwargs):
        dataset_name = kwargs["inputs"]["dataset_name"]

        indices = torch.tensor(
            self.eval_indices[dataset_name][
                self.eval_step_idxs[dataset_name] : self.eval_step_idxs[dataset_name]
                + args.per_device_eval_batch_size
            ]
        )
        self.eval_step_idxs[dataset_name] += args.per_device_eval_batch_size

        self.write_grads(
            self.eval_grad_buffers[dataset_name],
            self.mod_grads,
            indices,
        )
