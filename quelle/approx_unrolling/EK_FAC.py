import argparse
import itertools
import logging
import os
from typing import Dict

import torch
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from language_task import LanguageModelingTask
from safetensors.torch import load_file, save_file
from torch.utils import data
from transformers import default_data_collator

from quelle.approx_unrolling.logger_config import get_logger
from quelle.approx_unrolling.model_checkpoints import (
    ModelCheckpointManager,
)
from quelle.approx_unrolling.utils import TensorDict

BATCH_TYPE = Dict[str, torch.Tensor]
logger = get_logger(__name__)


def compute_EK_FAC(
    model: torch.nn.Module,
    task: LanguageModelingTask,
    train_dataset: data.Dataset,
    output_dir: str,
    EK_FAC_args: argparse.Namespace,
):
    logging.basicConfig(level=logging.INFO)

    # Define task and prepare model.
    model = prepare_model(model, task)

    if EK_FAC_args.use_compile:
        model = torch.compile(model)  # type: ignore

    analyzer = Analyzer(
        analysis_name="",
        output_dir=output_dir,
        model=model,
        task=task,
        profile=EK_FAC_args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = EK_FAC_args.factor_strategy
    factor_args = FactorArguments(strategy=EK_FAC_args.factor_strategy)
    if EK_FAC_args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(
            strategy=EK_FAC_args.factor_strategy, dtype=torch.bfloat16
        )
        factors_name += "_half"
    if EK_FAC_args.use_compile:
        factors_name += "_compile"
    # analyzer.fit_all_factors(
    #     factors_name=factors_name,
    #     dataset=train_dataset,
    #     per_device_batch_size=EK_FAC_args.per_device_batch_size,
    #     factor_args=factor_args,
    #     initial_per_device_batch_size_attempt=64,
    # )

    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=EK_FAC_args.per_device_batch_size,
        initial_per_device_batch_size_attempt=64,
        dataloader_kwargs=dataloader_kwargs,
        factor_args=factor_args,
    )


def compute_EK_FAC_checkpoints(
    checkpoint_manager: ModelCheckpointManager,
    task: LanguageModelingTask,
    train_dataset: data.Dataset,
    EK_FAC_args: argparse.Namespace,
    device="cuda",
):
    """Loads the model from the checkpoint directory and computes the EK-FAC matrix."""

    # Check if the model directory exists
    if not os.path.exists(checkpoint_manager.model_dir):
        raise FileNotFoundError(
            f"Model directory {checkpoint_manager.model_dir} does not exist."
        )

    for checkpoint in itertools.chain(*checkpoint_manager.all_checkpoints):
        checkpoint_path = (
            checkpoint_manager.model_dir / f"checkpoint_{checkpoint}" / "model.pt"
        )
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_path} does not exist."
            )

        # Load the model from the checkpoint

        loaded_checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint, device=device
        )

        output_dir = (
            checkpoint_manager.model_dir
            / f"checkpoint_{checkpoint}"
            / "influence_results"
        )

        compute_EK_FAC(
            model=loaded_checkpoint,
            output_dir=str(output_dir),
            task=task,
            train_dataset=train_dataset,
            EK_FAC_args=EK_FAC_args,
        )

        logger.info(
            f"Computed EK-FAC for checkpoint {checkpoint} and saved to {output_dir} \n"
            + "-" * 50
        )


def compute_average_covariance(checkpoint_manager: ModelCheckpointManager):
    for i, segment in enumerate(checkpoint_manager.all_checkpoints):
        activations = []
        gradients = []

        for checkpoint in segment:
            checkpoint_gradient_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / "factors_ekfac"
                / "activation_covariance.safetensors"
            )

            checkpoint_activation_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / "factors_ekfac"
                / "activation_covariance.safetensors"
            )
            gradients_covariance = TensorDict(load_file(checkpoint_gradient_path))
            gradients.append(gradients_covariance)

            activations_covariance = TensorDict(load_file(checkpoint_activation_path))

            activations.append(activations_covariance)

        segment_path = checkpoint_manager.model_dir / f"segment_{i}"
        segment_path.mkdir(parents=True, exist_ok=True)
        average_activation = sum(activations) / len(segment)
        average_gradient = sum(gradients) / len(segment)

        save_file(
            average_activation.to_dict(),
            segment_path / "average_activation_covariance.safetensors",
        )
        save_file(
            average_gradient.to_dict(),
            segment_path / "average_gradient_covariance.safetensors",
        )


def prepare_hessians(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
):
    # lambda_matrices = []
    # lambda_matrices_exp = []
    # # load lambda_matrices from the checkpoints

    # for checkpoint in checkpoint_manager.checkpoint_list:
    #     checkpoint_path = (
    #         checkpoint_manager.checkpoints_dir
    #         / checkpoint_manager.model_name
    #         / f"checkpoint_{checkpoint}"
    #         / "CHANGE_THIS"
    #         / "lambda_matrix.safetensors"
    #     )
    #     # Check if the checkpoint file exists
    #     if not os.path.exists(checkpoint_path):
    #         raise FileNotFoundError(
    #             f"Checkpoint file {checkpoint_path} does not exist."
    #         )
    #     lambda_matrices.append(load_file(checkpoint_path))

    # for lambda_matrix in lambda_matrices:
    #     lambda_matrix_exp = torch.exp(lambda_matrix)
    #     lambda_matrices_exp.append(lambda_matrix_exp)

    pass
