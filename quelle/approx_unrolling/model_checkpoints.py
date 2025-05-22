import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import torch
from transformers import GPTNeoXForCausalLM

from quelle.approx_unrolling.logger_config import get_logger

logger = get_logger(__name__)


class ModelCheckpointManager(ABC):
    """
    Abstract base class for managing model checkpoints.
    Handles saving/loading models with organized directory structure.
    """

    def __init__(
        self,
        checkpoint_list: List[int],
        model_name: Union[str, os.PathLike],
        checkpoints_dir: str = "./checkpoints",
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoints_dir: Directory to save checkpoints (default: "./checkpoints")
        """
        self.checkpoints_dir = Path(checkpoints_dir)

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_list = checkpoint_list
        self.model_name = model_name
        self.module_keys = None

    @abstractmethod
    def load_models(self, checkpoint: int, *args, **kwargs) -> torch.nn.Module:
        """
        Abstract method to load a model
        """
        pass

    def save_models(
        self,
        overwrite: bool = False,
    ):
        """
        Save a model to the checkpoints directory.

        Args:
            model: PyTorch model to save
            model_name: Name for the saved model
            metadata: Additional metadata to save with the model
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            overwrite: Whether to overwrite existing checkpoint

        Returns:
            Path to the saved checkpoint
        """
        # Create model-specific directory
        model_dir = self.checkpoints_dir / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint in self.checkpoint_list:
            checkpoint_path = model_dir / f"checkpoint_{checkpoint}"
            # Check if file exists and handle overwrite
            if checkpoint_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Checkpoint {checkpoint_path} already exists. Set overwrite=True to replace."  # noqa: E501
                )
            else:
                logger.warning(
                    f"Checkpoint {checkpoint_path} already exists. Overwriting..."
                )
                os.makedirs(checkpoint_path, exist_ok=True)

            try:
                loaded_model = self.load_models(checkpoint)
                torch.save(loaded_model, checkpoint_path / "model.pt")
            except Exception as e:
                logger.error(f"Failed to save model at checkpoint {checkpoint}: {e}")
                raise

        self.module_keys = [m[0] for m in loaded_model.named_modules()]  # type: ignore

        logger.info(f"Saved models to {model_dir}")
        return model_dir

    def load_checkpoint(self, checkpoint: int, device: str = "cpu") -> torch.nn.Module:
        """
        Load a saved checkpoint.

        Args:
            model_name: Name of the model
            checkpoint_name: Specific checkpoint file name (if None, loads latest)

        Returns:
            Model loaded from the checkpoint
        """
        model_checkpoint_dir = (
            self.checkpoints_dir
            / self.model_name
            / f"checkpoint_{checkpoint}"
            / "model.pt"
        )

        try:
            model = torch.load(
                model_checkpoint_dir, map_location=device, weights_only=False
            )  # Alternative way to load model?
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_checkpoint_dir}: {e}")
            raise


class PythiaCheckpoints(ModelCheckpointManager):
    """
    Checkpoint manager for Pythia models.
    """

    def __init__(
        self,
        checkpoint_list: List[int],
        model_name: str,
        checkpoints_dir: str = "./checkpoints",
    ):
        super().__init__(checkpoint_list, model_name, checkpoints_dir)

    def load_models(self, checkpoint: int) -> torch.nn.Module:
        """
        Load a Pythia model from a specific checkpoint.
        """

        model = GPTNeoXForCausalLM.from_pretrained(
            self.model_name,
            revision=f"step{checkpoint}",
            device_map="cpu",
        )

        return model


if __name__ == "__main__":
    checkpoint_list = [2000, 3000, 4000]
    model_name = "EleutherAI/pythia-14m"
    checkpoints_dir = "./checkpoints"

    pythia_manager = PythiaCheckpoints(checkpoint_list, model_name, checkpoints_dir)
    pythia_manager.save_models(overwrite=True)
