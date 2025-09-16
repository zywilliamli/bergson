import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "disabled"

import tempfile
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments

from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.utils import assert_type


class TestGradientCollectorCallbackOrderTracking:
    """Test the GradientCollectorCallback with order tracking functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def model(self):
        """Create a small test model."""
        config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
        return AutoModelForCausalLM.from_config(config)

    @pytest.fixture
    def dataset(self):
        """Create a small test dataset."""
        data = {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            "labels": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
        }
        return Dataset.from_dict(data)

    def test_single_gpu_order_tracking(self, temp_dir, model, dataset):
        """Test that every step has an associated order record in single-GPU mode."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(temp_dir / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=str(temp_dir / "gradients"),
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that every step has associated order records
        steps_with_records = set()
        for record in callback.order:
            steps_with_records.add(record["global_step"])

        # Get the actual number of training steps
        expected_steps = len(dataset) // training_args.per_device_train_batch_size
        if len(dataset) % training_args.per_device_train_batch_size != 0:
            expected_steps += 1

        # Verify we have records for all expected steps
        assert len(steps_with_records) == expected_steps
        # Expected steps are 1-indexed
        assert steps_with_records == set(range(1, expected_steps + 1))

        # Verify each record has required fields
        for record in callback.order:
            assert "_idx" in record
            assert "global_step" in record
            assert "epoch" in record
            assert isinstance(record["_idx"], int)
            assert isinstance(record["global_step"], int)
            assert isinstance(record["epoch"], int)

        # Verify indices are within valid range
        for record in callback.order:
            assert 0 <= record["_idx"] < len(dataset)

    def test_order_tracking_disabled(self, temp_dir, model, dataset):
        """Test that no order records are created when tracking is disabled."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(temp_dir / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=str(temp_dir / "gradients"),
            track_order=False,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify no order records were created
        assert callback.order is None

    def test_order_save_and_load(self, temp_dir, model, dataset):
        """Test that order records are properly saved and can be loaded."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(temp_dir / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=str(temp_dir / "gradients"),
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that order file was saved
        order_file = temp_dir / "gradients" / "order.hf"
        assert order_file.exists()

        # Load and verify the saved order
        saved_order = Dataset.load_from_disk(str(order_file))
        assert len(saved_order) == len(callback.order)

        # Verify the saved order matches the in-memory order
        for i, record in enumerate(saved_order):
            record = assert_type(dict, record)
            assert record["_idx"] == callback.order[i]["_idx"]
            assert record["global_step"] == callback.order[i]["global_step"]
            assert record["epoch"] == callback.order[i]["epoch"]
