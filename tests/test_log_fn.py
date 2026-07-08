from unittest.mock import MagicMock, patch

import pytest
import torch
import torchopt

from bergson.magic.data_stream import DataStream
from bergson.magic.trainer import Trainer
from bergson.utils.logging import wandb_log_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_log_fn_called_each_step(model, dataset):
    """log_fn is called once per training step with (step_idx, loss)."""
    model = model.to("cuda:0")
    opt = torchopt.adamw(1e-4)
    trainer, state = Trainer.initialize(model, opt)

    stream = DataStream(
        dataset,
        batch_size=2,
        device="cuda:0",
    )
    num_steps = len(stream)

    log = MagicMock()
    trainer.train(state, stream, inplace=True, log_fn=log)

    assert log.call_count == num_steps
    for i, call in enumerate(log.call_args_list):
        step, loss = call.args
        assert step == i
        assert isinstance(loss, float)


def test_wandb_log_fn_calls_wandb(monkeypatch):
    """wandb_log_fn initializes wandb and logs correctly."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    mock_wandb = MagicMock()
    mock_wandb.run = None
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log = wandb_log_fn("test-project", config={"lr": 1e-4})

        mock_wandb.init.assert_called_once_with(
            project="test-project", config={"lr": 1e-4}
        )

        log(5, 0.123)
        mock_wandb.log.assert_called_once_with({"train/loss": 0.123}, step=5)


def test_wandb_log_fn_reuses_existing_run(monkeypatch):
    """wandb_log_fn doesn't call init if a run already exists."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()  # pretend a run exists
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log = wandb_log_fn("test-project")

        mock_wandb.init.assert_not_called()

        log(0, 1.5)
        mock_wandb.log.assert_called_once_with({"train/loss": 1.5}, step=0)


def test_wandb_log_fn_noop_when_wandb_mode_disabled(monkeypatch):
    """With WANDB_MODE=disabled, wandb_log_fn is a no-op and never imports
    wandb. This mirrors a MAGIC run with wandb_project set but wandb either
    disabled or uninstalled: it must not crash post-training (bug repro
    bug4_magic_wandb.py)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    # Make any attempt to import wandb blow up, proving we short-circuit first.
    with patch.dict("sys.modules", {"wandb": None}):
        log = wandb_log_fn("my-magic-run", config={"lr": 1e-4})

        # Must return a callable no-op that does not raise.
        assert callable(log)
        assert log(0, 1.5) is None
        assert log(7, 0.123) is None


def test_wandb_log_fn_noop_when_wandb_missing(monkeypatch):
    """If wandb is not importable, wandb_log_fn degrades to a no-op with a
    warning instead of raising ModuleNotFoundError/ImportError."""
    # Exercise the import path (not the WANDB_MODE=disabled short-circuit).
    monkeypatch.delenv("WANDB_MODE", raising=False)
    # sys.modules["wandb"] = None makes `import wandb` raise ImportError.
    with patch.dict("sys.modules", {"wandb": None}):
        with pytest.warns(UserWarning, match="wandb is not installed"):
            log = wandb_log_fn("my-magic-run", config={"lr": 1e-4})

        assert callable(log)
        assert log(0, 1.5) is None


def test_wandb_log_fn_falls_back_when_init_fails(monkeypatch):
    """If wandb.init raises (e.g. dead daemon, bad auth, network), the caller
    gets a no-op log_fn and a warning, not an exception. Without this, a
    rank-0 wandb hang/failure deadlocks the rest of the world on the next
    distributed collective."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init.side_effect = TimeoutError("wandb-service did not respond")

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        with pytest.warns(UserWarning, match="wandb.init failed"):
            log = wandb_log_fn("test-project")

        # log_fn must be callable and a no-op (does not call wandb.log)
        log(0, 1.5)
        mock_wandb.log.assert_not_called()
