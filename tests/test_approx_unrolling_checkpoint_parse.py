"""Regression tests for checkpoint-name handling in the SOURCE
(approx-unrolling) pipeline.
"""

import pytest

from bergson.approx_unrolling.approx_unrolling_math import (
    _checkpoint_step,
    compute_lr_times_steps_per_segment,
)
from bergson.config import ApproxUnrollingConfig


def test_checkpoint_step_parses_hf_trainer_dirs():
    """``checkpoint-<N>`` dir names (and local paths ending in one) parse to N."""
    assert _checkpoint_step("checkpoint-1000") == 1000
    assert _checkpoint_step("/path/to/run/checkpoint-42") == 42
    # A bare digit-only step directory name is also accepted.
    assert _checkpoint_step("2000") == 2000
    assert _checkpoint_step("/path/to/run/1500") == 1500


def test_checkpoint_step_non_checkpoint_name_raises_valueerror():
    """A non-``checkpoint-N`` name raises a clear ValueError, not AttributeError."""
    with pytest.raises(ValueError) as excinfo:
        _checkpoint_step("EleutherAI/pythia-14m")
    # Error must be actionable: point the user at the explicit config knobs.
    msg = str(excinfo.value)
    assert "lr_list" in msg and "step_size_list" in msg


def test_compute_lr_times_steps_hf_id_does_not_attributeerror():
    """The shipped example (HF model ids, no lr/step lists) must not crash with
    AttributeError; it should raise an actionable ValueError instead."""
    cfg = ApproxUnrollingConfig(
        checkpoints=["EleutherAI/pythia-14m"],
        segments=1,
    )
    with pytest.raises(ValueError):
        compute_lr_times_steps_per_segment(cfg)


def test_compute_lr_times_steps_explicit_lists_bypasses_name_parsing():
    """When lr_list/step_size_list are set, name parsing is skipped entirely, so
    non-``checkpoint-N`` checkpoints work end-to-end."""
    cfg = ApproxUnrollingConfig(
        checkpoints=["EleutherAI/pythia-14m", "EleutherAI/pythia-14m"],
        segments=2,
        lr_list=[1e-5, 2e-5],
        step_size_list=[3, 4],
    )
    assert compute_lr_times_steps_per_segment(cfg) == [1e-5 * 3, 2e-5 * 4]
