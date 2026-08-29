import pytest

from bergson.config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.hessians.pipeline import hessian_pipeline


def test_hessian_pipeline_rejects_unit_normalize(tmp_path):
    """Cosine similarity (unit_normalize) is not supported with the
    Kronecker-factored Hessians hessian_pipeline fits and applies."""
    with pytest.raises(ValueError, match="unit_normalize"):
        hessian_pipeline(
            IndexConfig(run_path=str(tmp_path / "run")),
            HessianConfig(method="kfac"),
            ScoreConfig(),
            PreprocessConfig(unit_normalize=True),
            HessianPipelineConfig(),
        )
