import subprocess
from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocorrelation_hessian_e2e(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "hessian",
            "test_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--projection_dim",
            "4",
            "--token_batch_size",
            "1024",
            "--precision",
            "bf16",
            "--method",
            "autocorrelation",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert "Error" not in result.stderr, f"Error found in stderr:\n{result.stderr}"

    processor = GradientProcessor.load(tmp_path / "test_e2e")

    assert processor.hessians is not None
    assert processor.hessians_eigen is not None

    assert len(processor.hessians) > 0
    assert len(processor.hessians_eigen) > 0
