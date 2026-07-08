"""Regression test for the FAISS ANN off-the-shelf breakage.

`FaissIndex.__init__` reads each on-disk shard with ``IO_FLAG_MMAP`` (a CPU index).
When ``mmap_index=False`` (the shipped default) and ``device="cpu"``, the loader
used to call ``index_to_device(shard, "cpu")``, which *unconditionally* ran
``faiss.index_gpu_to_cpu`` on the already-CPU index, cloning it and raising
``RuntimeError: clone not supported ... OnDiskInvertedLists`` for any IVF/ANN index
mmap'd from disk. Only an exact ``Flat`` index survived. The CPU->GPU helper is now
``index_to_gpu`` and is skipped entirely when ``device="cpu"``.

These tests build tiny on-disk indices with ``device="cpu"`` and confirm the CPU
load path (``mmap_index=False``) no longer raises and returns usable neighbours for
both an ANN (IVF) index and an exact ``Flat`` index. Adapted from the bug2 repro.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bergson.config import FaissConfig
from bergson.query.faiss_index import FaissIndex


def _has_faiss() -> bool:
    try:
        import faiss  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


requires_faiss = pytest.mark.skipif(not _has_faiss(), reason="faiss not available")


def _write_gradient_store(root: Path, n: int, dim: int) -> np.ndarray:
    """Write a tiny on-disk gradient store matching bergson's info.json layout."""
    root.mkdir(parents=True, exist_ok=True)
    grad_sizes = {"module_a": dim}
    struct_dtype = {
        "names": list(grad_sizes),
        "formats": [f"({s},)<f4" for s in grad_sizes.values()],
        "itemsize": 4 * sum(grad_sizes.values()),
    }
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, dim)).astype("<f4")
    mm = np.memmap(
        root / "gradients.bin",
        dtype=np.dtype(struct_dtype),
        mode="w+",
        shape=(n,),
    )
    mm["module_a"] = data
    mm.flush()
    with (root / "info.json").open("w") as f:
        json.dump(
            {
                "num_grads": n,
                "dtype": struct_dtype,
                "grad_sizes": grad_sizes,
                "base_dtype": "float32",
            },
            f,
        )
    return data


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _build_and_load(
    grad_root: Path,
    idx_path: Path,
    index_factory: str,
    mmap_index: bool,
    query: np.ndarray,
):
    cfg = FaissConfig(
        index_factory=index_factory, mmap_index=mmap_index, num_shards=1
    )
    # unit_norm=True stores cosine-normalized gradients; with the inner-product
    # metric the self-match then has cosine 1.0, so a query row's own index is
    # deterministically its top neighbour (independent of vector magnitudes).
    FaissIndex.create_index(
        gradients_path=grad_root,
        faiss_path=idx_path,
        faiss_cfg=cfg,
        device="cpu",
        unit_norm=True,
        hessians={},
    )
    # This constructor is where the bug used to raise for an mmap'd IVF index.
    fi = FaissIndex(idx_path, device="cpu", mmap_index=mmap_index)
    distances, indices = fi.search(_l2_normalize(query), k=3)
    return distances, indices


@requires_faiss
def test_ann_ivf_cpu_load_does_not_raise(tmp_path: Path):
    """An IVF (ANN) index mmap'd from disk must load on CPU without raising."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    # device="cpu", mmap_index=False is the shipped default that used to crash.
    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_ivf",
        index_factory="IVF16,Flat",
        mmap_index=False,
        query=query,
    )

    assert indices.shape == (2, 3)
    # Each query vector is a stored gradient, so its own row should be the
    # top (self) match. IVF is approximate but with nprobe covering the tiny
    # index the exact self-match is recovered.
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    # Results must be real (not the -1 "not found" sentinel).
    assert (indices >= 0).all()


@requires_faiss
def test_exact_flat_cpu_load_still_works(tmp_path: Path):
    """The exact Flat path that already worked must keep working."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_flat",
        index_factory="Flat",
        mmap_index=False,
        query=query,
    )

    assert indices.shape == (2, 3)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    assert (indices >= 0).all()


@requires_faiss
def test_ann_ivf_mmap_index_true_still_works(tmp_path: Path):
    """The mmap_index=True path (skips index_to_gpu) must be unaffected."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_ivf_mmap",
        index_factory="IVF16,Flat",
        mmap_index=True,
        query=query,
    )

    assert indices.shape == (2, 3)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    assert (indices >= 0).all()
