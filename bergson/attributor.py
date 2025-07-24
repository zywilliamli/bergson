import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Generator

import faiss
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray
from torch import Tensor, nn
from tqdm import tqdm

from .data import load_unstructured_gradients
from .gradients import GradientCollector, GradientProcessor


class TraceResult:
    """Result of a .trace() call."""

    def __init__(self):
        # Should be set by the Attributor after a search
        self._indices: Tensor | None = None
        self._scores: Tensor | None = None

    @property
    def indices(self) -> Tensor:
        """The indices of the top-k examples."""
        if self._indices is None:
            raise ValueError("No indices available. Exit the context manager first.")

        return self._indices

    @property
    def scores(self) -> Tensor:
        """The attribution scores of the top-k examples."""
        if self._scores is None:
            raise ValueError("No scores available. Exit the context manager first.")

        return self._scores


def normalize_grads(
    grads: NDArray,
    device: torch.device | str,
    batch_size: int,
) -> NDArray:
    normalized_grads = np.zeros_like(grads).astype(grads.dtype)

    for i in tqdm(range(0, grads.shape[0], batch_size)):
        batch = torch.from_numpy(grads[i : i + batch_size]).to(device)
        normalized_grads[i : i + batch_size] = (
            (batch / batch.norm(dim=1, keepdim=True)).cpu().numpy()
        )

    return normalized_grads


def gradients_loader(root_dir: str):
    def load_shard(shard_dir: str) -> np.memmap:
        print("shard dir", shard_dir)
        with open(os.path.join(shard_dir, "info.json")) as f:
            info = json.load(f)

        if "grad_size" in info:
            return load_unstructured_gradients(shard_dir)

        dtype = info["dtype"]
        num_grads = info["num_grads"]

        return np.memmap(
            os.path.join(shard_dir, "gradients.bin"),
            dtype=dtype,
            mode="r",
            shape=(num_grads,),
        )

    root_path = Path(root_dir)
    if (root_path / "info.json").exists():
        yield load_shard(root_dir)
    else:
        for shard_path in sorted(root_path.iterdir()):
            if shard_path.is_dir():
                yield load_shard(str(shard_path))


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = True,
        batch_size: int = 1024,
        faiss_cfg: str = "Flat",
    ):
        """
        [Guidelines on building your FAISS configuration string](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

        Common configurations:
        - "Flat": exact nearest neighbors with brute force search.
        - "PQ6720": nearest neighbors with vector product quantization to 6720 elements.
            Reduces memory usage.
        - "IVF1024,Flat": approximate nearest neighbors with IVF1024 clustering.
            Enables faster queries at the cost of a slower initial index build.

        GPU indexes will be sharded across GPUs.
        """

        path = (
            Path("runs/faiss")
            / Path(index_path).stem
            / f"{faiss_cfg.replace(',', '_')}.index"
        )
        path.parent.mkdir(exist_ok=True, parents=True)

        if path.exists():
            index = faiss.read_index(str(path))
        else:
            print("Building FAISS index...")
            start = time()
            index = None

            dl = gradients_loader(index_path)

            for grads in dl:
                if grads.dtype.names is not None:
                    grads = structured_to_unstructured(grads)

                np_dtype = np.array(torch.tensor([], dtype=dtype)).dtype
                grads = grads.astype(np_dtype)

                if unit_norm:
                    grads = normalize_grads(grads, device, batch_size)

                if index is None:
                    index = faiss.index_factory(
                        grads.shape[1], faiss_cfg, faiss.METRIC_INNER_PRODUCT
                    )

                    if device != "cpu":
                        gpus = (
                            list(range(torch.cuda.device_count()))
                            if device == "cuda"
                            else [int(device.split(":")[1])]
                        )

                        options = faiss.GpuMultipleClonerOptions()
                        options.shard = True
                        index = faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

                    index.train(grads)

                index.add(grads)

            print(
                f"Built index in {(time() - start) / 60:.2f} minutes."
                f"Saving to {path}..."
            )

            faiss.write_index(faiss.index_gpu_to_cpu(index), str(path))
            print("Saved index.")

        self.index = index
        self.device = device
        self.dtype = dtype
        self.faiss_cfg = faiss_cfg

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(
        self, queries: Tensor, k: int, nprobe: int = 10
    ) -> tuple[Tensor, Tensor]:
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        If fewer than `k` examples are found FAISS will return items with the index -1
        and the maximum negative distance.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            nprobe: The number of FAISS vector clusters to search if using ANN.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        q = queries / queries.norm(dim=1, keepdim=True)
        q = q.to("cpu", non_blocking=True).numpy()

        self.index.nprobe = nprobe

        distances, indices = self.index.search(q, k)

        return torch.from_numpy(distances), torch.from_numpy(indices)

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int,
        *,
        precondition: bool = False,
        unit_norm: bool = True,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads: list[Tensor] = []
        result = TraceResult()

        def callback(name: str, g: Tensor):
            # Precondition the gradient using Cholesky solve
            if precondition:
                P = self.processor.preconditioners[name]
                g = g.flatten(1).type_as(P)
                g = torch.cholesky_solve(g.mT, P).mT
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads.append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = torch.cat(mod_grads, dim=1)

        if unit_norm:
            queries /= queries.norm(dim=1, keepdim=True)

        result._scores, result._indices = self.search(queries, k)
