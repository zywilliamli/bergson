from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Generator

import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch import Tensor, nn
from tqdm import tqdm

import faiss

from .data import load_gradients
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


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        faiss_cfg: str = "IVF1024,Flat",
        unit_norm: bool = True,
    ):
        """
        [Guidelines on choosing your FAISS configuration](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

        Attributor will build an index even when its building time is greater
        than the expected direct query time to ensure query time speed.
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
            grads = load_gradients(index_path)

            if grads.dtype.names is not None:
                grads = structured_to_unstructured(grads)

            np_dtype = np.array(torch.tensor([], dtype=dtype)).dtype
            if unit_norm:
                preprocessed = np.zeros_like(grads).astype(np_dtype)
                batch_size = 1024
                for i in tqdm(range(0, grads.shape[0], batch_size)):
                    batch = torch.from_numpy(grads[i : i + batch_size]).to(device)
                    batch /= batch.norm(dim=1, keepdim=True)
                    preprocessed[i : i + batch_size] = batch.cpu().numpy()
            else:
                preprocessed = np.copy(grads).astype(np_dtype)

            # Load them onto the desired device (slow)
            index = faiss.index_factory(
                preprocessed.shape[1], faiss_cfg, faiss.METRIC_INNER_PRODUCT
            )

            if device != "cpu":
                index = faiss.index_cpu_to_all_gpus(index)

            print("Building FAISS index...")
            start = time()
            index.train(preprocessed)
            print(f"Built index in {(time() - start) / 60:.2f} minutes")

            index.add(preprocessed)
            faiss.write_index(faiss.index_gpu_to_cpu(index), str(path))

        self.faiss_index = index
        self.device = device
        self.dtype = dtype
        self.faiss_cfg = faiss_cfg

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(
        self, queries: Tensor, k: int, nprobe: int = 100
    ) -> tuple[Tensor, Tensor]:
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        If fewer than `k` examples are found FAISS will return items with the index -1
        and the maximum negative distance.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            nprobe: The number of FAISS vector clusters to search

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        self.faiss_index.nprobe = nprobe

        q = queries.to("cpu", non_blocking=True).numpy()

        distances, indices = self.faiss_index.search(q, k)

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
