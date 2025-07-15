from contextlib import contextmanager
from typing import Generator

import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch import Tensor, nn

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
        dtype: torch.dtype = torch.float16,
        use_faiss: bool = True,
        unit_norm: bool = True,
    ):
        # Map the gradients into memory (very fast)
        mmap = load_gradients(index_path)
        if mmap.dtype.names is not None:
            mmap = structured_to_unstructured(mmap)

        # Load them onto the desired device (slow)
        self.grads = torch.tensor(mmap, device=device, dtype=dtype)

        # In-place normalize for numerical stability
        if unit_norm:
            self.grads /= self.grads.norm(dim=1, keepdim=True)

        # Load gradients into a FAISS index for fast queries
        if use_faiss:
            import faiss
            import numpy as np
            from tqdm import tqdm
            from time import time
            from pathlib import Path


            # batch_size = 100_000 if len(mmap) > 100_000 else len(mmap)
            name = Path(index_path).stem

            Path(f'faiss/{name}').mkdir(exist_ok=True, parents=True)

            index = faiss.index_factory(mmap.shape[1], "IVF1024_HNSW32,Flat")
            print("Training FAISS index...")
            start = time()
            index.train(mmap)
            # index.train(mmap[:batch_size])
            # faiss.write_index(index, f"faiss/{name}/clusters.index")
            print(f"Built clusters index in {(time() - start) / 60} minutes")

            # n_batches = len(mmap) // batch_size
            # for i in tqdm(range(n_batches)):
            # index = faiss.read_index(f"faiss/{name}/clusters.index")
            # index.add_with_ids(
            #     mmap[i * batch_size : (i + 1) * batch_size],
            #     np.arange(i * batch_size, (i + 1) * batch_size),
            # )
            index.add_with_ids(
                mmap,
                np.arange(len(mmap)),
            )
            # print(f"writing block_{i}.index with {i*batch_size} as starting index")
            faiss.write_index(index, f"faiss/{name}/train.index")


        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(self, queries: Tensor, k: int) -> torch.return_types.topk:
        """
        Search for the `k` nearest examples in the index based on the query or queries.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        return torch.topk(queries @ self.grads.mT, k)

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int,
        *,
        precondition: bool = False,
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
            mod_grads.append(
                g.to(self.grads.device, self.grads.dtype, non_blocking=True)
            )

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = torch.cat(mod_grads, dim=1)
        result._scores, result._indices = self.search(queries, k)
