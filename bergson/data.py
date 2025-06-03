import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class MemmapDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""

    def __init__(
        self,
        data_path: str,
        ctx_len: int,
        max_examples: int | None = None,
        dtype=np.uint16,
    ):
        mmap = np.memmap(data_path, dtype=dtype, mode="r").reshape(-1, ctx_len)
        self.mmap = mmap[:max_examples]

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return dict(input_ids=torch.from_numpy(self.mmap[idx].astype(np.int64)))

    def select(self, rng: range) -> "MemmapDataset":
        """Select a subset of the dataset."""
        mmap = MemmapDataset.__new__(MemmapDataset)
        mmap.mmap = self.mmap[rng.start : rng.stop]
        return mmap

    def shard(self, num_shards: int, shard_id: int) -> "MemmapDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        mmap = MemmapDataset.__new__(MemmapDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap


def compute_batches(lengths, max_tokens: int):
    """Split a list of lengths into batches that do not exceed `max_tokens`."""
    start = 0
    tokens_in_batch = 0
    batches = []

    for idx, length in enumerate(lengths):
        # Would adding this `length` exceed the capacity?
        if tokens_in_batch + length > max_tokens:
            # Close the previous batch: slice(start, idx)
            batches.append(slice(start, idx))

            # Start a new batch _with_ this item
            start = idx
            tokens_in_batch = length
        else:
            # It fits, so accumulate and keep going
            tokens_in_batch += length

    return batches


def pad_and_tensor(
    sequences: list[list[int]],
    labels: list[list[int]] | None = None,
    *,
    padding_value: int = 0,
    dtype: torch.dtype | None = torch.long,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length and convert them to tensors.
    Returns a tuple of padded sequences and labels. The labels are the same as the
    sequences, but with -100 for the padding positions, which is useful for ignoring
    padding in loss calculations.
    """
    if labels is None:
        labels = sequences

    # find max length
    max_len = max(len(seq) for seq in sequences)
    # pad each sequence
    padded = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    labels = [label + [-100] * (max_len - len(label)) for label in labels]

    # convert to tensor
    padded_tokens = torch.tensor(padded, dtype=dtype, device=device)
    padded_labels = torch.tensor(labels, dtype=dtype, device=device)
    return padded_tokens, padded_labels
