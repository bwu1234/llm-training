"""Memory-mapped next-token windows over a prepared token file (`uint16` or `uint32`)."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    """The same windows as `SlidingWindowTokenDataset`, read lazily from disk.

    Window `i` starts at `i * stride`; targets are the inputs shifted by one token. The
    operating system pages tokens in on demand, so memory use is independent of file size.
    """

    def __init__(self, path, max_length, stride, dtype="uint16"):
        self.tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.max_length = max_length
        self.starts = range(0, len(self.tokens) - max_length, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        window = torch.from_numpy(self.tokens[start : start + self.max_length + 1].astype(np.int64))
        return window[:-1], window[1:]
