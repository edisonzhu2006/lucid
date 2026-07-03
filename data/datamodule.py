"""Stage 0: stream frame sequences from .npz shards written by collect.py.

Yields fixed-length windows of consecutive frames that never cross an episode
boundary (or a shard boundary). Frames stay uint8 through the DataLoader —
normalize on the GPU, it's much cheaper than in the workers.

Stage-0 gate check (throughput benchmark):
    python -m data.datamodule --data-dir datasets/coinrun/train
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class FrameSequenceDataset(IterableDataset):
    """Streams (seq_len,) windows from shuffled shards with a mixing buffer.

    frames_only=True yields {"frames"}; otherwise adds {"actions", "rewards"}
    (evaluation/baseline use only — the world model must not see them).
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 16,
        stride: int | None = None,
        frames_only: bool = True,
        shuffle_buffer: int = 256,
        seed: int = 0,
    ):
        self.shards = sorted(Path(data_dir).glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {data_dir}")
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.frames_only = frames_only
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0

    def _windows(self, shard: Path, rng: np.random.Generator):
        with np.load(shard) as z:
            frames = z["frames"]
            firsts = z["firsts"]
            extras = None if self.frames_only else (z["actions"], z["rewards"])
        # episode segments: split at each first-frame marker
        starts = [0, *np.flatnonzero(firsts[1:]) + 1, len(frames)]
        index = []
        for lo, hi in zip(starts[:-1], starts[1:]):
            index.extend(range(lo, hi - self.seq_len + 1, self.stride))
        rng.shuffle(index)
        for i in index:
            item = {"frames": torch.from_numpy(frames[i : i + self.seq_len].copy())}
            if extras is not None:
                item["actions"] = torch.from_numpy(extras[0][i : i + self.seq_len].copy())
                item["rewards"] = torch.from_numpy(extras[1][i : i + self.seq_len].copy())
            yield item

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng((self.seed, self._epoch, wid))
        self._epoch += 1

        shards = list(self.shards)
        np.random.default_rng((self.seed, self._epoch)).shuffle(shards)
        shards = shards[wid::nw]

        # shuffle buffer: mixes windows across neighboring shards
        buf = []
        for shard in shards:
            for item in self._windows(shard, rng):
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    yield buf.pop(rng.integers(len(buf)))
        rng.shuffle(buf)
        yield from buf


def make_dataloader(
    data_dir: str | Path,
    batch_size: int = 32,
    seq_len: int = 16,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    dataset = FrameSequenceDataset(data_dir, seq_len=seq_len, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )


def _benchmark(args: argparse.Namespace) -> None:
    loader = make_dataloader(
        args.data_dir, args.batch_size, args.seq_len, args.num_workers
    )
    it = iter(loader)
    next(it)  # warm up workers
    t0, frames, nbytes, done, fresh_epoch = time.perf_counter(), 0, 0, 0, False
    while done < args.num_batches:
        try:
            batch = next(it)["frames"]
        except StopIteration:
            if fresh_epoch:
                raise RuntimeError("dataset too small for a single full batch")
            it, fresh_epoch = iter(loader), True
            continue
        fresh_epoch = False
        frames += batch.shape[0] * batch.shape[1]
        nbytes += batch.numel()
        done += 1
    dt = time.perf_counter() - t0
    print(
        f"{frames / dt:,.0f} frames/s  ({nbytes / dt / 1e6:,.0f} MB/s)  "
        f"[batch={args.batch_size} seq={args.seq_len} workers={args.num_workers}]"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage-0 gate: dataloader throughput")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=50)
    _benchmark(p.parse_args())
