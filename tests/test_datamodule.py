import numpy as np
import torch

from data.datamodule import FrameSequenceDataset, make_dataloader


def test_windows_never_cross_episode_boundaries(shard_dir):
    ds = FrameSequenceDataset(shard_dir, seq_len=16, frames_only=False, seed=1)
    # reconstruct firsts per shard to check every yielded window
    firsts = {p: np.load(p)["firsts"] for p in ds.shards}
    frames = {p: np.load(p)["frames"] for p in ds.shards}
    n = 0
    for item in ds:
        assert item["frames"].shape == (16, 64, 64, 3)
        assert item["frames"].dtype == torch.uint8
        assert item["actions"].shape == (16,) and item["rewards"].shape == (16,)
        # locate the window in its source shard and check no interior episode start
        found = False
        w = item["frames"].numpy()
        for p, f in frames.items():
            for i in range(len(f) - 16 + 1):
                if np.array_equal(f[i : i + 16], w):
                    assert not firsts[p][i + 1 : i + 16].any()
                    found = True
                    break
            if found:
                break
        assert found
        n += 1
    assert n > 10


def test_dataloader_batches(shard_dir):
    loader = make_dataloader(shard_dir, batch_size=4, seq_len=8, num_workers=0)
    batch = next(iter(loader))
    assert batch["frames"].shape == (4, 8, 64, 64, 3)
    assert batch["frames"].dtype == torch.uint8
