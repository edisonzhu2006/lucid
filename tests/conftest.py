"""Shared fixtures: synthetic CoinRun-like shards (moving square on a gradient).

Compressible, deterministic frames so overfit smoke tests converge on CPU.
"""

import numpy as np
import pytest
import torch


def make_frames(t: int, size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(0, 200, size), np.linspace(0, 200, size))
    bg = np.stack([gx, gy, np.full_like(gx, 80.0)], axis=-1)
    frames = np.repeat(bg[None], t, axis=0)
    x, y, vx, vy = size // 2, size // 2, 3, 2
    for i in range(t):
        x, y = (x + vx) % (size - 8), (y + vy) % (size - 8)
        frames[i, y : y + 8, x : x + 8] = [255, 40, 40]
    frames += rng.normal(0, 2, frames.shape)
    return frames.clip(0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    for e in range(2):
        t = 200
        firsts = np.zeros(t, dtype=bool)
        firsts[[0, 90]] = True
        np.savez(
            out / f"seed0_env{e:03d}_000000.npz",
            frames=make_frames(t, seed=e),
            actions=rng.integers(0, 15, t).astype(np.int32),
            rewards=(rng.random(t) < 0.02).astype(np.float32) * 10.0,
            firsts=firsts,
        )
    return out


@pytest.fixture(scope="session")
def frame_batch() -> torch.Tensor:
    """(8, 64, 64, 3) uint8 batch for overfit smoke tests."""
    return torch.from_numpy(make_frames(8))
