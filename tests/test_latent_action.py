import numpy as np
import torch

from eval.controllability import controllability_probe, lam_sweep_grid
from models.latent_action import LatentActionModel
from models.tokenizer import img_to_float


def make_pair_batch(n=24, size=64, seed=0):
    """Pairs where the square moves left/right/up — 3 true actions to discover."""
    rng = np.random.default_rng(seed)
    moves = [(-6, 0), (6, 0), (0, -6)]
    x_t = np.zeros((n, size, size, 3), np.uint8)
    x_tp1 = np.zeros_like(x_t)
    labels = []
    for i in range(n):
        gx, gy = np.meshgrid(np.linspace(0, 150, size), np.linspace(0, 150, size))
        bg = np.stack([gx, gy, np.full_like(gx, 60)], -1).astype(np.uint8)
        px, py = rng.integers(12, size - 20, 2)
        m = i % 3
        dx, dy = moves[m]
        labels.append(m)
        for arr, (x, y) in ((x_t[i], (px, py)), (x_tp1[i], (px + dx, py + dy))):
            arr[:] = bg
            arr[y : y + 8, x : x + 8] = [255, 40, 40]
    return (
        img_to_float(torch.from_numpy(x_t)),
        img_to_float(torch.from_numpy(x_tp1)),
        torch.tensor(labels),
    )


def test_shapes_and_ranges():
    lam = LatentActionModel(num_codes=6, code_dim=16, base_ch=16)
    x_t, x_tp1, _ = make_pair_batch(8)
    pred, idx, vq_loss = lam(x_t, x_tp1)
    assert pred.shape == x_t.shape
    assert idx.shape == (8,) and idx.min() >= 0 and idx.max() < 6
    assert vq_loss.ndim == 0 and torch.isfinite(vq_loss)
    assert lam.encode_action(x_t, x_tp1).shape == (8,)
    assert lam.decode_with_code(x_t, 0).shape == x_t.shape


def test_learns_actions_without_labels():
    torch.manual_seed(0)
    lam = LatentActionModel(num_codes=6, code_dim=16, base_ch=16)
    x_t, x_tp1, _ = make_pair_batch(24)
    opt = torch.optim.AdamW(lam.parameters(), lr=1e-3)
    first = None
    for _ in range(80):
        pred, idx, vq_loss = lam(x_t, x_tp1)
        loss = torch.nn.functional.mse_loss(pred, x_tp1) + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.5 * first, f"no convergence: {first:.4f} -> {loss.item():.4f}"
    lam.eval()
    assert lam.encode_action(x_t, x_tp1).unique().numel() >= 2, "codebook collapsed"


def test_probe_and_sweep_grid():
    lam = LatentActionModel(num_codes=4, code_dim=16, base_ch=16).eval()
    x_t, _, _ = make_pair_batch(6)
    grid = lam_sweep_grid(lam, x_t)
    assert grid.shape == (6 * 64, (4 + 1) * 64, 3) and grid.dtype == np.uint8
    probe = controllability_probe(lam, x_t)
    assert set(probe) == {"within_code_cosine", "between_code_cosine", "controllability_gap"}
    assert all(np.isfinite(v) for v in probe.values())
