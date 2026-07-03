"""Stage-2 gate: does each latent action code do one consistent thing?

The sweep figure (rows = start frames, columns = codes) is the project's key
portfolio visual. The probe quantifies it: per-code frame deltas should agree
across start frames (within-code cosine high) and differ across codes
(between-code cosine low).
"""

import numpy as np
import torch
import torch.nn.functional as F

from models.tokenizer import float_to_img


@torch.no_grad()
def lam_sweep_grid(lam, x_t: torch.Tensor) -> np.ndarray:
    """Sweep every code over start frames x_t (B, 3, H, W) -> HWC uint8 grid.

    Column 0 is the start frame, columns 1..K the predicted next frame per code.
    """
    cols = [float_to_img(x_t).cpu().numpy()]
    for k in range(lam.num_codes):
        pred = lam.decode_with_code(x_t, k).clamp(-1, 1)
        cols.append(float_to_img(pred).cpu().numpy())
    rows = [np.concatenate([col[b] for col in cols], axis=1) for b in range(x_t.shape[0])]
    return np.concatenate(rows, axis=0)


def save_sweep_figure(lam, x_t: torch.Tensor, path) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, lam_sweep_grid(lam, x_t))


@torch.no_grad()
def controllability_probe(lam, x_t: torch.Tensor) -> dict[str, float]:
    """-> within-code / between-code delta cosine similarity + their gap."""
    deltas = []  # (K, B, D) flattened frame deltas per code
    for k in range(lam.num_codes):
        pred = lam.decode_with_code(x_t, k)
        deltas.append((pred - x_t).flatten(1))
    d = F.normalize(torch.stack(deltas), dim=-1)  # (K, B, D)
    within = torch.einsum("kbd,kcd->kbc", d, d)
    b = x_t.shape[0]
    off = ~torch.eye(b, dtype=torch.bool, device=d.device)
    within_score = within[:, off].mean().item()
    centroids = F.normalize(d.mean(1), dim=-1)  # (K, D)
    between = centroids @ centroids.T
    k = lam.num_codes
    off_k = ~torch.eye(k, dtype=torch.bool, device=d.device)
    between_score = between[off_k].mean().item()
    return {
        "within_code_cosine": within_score,
        "between_code_cosine": between_score,
        "controllability_gap": within_score - between_score,
    }
