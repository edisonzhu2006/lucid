"""Frechet Video Distance over I3D features (rollout quality).

The Frechet math is dependency-free and unit-tested; feature extraction uses
the torchscript I3D from the StyleGAN-V port of the original TF-FVD network
(downloaded on first use — cluster runs only).
"""

import numpy as np
import torch

I3D_URL = (
    "https://github.com/universome/stylegan-v/releases/download/fvd/i3d_torchscript.pt"
)


def gaussian_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """||mu1-mu2||^2 + Tr(s1 + s2 - 2 sqrt(s1 s2)), scipy-free via eigh."""
    diff = mu1 - mu2
    # sqrtm(s1) via symmetric eigendecomposition
    vals, vecs = np.linalg.eigh(sigma1)
    s1h = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    m = s1h @ sigma2 @ s1h
    covmean_trace = np.sqrt(np.clip(np.linalg.eigvalsh(m), 0, None)).sum()
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * covmean_trace)


@torch.no_grad()
def i3d_features(videos: np.ndarray, device="cuda", batch_size: int = 8) -> np.ndarray:
    """videos uint8 (N, T, H, W, 3), T >= 9 -> (N, 400) I3D features."""
    import torch.hub

    path = torch.hub.get_dir() + "/i3d_torchscript.pt"
    try:
        i3d = torch.jit.load(path).eval().to(device)
    except (RuntimeError, FileNotFoundError, ValueError):
        torch.hub.download_url_to_file(I3D_URL, path)
        i3d = torch.jit.load(path).eval().to(device)
    feats = []
    for i in range(0, len(videos), batch_size):
        v = torch.from_numpy(videos[i : i + batch_size]).to(device)
        v = v.permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # (B, C, T, H, W)
        v = torch.nn.functional.interpolate(
            v, size=(v.shape[2], 224, 224), mode="trilinear", align_corners=False
        )
        feats.append(i3d(v, rescale=False, resize=False, return_features=True).cpu().numpy())
    return np.concatenate(feats)


def fvd(real: np.ndarray, fake: np.ndarray, device="cuda") -> float:
    """FVD between two uint8 video arrays (N, T, H, W, 3)."""
    return frechet_distance(
        *gaussian_stats(i3d_features(real, device)),
        *gaussian_stats(i3d_features(fake, device)),
    )
