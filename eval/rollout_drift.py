"""Stage-3/6 gate metric: how fast do rollouts drift from reality?

Roll the model forward from a ground-truth context using the TRUE latent
action sequence (extracted by the LAM from the held-out video), then compare
imagined vs real frames per horizon step. Works on any (gt, pred) frame
arrays, so the transformer and diffusion ablations share the same curve.

CLI (token dynamics):
    python -m eval.rollout_drift --tokenizer <ckpt> --lam <ckpt> \
        --dynamics <ckpt> --data-dir datasets/coinrun/heldout --horizon 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.tokenizer import img_to_float
from train import common


def drift_curves(gt: np.ndarray, pred: np.ndarray) -> dict:
    """uint8 (B, T, H, W, 3) x2 -> per-step MSE and PSNR curves."""
    g = gt.astype(np.float64) / 127.5 - 1
    p = pred.astype(np.float64) / 127.5 - 1
    mse = ((g - p) ** 2).mean(axis=(0, 2, 3, 4))
    psnr = 10 * np.log10(4.0 / np.clip(mse, 1e-10, None))
    return {"mse": mse.tolist(), "psnr": psnr.tolist()}


def plot_drift(curves: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(curves["psnr"]) + 1), curves["psnr"])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("PSNR vs ground truth (dB)")
    ax.set_title("Long-horizon drift")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def token_dynamics_drift(tokenizer, lam, dynamics, shard_dir, horizon: int = 32,
                         num_windows: int = 16, decode_steps: int = 8,
                         device: str = "cpu", seed: int = 0) -> dict:
    from data.datamodule import FrameSequenceDataset
    from models.tokenizer import float_to_img
    from train.train_dynamics import encode_batch

    c = dynamics.context_frames
    ds = FrameSequenceDataset(shard_dir, seq_len=c + horizon, seed=seed)
    windows = []
    for item in ds:
        windows.append(item["frames"])
        if len(windows) >= num_windows:
            break
    frames = torch.stack(windows).to(device)  # (B, C+T, H, W, 3)
    tokens, actions = encode_batch(tokenizer, lam, frames, dynamics.null_action)
    gen = dynamics.rollout(
        tokens[:, :c], actions[:, :c], actions[:, c:], steps=decode_steps
    )
    b, t, n = gen.shape
    side = int(n**0.5)
    decoded = tokenizer.decode_indices(gen.reshape(b * t, side, side)).clamp(-1, 1)
    pred = float_to_img(decoded).cpu().numpy().reshape(b, t, *frames.shape[2:])
    gt = frames[:, c:].cpu().numpy()
    return drift_curves(gt, pred)


def main() -> None:
    p = argparse.ArgumentParser(description="Rollout drift vs horizon")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--lam", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--num-windows", type=int, default=16)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--out-dir", default="results/drift")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from interactive.play import load_models
    from models.latent_action import LatentActionModel
    from train.train_dynamics import load_frozen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, dynamics = load_models(args.tokenizer, args.dynamics, device)
    lam = load_frozen(args.lam, LatentActionModel, device=device)
    curves = token_dynamics_drift(
        tokenizer, lam, dynamics, args.data_dir, args.horizon,
        args.num_windows, args.decode_steps, device, args.seed,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.json").write_text(json.dumps(curves, indent=2))
    plot_drift(curves, out / "drift.png")
    print(f"wrote {out}/drift.json and drift.png "
          f"(final-step PSNR {curves['psnr'][-1]:.2f} dB)")


if __name__ == "__main__":
    main()
