"""Stage 4: play the learned world model.

Keyboard keys 1..K inject latent action codes into the dynamics rollout and
the decoded frame renders live (pygame). Once the Stage-2 sweep tells you
which code means left/right/jump, pass --key-map to bind arrows.

Interactive:  python -m interactive.play --tokenizer results/tokenizer/tokenizer.pt \
                  --dynamics results/dynamics/dynamics.pt --data-dir datasets/coinrun/train \
                  --key-map "left=3,right=1,up=5" --record play.gif
Headless GIF: add --headless --steps 64 (random or --actions "1,1,5,3,...")
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from models.dynamics_transformer import DynamicsTransformer
from models.tokenizer import Tokenizer, float_to_img, img_to_float
from train import common


def load_models(tokenizer_ckpt: str, dynamics_ckpt: str, device) -> tuple[Tokenizer, DynamicsTransformer]:
    tok_ckpt = common.load_checkpoint(tokenizer_ckpt, map_location=device)
    tokenizer = Tokenizer(
        tuple(tok_ckpt["cfg"]["model"]["levels"]), tok_ckpt["cfg"]["model"]["base_ch"]
    ).to(device)
    tokenizer.load_state_dict(tok_ckpt["model"])
    tokenizer.eval().requires_grad_(False)

    dyn_ckpt = common.load_checkpoint(dynamics_ckpt, map_location=device)
    sd, mcfg = dyn_ckpt["model"], dyn_ckpt["cfg"]["model"]
    dynamics = DynamicsTransformer(
        vocab_size=sd["token_emb.weight"].shape[0] - 1,
        num_actions=sd["action_emb.weight"].shape[0] - 1,
        tokens_per_frame=sd["spatial_emb"].shape[2],
        context_frames=sd["temporal_emb"].shape[1] - 1,
        dim=mcfg["dim"],
        depth=mcfg["depth"],
        heads=mcfg["heads"],
    ).to(device)
    dynamics.load_state_dict(sd)
    dynamics.eval().requires_grad_(False)
    return tokenizer, dynamics


class WorldModelSession:
    """Maintains rolling token context; step(code) dreams the next frame."""

    def __init__(self, tokenizer, dynamics, init_frames: np.ndarray,
                 decode_steps: int = 8, temperature: float = 1.0, device="cpu"):
        self.tokenizer, self.dynamics, self.device = tokenizer, dynamics, device
        self.decode_steps, self.temperature = decode_steps, temperature
        c = dynamics.context_frames
        frames = torch.from_numpy(init_frames[-c:]).to(device)
        if frames.shape[0] < c:  # bootstrap: repeat the oldest frame
            pad = frames[:1].expand(c - frames.shape[0], -1, -1, -1)
            frames = torch.cat([pad, frames])
        tokens = tokenizer.encode_indices(img_to_float(frames)).flatten(1)
        self.ctx = deque([t for t in tokens], maxlen=c)
        self.ctx_actions = deque([dynamics.null_action] * c, maxlen=c)

    @torch.no_grad()
    def step(self, code: int) -> np.ndarray:
        """Advance the dream by one latent action -> uint8 (H, W, 3) frame."""
        ctx = torch.stack(list(self.ctx))[None]
        ctx_a = torch.tensor([list(self.ctx_actions)], device=self.device)
        action = torch.tensor([code], device=self.device)
        nxt = self.dynamics.generate(
            ctx, ctx_a, action, steps=self.decode_steps, temperature=self.temperature
        )
        self.ctx.append(nxt[0])
        self.ctx_actions.append(code)
        side = int(nxt.shape[-1] ** 0.5)
        img = self.tokenizer.decode_indices(nxt.reshape(1, side, side)).clamp(-1, 1)
        return float_to_img(img)[0].cpu().numpy()


def init_frames_from_shard(data_dir: str, context: int, seed: int = 0) -> np.ndarray:
    shards = sorted(Path(data_dir).glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {data_dir}")
    rng = np.random.default_rng(seed)
    with np.load(shards[rng.integers(len(shards))]) as z:
        start = rng.integers(0, max(1, len(z["frames"]) - context))
        return z["frames"][start : start + context]


def run_headless(session, num_actions, args) -> list[np.ndarray]:
    rng = np.random.default_rng(args.seed)
    if args.actions:
        codes = [int(a) for a in args.actions.split(",")]
    else:
        codes = rng.integers(0, num_actions, args.steps).tolist()
    frames = [session.step(c) for c in codes]
    print(f"dreamed {len(frames)} frames")
    return frames


def run_interactive(session, num_actions, key_map: dict, args) -> list[np.ndarray]:
    import pygame

    scale = args.scale
    pygame.init()
    screen = pygame.display.set_mode((64 * scale, 64 * scale))
    pygame.display.set_caption("lucid — playing the dream (1..K codes, ESC quits)")
    clock = pygame.time.Clock()
    frames, code = [], 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
        pressed = pygame.key.get_pressed()
        for k in range(min(num_actions, 9)):
            if pressed[pygame.K_1 + k]:
                code = k
        for name, mapped in key_map.items():
            if pressed[getattr(pygame, f"K_{name}")]:
                code = mapped
        frame = session.step(code)
        frames.append(frame)
        surf = pygame.surfarray.make_surface(np.repeat(np.repeat(frame, scale, 0), scale, 1).swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="Play the learned world model")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--data-dir", required=True, help="shards for the seed frames")
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--key-map", default="", help='e.g. "left=3,right=1,up=5"')
    p.add_argument("--record", default="", help="write a GIF here on exit")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=64, help="headless: number of steps")
    p.add_argument("--actions", default="", help='headless: e.g. "1,1,5,3"')
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, dynamics = load_models(args.tokenizer, args.dynamics, device)
    init = init_frames_from_shard(args.data_dir, dynamics.context_frames, args.seed)
    session = WorldModelSession(
        tokenizer, dynamics, init, args.decode_steps, args.temperature, device
    )
    key_map = dict(
        (k, int(v)) for k, v in (kv.split("=") for kv in args.key_map.split(",") if kv)
    )
    if args.headless:
        frames = run_headless(session, dynamics.null_action, args)
    else:
        frames = run_interactive(session, dynamics.null_action, key_map, args)
    if args.record and frames:
        import imageio.v3 as iio

        iio.imwrite(args.record, np.stack(frames), duration=1000 // args.fps, loop=0)
        print(f"wrote {args.record}")


if __name__ == "__main__":
    main()
