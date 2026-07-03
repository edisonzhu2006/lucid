"""Stage 0: collect CoinRun frames with a random + lightly-scripted policy.

Frames are the world model's only training signal. True actions and rewards
are saved into the same shards but are reserved for evaluation and baselines —
the world model must never see them.

Shard format (one .npz per env per chunk_len steps):
    frames  uint8   (T, 64, 64, 3)
    actions int32   (T,)   action taken at t (evaluation only)
    rewards float32 (T,)   reward received after acting at t (evaluation only)
    firsts  bool    (T,)   True where frame t starts a new episode

Run:  python -m data.collect            (defaults from configs/collect.yaml)
      python -m data.collect total_steps=100000 num_envs=8   # tiny debug run
"""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# indices into procgen's fixed 15-action combo list
NOOP, LEFT, LEFT_JUMP, JUMP, RIGHT, RIGHT_JUMP = 4, 1, 2, 5, 7, 8
NUM_ACTIONS = 15


def sample_actions(rng: np.random.Generator, n: int, p: DictConfig) -> np.ndarray:
    choices = np.array([RIGHT, RIGHT_JUMP, JUMP, LEFT, NOOP, -1])
    probs = np.array([p.p_right, p.p_right_jump, p.p_jump, p.p_left, p.p_noop, p.p_random])
    acts = rng.choice(choices, size=n, p=probs / probs.sum())
    random_mask = acts == -1
    acts[random_mask] = rng.integers(0, NUM_ACTIONS, random_mask.sum())
    return acts.astype(np.int32)


class ShardWriter:
    def __init__(self, out_dir: Path, seed: int, compress: bool):
        self.out_dir = out_dir
        self.seed = seed
        self.save = np.savez_compressed if compress else np.savez
        self.counts = {}

    def flush(self, env_id: int, frames, actions, rewards, firsts) -> None:
        k = self.counts.get(env_id, 0)
        path = self.out_dir / f"seed{self.seed}_env{env_id:03d}_{k:06d}.npz"
        self.save(
            path,
            frames=np.asarray(frames, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            firsts=np.asarray(firsts, dtype=bool),
        )
        self.counts[env_id] = k + 1


@hydra.main(version_base=None, config_path="../configs", config_name="collect")
def main(cfg: DictConfig) -> None:
    from procgen import ProcgenGym3Env  # import here: not installable on Apple Silicon

    out_dir = Path(hydra.utils.get_original_cwd()) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
    )

    env = ProcgenGym3Env(
        num=cfg.num_envs,
        env_name=cfg.env_name,
        num_levels=cfg.num_levels,
        start_level=cfg.start_level,
        distribution_mode=cfg.distribution_mode,
        rand_seed=cfg.seed,
    )
    rng = np.random.default_rng(cfg.seed)
    writer = ShardWriter(out_dir, cfg.seed, cfg.compress)

    n = cfg.num_envs
    buf = {e: {"frames": [], "actions": [], "rewards": [], "firsts": []} for e in range(n)}

    _, obs, first = env.observe()
    steps_per_env = cfg.total_steps // n
    for t in range(steps_per_env):
        acts = sample_actions(rng, n, cfg.policy)
        frame, was_first = obs["rgb"], first
        env.act(acts)
        rew, obs, first = env.observe()
        for e in range(n):
            b = buf[e]
            b["frames"].append(frame[e])
            b["actions"].append(acts[e])
            b["rewards"].append(rew[e])
            b["firsts"].append(was_first[e])
            if len(b["frames"]) == cfg.chunk_len:
                writer.flush(e, **b)
                buf[e] = {"frames": [], "actions": [], "rewards": [], "firsts": []}
        if (t + 1) % 1000 == 0:
            print(f"step {t + 1}/{steps_per_env}  ({(t + 1) * n:,} frames)")

    for e in range(n):  # partial final chunks
        if buf[e]["frames"]:
            writer.flush(e, **buf[e])
    total = sum(writer.counts.values())
    print(f"done: {total} shards -> {out_dir}")


if __name__ == "__main__":
    main()
