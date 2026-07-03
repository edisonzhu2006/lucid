"""Stage 5 evaluation — the money result.

Deploy the imagination-trained policy in REAL CoinRun and measure score /
completion vs a random floor. The agent picks latent codes; the Stage-2
correspondence (latent code <-> true action confusion matrix, built from
held-out labeled shards) maps them onto real controls.

Mapping only:   python -m eval.real_game_eval --lam <ckpt> --data-dir <shards>
Full eval:      add --tokenizer <ckpt> --agent <ckpt> --episodes 100   (needs procgen)
"""

import argparse
import json

import numpy as np
import torch

from models.latent_action import LatentActionModel
from models.tokenizer import Tokenizer, img_to_float
from train import common


@torch.no_grad()
def build_action_mapping(lam, shard_dir, num_real_actions: int = 15,
                         max_pairs: int = 20000, device: str = "cpu") -> dict:
    """Confusion of latent code vs true action on held-out labeled pairs.

    -> {mapping (K,), confusion (K, A), purity} — purity is the fraction of
    pairs whose latent code maps to their true action (upper-bounds transfer).
    """
    from pathlib import Path

    shards = sorted(Path(shard_dir).glob("*.npz"))
    confusion = np.zeros((lam.num_codes, num_real_actions), dtype=np.int64)
    seen = 0
    for shard in shards:
        if seen >= max_pairs:
            break
        with np.load(shard) as z:
            frames, actions, firsts = z["frames"], z["actions"], z["firsts"]
        take = min(len(frames) - 1, max_pairs - seen)
        valid = ~firsts[1 : take + 1]  # exclude episode-boundary transitions
        x = img_to_float(torch.from_numpy(frames[: take + 1]).to(device))
        codes = lam.encode_action(x[:-1], x[1:]).cpu().numpy()
        np.add.at(confusion, (codes[valid], actions[:take][valid]), 1)
        seen += int(valid.sum())
    mapping = confusion.argmax(axis=1)
    purity = confusion.max(axis=1).sum() / max(confusion.sum(), 1)
    return {"mapping": mapping, "confusion": confusion, "purity": float(purity)}


@torch.no_grad()
def evaluate_in_real_game(agent, tokenizer, mapping: np.ndarray, episodes: int = 100,
                          env_name: str = "coinrun", distribution_mode: str = "hard",
                          policy: str = "agent", seed: int = 0, device: str = "cpu") -> dict:
    """Run the policy in real procgen -> {mean_return, completion_rate, returns}.

    policy="random" gives the floor with the identical protocol.
    """
    from procgen import ProcgenGym3Env  # cluster / x86 only

    env = ProcgenGym3Env(
        num=1, env_name=env_name, num_levels=0,
        start_level=10_000_000,  # disjoint from training levels
        distribution_mode=distribution_mode, rand_seed=seed,
    )
    rng = np.random.default_rng(seed)
    returns, completions = [], []
    ep_ret = 0.0
    _, obs, _ = env.observe()
    while len(returns) < episodes:
        if policy == "random":
            code = rng.integers(len(mapping))
        else:
            frame = torch.from_numpy(obs["rgb"][0]).to(device)
            tokens = tokenizer.encode_indices(img_to_float(frame[None])).flatten(1)
            code = agent.policy(tokens).sample().item()
        env.act(np.array([mapping[code]], dtype=np.int32))
        rew, obs, first = env.observe()
        ep_ret += float(rew[0])
        if first[0]:
            returns.append(ep_ret)
            completions.append(ep_ret > 0)
            ep_ret = 0.0
    return {
        "mean_return": float(np.mean(returns)),
        "completion_rate": float(np.mean(completions)),
        "episodes": episodes,
        "returns": returns,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Real-game evaluation")
    p.add_argument("--lam", required=True, help="latent action model ckpt")
    p.add_argument("--data-dir", required=True, help="held-out labeled shards")
    p.add_argument("--tokenizer", default="", help="needed for --episodes > 0")
    p.add_argument("--agent", default="", help="imagination agent ckpt")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--out", default="results/real_game_eval.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = common.load_checkpoint(args.lam, map_location=device)
    lam = LatentActionModel(**ckpt["cfg"]["model"]).to(device)
    lam.load_state_dict(ckpt["model"])
    lam.eval()

    result = build_action_mapping(lam, args.data_dir, device=device)
    print(f"code -> real action mapping: {result['mapping'].tolist()}")
    print(f"mapping purity: {result['purity']:.3f}")
    out = {"mapping": result["mapping"].tolist(), "purity": result["purity"]}

    if args.episodes > 0:
        from behavior.imagination import ImaginationAgent

        tok_ckpt = common.load_checkpoint(args.tokenizer, map_location=device)
        tokenizer = Tokenizer(
            tuple(tok_ckpt["cfg"]["model"]["levels"]), tok_ckpt["cfg"]["model"]["base_ch"]
        ).to(device)
        tokenizer.load_state_dict(tok_ckpt["model"])
        tokenizer.eval()
        agent_ckpt = common.load_checkpoint(args.agent, map_location=device)
        agent = ImaginationAgent(**agent_ckpt["cfg"]["agent_kwargs"]).to(device)
        agent.load_state_dict(agent_ckpt["agent"])
        agent.eval()
        for policy in ("agent", "random"):
            out[policy] = evaluate_in_real_game(
                agent, tokenizer, result["mapping"], args.episodes,
                policy=policy, seed=args.seed, device=device,
            )
            print(f"{policy}: return={out[policy]['mean_return']:.2f} "
                  f"completion={out[policy]['completion_rate']:.1%}")

    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
