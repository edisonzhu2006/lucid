import numpy as np
import torch

from behavior.imagination import (
    ImaginationAgent,
    ImaginationTrainer,
    RewardModel,
    lambda_returns,
)
from eval.real_game_eval import build_action_mapping
from models.dynamics_transformer import DynamicsTransformer
from models.latent_action import LatentActionModel


def test_lambda_returns_match_hand_computation():
    # H=2, gamma=0.9, lam=0.8; single batch element
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 1.0]])
    bootstrap = torch.tensor([3.0])
    # t=1: r=2 + 0.9*((1-0.8)*3 + 0.8*3) = 2 + 2.7 = 4.7
    # t=0: r=1 + 0.9*((1-0.8)*1.0 + 0.8*4.7) = 1 + 0.9*(0.2 + 3.76) = 4.564
    out = lambda_returns(rewards, values, bootstrap, gamma=0.9, lam=0.8)
    assert torch.allclose(out, torch.tensor([[4.564, 4.7]]), atol=1e-5)


def tiny_world(vocab=32, n=16, ctx=3, actions=4):
    torch.manual_seed(0)
    dynamics = DynamicsTransformer(
        vocab_size=vocab, num_actions=actions, tokens_per_frame=n,
        context_frames=ctx, dim=32, depth=1, heads=2,
    ).eval()
    reward = RewardModel(vocab, n, dim=16, state_dim=32, hidden=32).eval()
    agent = ImaginationAgent(vocab, actions, n, dim=16, state_dim=32, hidden=32)
    return dynamics, reward, agent


def test_imagination_update_runs_and_learns_something():
    dynamics, reward, agent = tiny_world()
    trainer = ImaginationTrainer(
        dynamics, reward, agent, horizon=4, decode_steps=1, device="cpu",
        actor_lr=1e-3, critic_lr=1e-3,
    )
    ctx = torch.randint(0, 32, (6, 3, 16))
    ctx_a = torch.randint(0, 4, (6, 3))
    before = [p.clone() for p in agent.actor.parameters()]
    for _ in range(3):
        metrics = trainer.update(ctx, ctx_a)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
    changed = any(
        not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())
    )
    assert changed, "actor parameters never updated"


def test_imagine_shapes():
    dynamics, reward, agent = tiny_world()
    trainer = ImaginationTrainer(dynamics, reward, agent, horizon=5, decode_steps=1)
    out = trainer.imagine(torch.randint(0, 32, (2, 3, 16)), torch.randint(0, 4, (2, 3)))
    assert out["states"].shape == (2, 5, 16)
    assert out["actions"].shape == (2, 5) and out["actions"].max() < 4
    assert out["rewards"].shape == (2, 5)
    assert out["final_state"].shape == (2, 16)


def test_build_action_mapping(shard_dir):
    torch.manual_seed(0)
    lam = LatentActionModel(num_codes=4, code_dim=16, base_ch=16).eval()
    out = build_action_mapping(lam, shard_dir, num_real_actions=15, max_pairs=100)
    assert out["mapping"].shape == (4,)
    assert (out["mapping"] >= 0).all() and (out["mapping"] < 15).all()
    assert 0.0 <= out["purity"] <= 1.0
    assert out["confusion"].sum() > 0
