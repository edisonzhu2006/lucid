import torch

from models.dynamics_transformer import DynamicsTransformer


def tiny_model(vocab=32, actions=4, n=16, ctx=3):
    torch.manual_seed(0)
    return DynamicsTransformer(
        vocab_size=vocab, num_actions=actions, tokens_per_frame=n,
        context_frames=ctx, dim=64, depth=2, heads=4,
    )


def make_batch(b=16, f=4, n=16, vocab=32, actions=4):
    """Deterministic toy dynamics: next frame = prev rolled by (action+1)."""
    torch.manual_seed(1)
    tokens = torch.zeros(b, f, n, dtype=torch.long)
    tokens[:, 0] = torch.randint(0, vocab, (b, n))
    acts = torch.randint(0, actions, (b, f))
    acts[:, 0] = actions  # null action for frame 0
    for t in range(1, f):
        shift = acts[:, t] + 1
        for i in range(b):
            tokens[i, t] = torch.roll(tokens[i, t - 1], int(shift[i]))
    return tokens, acts


def test_forward_and_loss_shapes():
    model = tiny_model()
    tokens, acts = make_batch()
    logits = model(tokens, acts)
    assert logits.shape == (16, 4, 16, 32)
    out = model.training_loss(tokens, acts)
    assert torch.isfinite(out["loss"]) and 0 <= out["masked_acc"] <= 1


def test_generate_and_rollout_shapes():
    model = tiny_model().eval()
    tokens, acts = make_batch()
    nxt = model.generate(tokens[:, :3], acts[:, :3], acts[:, 3], steps=4)
    assert nxt.shape == (16, 16)
    assert nxt.max() < 32 and (nxt != model.mask_id).all()
    seq = model.rollout(tokens[:, :3], acts[:, :3], torch.randint(0, 4, (16, 5)), steps=2)
    assert seq.shape == (16, 5, 16) and (seq != model.mask_id).all()


def test_learns_toy_dynamics():
    model = tiny_model()
    tokens, acts = make_batch(b=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        out = model.training_loss(tokens, acts)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        first = first or out["loss"].item()
    assert out["loss"].item() < 0.5 * first, (
        f"no convergence: {first:.3f} -> {out['loss'].item():.3f}"
    )
    # action-conditioning check: generated frame should match the rolled target
    # better than chance for the trained toy rule
    model.eval()
    nxt = model.generate(tokens[:, :3], acts[:, :3], acts[:, 3], steps=8, temperature=1e-4)
    acc = (nxt == tokens[:, 3]).float().mean().item()
    assert acc > 0.2, f"generation ignores dynamics: acc={acc:.3f} (chance ~{1 / 32:.3f})"
