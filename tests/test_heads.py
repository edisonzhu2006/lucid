import torch

from models.heads import DistHead, PolicyHead, StateEncoder, symexp, symlog, two_hot


def test_symlog_symexp_roundtrip():
    x = torch.tensor([-1000.0, -1.5, 0.0, 0.3, 42.0, 1e6])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4)


def test_two_hot_is_exact():
    bins = torch.linspace(-20, 20, 255)
    x = torch.randn(64) * 5
    t = two_hot(x, bins)
    assert torch.allclose(t.sum(-1), torch.ones(64), atol=1e-5)
    assert ((t > 0).sum(-1) <= 2).all()
    assert torch.allclose((t * bins).sum(-1), x, atol=1e-4)  # decodes exactly


def test_dist_head_zero_init_and_loss():
    head = DistHead(32, hidden=64)
    state = torch.randn(16, 32)
    logits = head(state)
    assert torch.allclose(head.mean(logits), torch.zeros(16), atol=1e-5)
    target = torch.rand(16) * 10
    loss = head.loss(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # a few grad steps should move predictions toward the target
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        head.loss(head(state), target).backward()
        opt.step()
    err = (head.mean(head(state)) - target).abs().mean()
    assert err < 1.0, f"two-hot regression not fitting: err={err:.3f}"


def test_policy_and_state_encoder_shapes():
    enc = StateEncoder(vocab_size=240, tokens_per_frame=64, dim=32, out_dim=48)
    tokens = torch.randint(0, 240, (8, 64))
    state = enc(tokens)
    assert state.shape == (8, 48)
    pol = PolicyHead(48, num_actions=8, hidden=32)
    dist = pol(state)
    a = dist.sample()
    assert a.shape == (8,) and a.max() < 8
    assert torch.isfinite(dist.entropy()).all()
