import torch

from models.tokenizer import FSQ, Tokenizer, float_to_img, img_to_float


def test_fsq_index_roundtrip():
    fsq = FSQ((8, 6, 5))
    assert fsq.codebook_size == 240
    z = torch.randn(1000, 3) * 4
    zq = fsq(z)
    idx = fsq.codes_to_indices(zq)
    assert idx.min() >= 0 and idx.max() < 240
    assert torch.allclose(fsq.indices_to_codes(idx), zq, atol=1e-5)


def test_img_conversion_roundtrip(frame_batch):
    x = img_to_float(frame_batch)
    assert x.shape == (8, 3, 64, 64) and x.min() >= -1 and x.max() <= 1
    assert torch.equal(float_to_img(x), frame_batch)


def test_tokenizer_shapes(frame_batch):
    tok = Tokenizer(base_ch=16)
    x = img_to_float(frame_batch)
    assert tok(x).shape == x.shape
    idx = tok.encode_indices(x)
    assert idx.shape == (8, 8, 8) and idx.max() < tok.vocab_size
    assert tok.decode_indices(idx).shape == x.shape


def test_tokenizer_overfits_one_batch(frame_batch):
    torch.manual_seed(0)
    tok = Tokenizer(base_ch=16)
    x = img_to_float(frame_batch)
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-4)
    losses = []
    for _ in range(60):
        loss = torch.nn.functional.mse_loss(tok(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"no convergence: {losses[0]:.4f} -> {losses[-1]:.4f}"
