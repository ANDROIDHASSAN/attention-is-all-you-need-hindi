"""Checks for the things that, if wrong, still train but silently give a worse model.

    python check.py
"""

import math

import torch
import torch.nn.functional as F

from bleu import corpus_bleu, tokenize
from bpe import BPE
from model import Config, Transformer, sinusoids
from train import noam
from translate import beam_search, greedy

torch.manual_seed(0)
PAD, BOS, EOS, V = 0, 1, 2, 50


def tiny(**kw) -> Transformer:
    return Transformer(Config(vocab=V, d_model=32, n_heads=4, d_ff=64, n_layers=2, pad=PAD, **kw)).eval()


def check_sinusoids() -> None:
    pe = sinusoids(100, 16)
    pos, i = 7, 3  # dimension 2i and 2i+1
    angle = pos / 10000 ** (2 * i / 16)
    assert math.isclose(pe[pos, 2 * i].item(), math.sin(angle), abs_tol=1e-6)
    assert math.isclose(pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-6)


def check_decoder_is_causal() -> None:
    """Changing target token j must not change the prediction at any position before j."""
    m = tiny()
    src = torch.randint(3, V, (2, 9))
    tgt = torch.randint(3, V, (2, 8))
    before = m(src, tgt)
    tgt2 = tgt.clone()
    tgt2[:, 5] = (tgt2[:, 5] + 1 - 3) % (V - 3) + 3
    after = m(src, tgt2)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-5), "future target token leaked into the past"
    assert not torch.allclose(before[:, 5:], after[:, 5:]), "the changed token should matter from its own position on"


def check_padding_is_ignored() -> None:
    """Padding the source (as batching does) must not change the translation."""
    m = tiny()
    src = torch.randint(3, V, (1, 6))
    tgt = torch.randint(3, V, (1, 5))
    padded = torch.cat([src, torch.full((1, 4), PAD)], dim=1)
    assert torch.allclose(m(src, tgt), m(padded, tgt), atol=1e-5)


def check_embeddings_are_shared() -> None:
    m = tiny()
    names = [n for n, p in m.named_parameters() if p.shape == (V, 32)]
    assert names == ["embed.weight"], names  # one matrix for source, target and output (section 3.4)


def check_learning_rate() -> None:
    peak = noam(4000, 512, 4000)
    assert math.isclose(peak, 512**-0.5 * 4000**-0.5)
    assert noam(2000, 512, 4000) < peak and noam(8000, 512, 4000) < peak


def check_bleu() -> None:
    refs = ["यह एक बिल्ली है।", "the cat is on the mat ."]
    assert math.isclose(corpus_bleu(refs, refs), 100.0)
    assert corpus_bleu(["क ख ग घ", "a b c d"], refs) == 0.0
    assert tokenize("बिल्ली है।") == ["बिल्ली", "है", "।"]  # danda split off, vowel signs stay on the letter


def check_bpe_round_trip() -> None:
    text = "Hello, दुनिया! मेरा नाम Hassan है। 42 km/h"
    tok = BPE.train(text * 30, 300)
    assert tok.decode(tok.encode(text)) == text


def check_beam_1_is_greedy() -> None:
    m = tiny()
    for _ in range(5):
        src = torch.randint(3, V, (1, 7))
        assert beam_search(m, src, BOS, EOS, beam=1) == greedy(m, src, BOS, EOS)[0]


def check_can_overfit() -> None:
    """The whole pipeline (masks, shift by one, loss) is right if the model can learn to copy-reverse 8 pairs."""
    m = Transformer(Config(vocab=V, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.0, pad=PAD))
    src = torch.randint(3, V, (8, 6))
    tgt = src.flip(1)
    tgt_in = torch.cat([torch.full((8, 1), BOS), tgt], 1)
    tgt_out = torch.cat([tgt, torch.full((8, 1), EOS)], 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(src, tgt_in).flatten(0, 1), tgt_out.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    assert greedy(m, src, BOS, EOS) == tgt.tolist(), f"loss {loss.item():.4f}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            fn()
            print(f"ok  {name[6:]}")
