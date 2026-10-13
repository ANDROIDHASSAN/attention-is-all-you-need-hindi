"""The Transformer of "Attention Is All You Need" (Vaswani et al. 2017, arXiv 1706.03762), as written in the paper.

Section numbers below refer to the paper. Two choices that later models changed are kept as the paper has them:
  - POST-norm: LayerNorm(x + Sublayer(x)) (section 3.1). GPT-2 and later moved the norm in front of the
    sublayer; post-norm is why the paper needs a learning-rate warmup.
  - SINUSOIDAL positions added to the embeddings (section 3.5), not learned and not rotary.

    source ids -> embed * sqrt(d) + PE -> N x [self-attention, FFN] ----------------------- memory
    target ids -> embed * sqrt(d) + PE -> N x [masked self-attention, attention over memory, FFN] -> linear
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 8000
    d_model: int = 256  # paper base: 512
    n_heads: int = 8  # paper base: 8, so d_k = d_v = 32 here (64 in the paper)
    d_ff: int = 1024  # paper base: 2048 (always 4 x d_model)
    n_layers: int = 4  # paper base: 6 encoder + 6 decoder
    dropout: float = 0.1  # P_drop, section 5.4
    max_len: int = 512
    pad: int = 0  # set from the tokenizer
    learned_pos: bool = False  # Table 3 row (E): learned positional embeddings instead of sinusoids


def sinusoids(max_len: int, d_model: int) -> torch.Tensor:
    """Section 3.5: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
    Each pair of dimensions is a clock at its own frequency; PE(pos + k) is a rotation of PE(pos), so a
    relative offset is a linear function the attention layers can learn."""
    pos = torch.arange(max_len).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class MultiHeadAttention(nn.Module):
    """Section 3.2.2. Each head: softmax(Q K^T / sqrt(d_k)) V on its own d_k-sized projection; heads are
    concatenated and projected back with W^O."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.h, self.d_k = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(cfg.d_model, cfg.d_model) for _ in range(4))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) asks the questions; memory: (B, S, d) is attended to (x itself for self-attention).
        # mask: (B or 1, 1, T or 1, S), True where attention is allowed.
        B, T, _ = x.shape
        split = lambda t: t.view(B, -1, self.h, self.d_k).transpose(1, 2)  # (B, h, len, d_k)
        q, k, v = split(self.q(x)), split(self.k(memory)), split(self.v(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, T, S), equation (1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.drop(scores.softmax(dim=-1))  # attention dropout, as in the paper's tensor2tensor code
        return self.o((weights @ v).transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    """Section 3.3: FFN(x) = max(0, x W1 + b1) W2 + b2, applied to every position separately."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.w1, self.w2 = nn.Linear(cfg.d_model, cfg.d_ff), nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn, self.ff = MultiHeadAttention(cfg), FeedForward(cfg)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, src_mask)))  # LayerNorm(x + Sublayer(x)), section 3.1
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.self_attn, self.cross_attn, self.ff = MultiHeadAttention(cfg), MultiHeadAttention(cfg), FeedForward(cfg)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(cfg.d_model) for _ in range(3))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, y: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(y + self.drop(self.self_attn(y, y, tgt_mask)))  # masked: position i sees only <= i
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, src_mask)))  # queries from decoder, K/V from encoder
        return self.norm3(y + self.drop(self.ff(y)))


class Transformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Section 3.4: one matrix for the source embedding, the target embedding and the pre-softmax linear.
        # Works because the BPE vocabulary is shared between English and Hindi.
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=cfg.pad)
        nn.init.normal_(self.embed.weight, std=cfg.d_model**-0.5)  # so embed * sqrt(d) starts at unit scale
        if cfg.learned_pos:
            self.pos = nn.Parameter(torch.randn(cfg.max_len, cfg.d_model) * 0.02)
        else:
            self.register_buffer("pos", sinusoids(cfg.max_len, cfg.d_model), persistent=False)
        self.encoder = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.decoder = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.drop = nn.Dropout(cfg.dropout)
        for m in self.modules():  # Glorot uniform, the tensor2tensor default the paper's models were trained with
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        # Section 3.4: "we multiply those weights by sqrt(d_model)"; section 5.4: dropout on embeddings + PE.
        return self.drop(self.embed(ids) * math.sqrt(self.cfg.d_model) + self.pos[: ids.size(1)])

    def src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.cfg.pad)[:, None, None, :]  # (B, 1, 1, S): never attend to padding

    def tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        T = tgt.size(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=tgt.device).tril()  # (T, T): i sees j <= i
        return causal[None, None] & (tgt != self.cfg.pad)[:, None, None, :]

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x, mask = self._embed(src), self.src_mask(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Logits for the next token at every target position."""
        y, src_mask, tgt_mask = self._embed(tgt), self.src_mask(src), self.tgt_mask(tgt)
        for layer in self.decoder:
            y = layer(y, memory, src_mask, tgt_mask)
        return y @ self.embed.weight.T  # tied output projection

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(tgt_in, self.encode(src), src)
