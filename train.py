"""Training, section 5 of the paper, scaled to one laptop GPU (Apple M1 Pro, MPS).

    paper (base)                                   here
    batches of ~25k source + ~25k target tokens    --max-tokens per batch (padded, per side)
    Adam, beta1 0.9, beta2 0.98, eps 1e-9          same
    lr = d^-0.5 * min(step^-0.5, step * warmup^-1.5), warmup 4000   same formula
    label smoothing 0.1                            same
    100k steps, 12 hours on 8 P100 GPUs            --steps, on one laptop

    python train.py --name base --steps 60000
    python train.py --name heads1 --heads 1 --train-pairs 300000 --steps 15000    # an ablation (Table 3)

A run writes runs/<name>/train.log lines, step<N>.pt checkpoints (the last 5 are kept for averaging) and
last.pt with the optimizer, so a killed run continues where it stopped when started again.
"""

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bleu import corpus_bleu
from bpe import BPE
from model import Config, Transformer
from translate import translate

ROOT = Path(__file__).parent
DATA = ROOT / "data"


class Pairs:
    """One split's token arrays. Pair i is sliced when asked for: building 1.4M numpy views up front cost
    about half a gigabyte."""

    def __init__(self, name: str, limit: int | None = None) -> None:
        z = np.load(DATA / f"{name}.npz")
        self.src, self.so, self.tgt, self.to = z["src"], z["src_off"], z["tgt"], z["tgt_off"]
        self.idx = np.arange(len(self.so) - 1)
        if limit:
            self.idx = np.sort(np.random.default_rng(0).choice(self.idx, limit, replace=False))
        self.src_len = np.diff(self.so)[self.idx]
        self.tgt_len = np.diff(self.to)[self.idx]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        j = self.idx[i]
        return self.src[self.so[j] : self.so[j + 1]], self.tgt[self.to[j] : self.to[j + 1]]


def make_batches(pairs: Pairs, max_tokens: int) -> list[list[int]]:
    """Section 5.1: "sentence pairs were batched together by approximate sequence length". Sorting by length
    keeps padding low; a batch grows until its padded size (rows x longest side + 1) reaches max_tokens."""
    order = np.lexsort((pairs.tgt_len, pairs.src_len)).tolist()
    longest_side = (np.maximum(pairs.src_len, pairs.tgt_len) + 1).tolist()
    batches, cur, longest = [], [], 0
    for i in order:
        n = longest_side[i]
        if cur and (len(cur) + 1) * max(longest, n) > max_tokens:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    return batches + [cur] if cur else batches


def collate(pairs: Pairs, idx: list[int], tok: BPE, device: str) -> tuple[torch.Tensor, ...]:
    """src = sentence + <eos>; decoder input = <bos> + target; decoder output = target + <eos> (teacher forcing)."""
    src = [list(pairs[i][0]) + [tok.eos] for i in idx]
    tgt = [list(pairs[i][1]) for i in idx]
    # Round the padded length up to a multiple of 8: far fewer distinct tensor shapes, and on MPS every new
    # shape keeps its own cached buffers (without this the run grew to 5 GB of wired memory and swapped).
    width = lambda rows: -(-max(map(len, rows)) // 8) * 8
    pad = lambda rows: torch.tensor([r + [tok.pad] * (width(rows) - len(r)) for r in rows])
    return (
        pad(src).to(device),
        pad([[tok.bos] + t for t in tgt]).to(device),
        pad([t + [tok.eos] for t in tgt]).to(device),
    )


def noam(step: int, d_model: int, warmup: int) -> float:
    """Equation (3): rise linearly for `warmup` steps, then decay as 1 / sqrt(step)."""
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


@torch.no_grad()
def dev_loss(model: Transformer, pairs: Pairs, tok: BPE, device: str) -> float:
    """Plain negative log-likelihood per target token (no smoothing), so runs with different smoothing compare."""
    model.eval()
    total = n = 0
    for idx in make_batches(pairs, 4000):
        src, tgt_in, tgt_out = collate(pairs, idx, tok, device)
        logits = model(src, tgt_in)
        total += F.cross_entropy(logits.flatten(0, 1), tgt_out.flatten(), ignore_index=tok.pad, reduction="sum").item()
        n += (tgt_out != tok.pad).sum().item()
    model.train()
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--learned-pos", action="store_true")
    ap.add_argument("--train-pairs", type=int, help="random subset, for ablations")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    tok = BPE.load(DATA / "bpe.json")
    cfg = Config(vocab=tok.vocab_size, d_model=args.d_model, n_heads=args.heads, d_ff=4 * args.d_model,
                 n_layers=args.layers, dropout=args.dropout, pad=tok.pad, learned_pos=args.learned_pos)
    model = Transformer(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam(s + 1, cfg.d_model, args.warmup))

    step = 0
    if (run / "last.pt").exists():
        state = torch.load(run / "last.pt", map_location=args.device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        print(f"resumed at step {step}")

    train = Pairs("train", args.train_pairs)
    dev = Pairs("dev")
    dev_src = (DATA / "dev.en").read_text().splitlines()
    dev_ref = (DATA / "dev.hi").read_text().splitlines()
    batches = make_batches(train, args.max_tokens)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args}\n{n_params / 1e6:.1f}M parameters, {len(train):,} pairs, {len(batches):,} batches per epoch", flush=True)

    rng = random.Random(step)
    queue: list[list[int]] = []
    t0, tokens, loss_sum, loss_n = time.time(), 0, 0.0, 0
    model.train()
    while step < args.steps:
        for _ in range(args.accum):
            if not queue:
                queue = batches[:]
                rng.shuffle(queue)
            src, tgt_in, tgt_out = collate(train, queue.pop(), tok, args.device)
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt_out.flatten(), ignore_index=tok.pad,
                                   label_smoothing=args.smoothing)  # section 5.4, epsilon_ls = 0.1
            (loss / args.accum).backward()
            n_tgt = (tgt_out != tok.pad).sum().item()
            tokens += n_tgt + (src != tok.pad).sum().item()
            loss_sum, loss_n = loss_sum + loss.item() * n_tgt, loss_n + n_tgt
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 100 == 0:
            dt = time.time() - t0
            mem = ""
            if args.device == "mps":
                mem = f"  {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
                torch.mps.empty_cache()
            print(f"step {step:6d}  loss {loss_sum / loss_n:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{tokens / dt:,.0f} tok/s  {dt / 60:.1f} min{mem}", flush=True)
            loss_sum = loss_n = 0
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": step}, run / f"step{step}.pt")
            for old in sorted(run.glob("step*.pt"), key=lambda p: int(p.stem[4:]))[:-5]:
                old.unlink()  # keep the last 5 for checkpoint averaging
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, run / "last.pt")
        if step % args.eval_every == 0 or step == args.steps:
            nll = dev_loss(model, dev, tok, args.device)
            model.eval()
            hyp = translate(model, tok, dev_src, beam=1, alpha=0.6, device=args.device)
            model.train()
            print(f"eval {step:6d}  dev nll {nll:.3f}  ppl {np.exp(nll):.1f}  dev BLEU (greedy) "
                  f"{corpus_bleu(hyp, dev_ref):.2f}  | {hyp[1][:80]}", flush=True)


if __name__ == "__main__":
    main()
