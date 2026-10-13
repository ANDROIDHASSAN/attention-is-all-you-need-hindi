"""Decoding and evaluation, as in section 6.1 of the paper:
  - average the last 5 checkpoints
  - beam search, beam size 4, length penalty alpha = 0.6 (Wu et al. 2016)
  - maximum output length = input length + 50, stop early when the beam is done

    python translate.py runs/base                       # test BLEU with beam 4
    python translate.py runs/base --beam 1 --avg 1      # greedy, last checkpoint only
    python translate.py runs/base --text "How are you?"
"""

import argparse
import time
from pathlib import Path

import torch

from bleu import corpus_bleu
from bpe import BPE
from model import Config, Transformer

DATA = Path(__file__).parent / "data"


def load(run: Path, avg: int, device: str) -> Transformer:
    """Average the parameters of the last `avg` checkpoints of a run (section 6.1)."""
    paths = sorted(run.glob("step*.pt"), key=lambda p: int(p.stem[4:]))[-avg:]
    states = [torch.load(p, map_location="cpu") for p in paths]
    avg_state = {k: sum(s["model"][k].float() for s in states) / len(states) for k in states[0]["model"]}
    model = Transformer(Config(**states[0]["cfg"]))
    model.load_state_dict(avg_state)
    print(f"averaged {len(paths)} checkpoints: {', '.join(p.stem for p in paths)}")
    return model.to(device).eval()


def length_penalty(n: int, alpha: float) -> float:
    return ((5 + n) / 6) ** alpha  # GNMT: longer outputs are not punished for having more log-prob terms


@torch.no_grad()
def greedy(model: Transformer, src: torch.Tensor, bos: int, eos: int, max_extra: int = 50) -> list[list[int]]:
    memory, pad = model.encode(src), model.cfg.pad
    ys = torch.full((src.size(0), 1), bos, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(src.size(1) + max_extra):
        nxt = model.decode(ys, memory, src)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done |= nxt == eos
        if done.all():
            break
    return [cut(row.tolist()[1:], eos) for row in ys]


@torch.no_grad()
def beam_search(model: Transformer, src: torch.Tensor, bos: int, eos: int, beam: int = 4, alpha: float = 0.6,
                max_extra: int = 50) -> list[int]:
    """One sentence (src: (1, S)). Keep the `beam` best partial translations by total log-probability;
    a translation that emits <eos> is finished and scored with the length penalty."""
    memory = model.encode(src)
    alive: list[tuple[list[int], float]] = [([bos], 0.0)]
    finished: list[tuple[list[int], float]] = []
    for _ in range(src.size(1) + max_extra):
        ys = torch.tensor([seq for seq, _ in alive], device=src.device)
        n = len(alive)
        logp = model.decode(ys, memory.expand(n, -1, -1), src.expand(n, -1))[:, -1].log_softmax(-1)
        total = logp + torch.tensor([s for _, s in alive], device=src.device)[:, None]
        scores, idx = total.view(-1).topk(2 * beam)  # 2x so that `beam` candidates survive even if some end here
        alive = []
        for score, i in zip(scores.tolist(), idx.tolist()):
            prev, tok = divmod(i, logp.size(1))
            seq = [*ys[prev].tolist(), tok]
            if tok == eos:
                finished.append((seq, score / length_penalty(len(seq) - 1, alpha)))
            else:
                alive.append((seq, score))
            if len(alive) == beam:
                break
        if len(finished) >= beam:
            break
    if not finished:
        finished = [(seq, s / length_penalty(len(seq) - 1, alpha)) for seq, s in alive]
    return cut(max(finished, key=lambda f: f[1])[0][1:], eos)


def cut(ids: list[int], eos: int) -> list[int]:
    return ids[: ids.index(eos)] if eos in ids else ids


def translate(model: Transformer, tok: BPE, sentences: list[str], beam: int, alpha: float, device: str,
              batch: int = 64) -> list[str]:
    out = []
    for i in range(0, len(sentences), batch if beam == 1 else 1):
        chunk = sentences[i : i + (batch if beam == 1 else 1)]
        ids = [tok.encode(s)[: model.cfg.max_len - 51] + [tok.eos] for s in chunk]
        src = torch.full((len(ids), max(map(len, ids))), tok.pad, dtype=torch.long)
        for j, s in enumerate(ids):
            src[j, : len(s)] = torch.tensor(s)
        src = src.to(device)
        if beam == 1:
            out += [tok.decode(h) for h in greedy(model, src, tok.bos, tok.eos)]
        else:
            out.append(tok.decode(beam_search(model, src, tok.bos, tok.eos, beam, alpha)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--avg", type=int, default=5)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--text")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    tok = BPE.load(DATA / "bpe.json")
    model = load(args.run, args.avg, args.device)
    if args.text:
        print(translate(model, tok, [args.text], args.beam, args.alpha, args.device)[0])
        return
    src = (DATA / f"{args.split}.en").read_text().splitlines()
    ref = (DATA / f"{args.split}.hi").read_text().splitlines()
    t0 = time.time()
    hyp = translate(model, tok, src, args.beam, args.alpha, args.device)
    name = f"{args.split}.beam{args.beam}.avg{args.avg}.hi"
    (args.run / name).write_text("\n".join(hyp) + "\n")
    print(f"{args.split}: {len(hyp)} sentences in {time.time() - t0:.0f}s, BLEU {corpus_bleu(hyp, ref):.2f}  -> {args.run / name}")
    print(f"sacreBLEU: uvx sacrebleu {DATA / (args.split + '.hi')} -i {args.run / name} -m bleu chrf")


if __name__ == "__main__":
    main()
