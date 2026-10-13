"""IIT Bombay English-Hindi corpus -> cleaned pairs -> shared BPE -> token arrays.

Corpus: Kunchukuttan, Mehta & Bhattacharyya 2018 (arXiv 1710.02855), from Hugging Face `cfilt/iitb-english-hindi`.
1,659,083 training pairs, 520 dev, 2,507 test. The test set is the standard English-Hindi benchmark (WAT).

Run once (pyarrow is only needed here, to read the parquet files):
    uv run --python <venv python> --with pyarrow data.py
Writes data/bpe.json, data/{train,dev,test}.npz and the raw dev/test text used as BLEU references.
"""

import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bpe import BPE

DATA = Path(__file__).parent / "data"
VOCAB = 8000  # shared English + Hindi; the paper used ~37k for a 65M model on 4.5M pairs
BPE_SAMPLE = 150_000  # pairs used to learn the merges (pure-Python BPE; the full corpus would take hours)
MAX_LEN = 100  # tokens per side, training pairs only

LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def read_split(name: str) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "cfilt/iitb-english-hindi", f"data/{name}-00000-of-00001.parquet", repo_type="dataset", local_dir=DATA / "raw"
    )
    rows = pq.read_table(path).column("translation").to_pylist()
    return [(" ".join(r["en"].split()), " ".join(r["hi"].split())) for r in rows]  # collapse runs of whitespace


def clean(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Training pairs only. The IITB training set mixes parallel text with software UI strings and
    untranslated lines; keep pairs that look like English on one side and Hindi on the other."""
    seen, out = set(), []
    for en, hi in pairs:
        if en == hi or not LATIN.search(en) or not DEVANAGARI.search(hi) or (en, hi) in seen:
            continue
        seen.add((en, hi))
        out.append((en, hi))
    return out


_tok: BPE


def _init(merges: list) -> None:
    global _tok
    _tok = BPE([tuple(m) for m in merges])


def _encode(lines: list[str]) -> list[list[int]]:
    return [_tok.encode(s) for s in lines]


def encode_all(tok: BPE, lines: list[str], workers: int = 8) -> list[list[int]]:
    parts = [lines[i : i + 20_000] for i in range(0, len(lines), 20_000)]
    with Pool(workers, initializer=_init, initargs=(tok.merges,)) as pool:
        return [ids for part in pool.map(_encode, parts) for ids in part]


def pack(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Many variable-length lists as one flat int16 array plus offsets: ~10x less memory than Python lists."""
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in seqs])
    return np.fromiter((t for s in seqs for t in s), dtype=np.int16, count=offsets[-1]), offsets


def main() -> None:
    t0 = time.time()
    splits = {"train": clean(read_split("train")), "dev": read_split("validation"), "test": read_split("test")}
    print(f"train {len(splits['train']):,} pairs after cleaning, dev {len(splits['dev'])}, test {len(splits['test'])}")

    for name in ("dev", "test"):  # untouched references for BLEU
        for i, lang in enumerate(("en", "hi")):
            (DATA / f"{name}.{lang}").write_text("\n".join(p[i] for p in splits[name]) + "\n")

    sample = random.Random(0).sample(splits["train"], BPE_SAMPLE)
    tok = BPE.train("\n".join(s for pair in sample for s in pair), VOCAB)
    tok.save(DATA / "bpe.json")
    print(f"BPE: {tok.vocab_size} tokens learned in {time.time() - t0:.0f}s")

    for name, pairs in splits.items():
        src = encode_all(tok, [p[0] for p in pairs])
        tgt = encode_all(tok, [p[1] for p in pairs])
        if name == "train":
            keep = [
                i for i, (s, t) in enumerate(zip(src, tgt))
                if 0 < len(s) <= MAX_LEN and 0 < len(t) <= MAX_LEN and max(len(s), len(t)) <= 3 * min(len(s), len(t)) + 3
            ]
            src, tgt = [src[i] for i in keep], [tgt[i] for i in keep]
        (src_ids, src_off), (tgt_ids, tgt_off) = pack(src), pack(tgt)
        np.savez(DATA / f"{name}.npz", src=src_ids, src_off=src_off, tgt=tgt_ids, tgt_off=tgt_off)
        n_tok = len(src_ids) + len(tgt_ids)
        print(f"{name}: {len(src):,} pairs, {len(src_ids):,} en + {len(tgt_ids):,} hi tokens ({n_tok / len(src) / 2:.1f}/sentence)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
