"""Corpus BLEU (Papineni et al. 2002), the metric the Transformer paper reports.

BLEU = brevity_penalty * exp(mean over n=1..4 of log p_n)
  p_n = clipped n-gram matches / n-grams in the hypotheses, summed over the WHOLE corpus (not averaged per sentence)
  brevity_penalty = exp(1 - ref_len / hyp_len) if the hypotheses are shorter than the references, else 1

The number depends on tokenization, so `tokenize` splits punctuation off words. Unicode categories decide
what punctuation is, so the Devanagari danda (।) is split off but vowel signs (ा ि ी, category M) stay on
their letter. The headline numbers in the README come from sacreBLEU; check.py shows this code agrees with it.
"""

import math
import unicodedata
from collections import Counter


def tokenize(text: str) -> list[str]:
    out = []
    for ch in text:
        out.append(f" {ch} " if unicodedata.category(ch)[0] in "PS" else ch)
    return "".join(out).split()


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps: list[str], refs: list[str], tok=tokenize) -> float:
    matches, totals = [0] * 4, [0] * 4
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        h, r = tok(hyp), tok(ref)
        hyp_len, ref_len = hyp_len + len(h), ref_len + len(r)
        for n in range(1, 5):
            hn, rn = ngrams(h, n), ngrams(r, n)
            matches[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())  # clipped: a word counts at most as often as in the ref
            totals[n - 1] += max(len(h) - n + 1, 0)
    if min(matches) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(matches, totals)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(log_p)
