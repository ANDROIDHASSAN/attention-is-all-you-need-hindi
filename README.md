# Attention Is All You Need — English → Hindi, from scratch

A from-scratch PyTorch reimplementation of the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
with byte-pair encoding ([Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)), trained for English → Hindi
translation on the [IIT Bombay English-Hindi corpus](https://arxiv.org/abs/1710.02855) on one laptop GPU
(Apple M1 Pro, 16 GB).

Everything the paper describes is written by hand: multi-head attention, sinusoidal positions, post-norm
encoder-decoder, shared embeddings, the warmup learning-rate schedule, label smoothing, length-bucketed batching,
checkpoint averaging, beam search with length penalty, BPE and BLEU. PyTorch is used for tensors, autograd and Adam.

> **Status:** the main run is trained and scored (below). The Table 3 ablations are queued.

## Paper → code

| Paper | What | Code |
|---|---|---|
| §3.1 | Encoder-decoder stack, `LayerNorm(x + Sublayer(x))` (post-norm) | `EncoderLayer`, `DecoderLayer` in [model.py](model.py) |
| §3.2.1, eq. 1 | Scaled dot-product attention, `softmax(QKᵀ/√d_k)V` | `MultiHeadAttention.forward` |
| §3.2.2 | Multi-head attention, h heads of size d_k = d_model / h | `MultiHeadAttention` |
| §3.2.3 | Decoder self-attention masked to earlier positions; encoder-decoder attention | `Transformer.tgt_mask`, `DecoderLayer` |
| §3.3 | Position-wise feed-forward, ReLU, d_ff = 4 · d_model | `FeedForward` |
| §3.4 | One weight matrix for source embedding, target embedding and pre-softmax linear; embeddings × √d_model | `Transformer.embed`, `_embed`, `decode` |
| §3.5 | Sinusoidal positional encoding | `sinusoids` |
| §5.1 | Shared source-target BPE vocabulary; batches by approximate length | [bpe.py](bpe.py), `make_batches` in [train.py](train.py) |
| §5.3, eq. 3 | Adam (β₁ 0.9, β₂ 0.98, ε 1e-9), `lr = d^-0.5 · min(step^-0.5, step · warmup^-1.5)`, warmup 4000 | `noam`, `main` in [train.py](train.py) |
| §5.4 | Residual and embedding dropout 0.1, label smoothing ε = 0.1 | `Config.dropout`, `cross_entropy(..., label_smoothing)` |
| §6.1 | Average the last 5 checkpoints; beam 4, length penalty α = 0.6; max length = input + 50 | `load`, `beam_search` in [translate.py](translate.py) |
| Table 3 | Ablations: number of heads, label smoothing, dropout, learned positions | [ablations.sh](ablations.sh) |
| — | Corpus BLEU (Papineni et al., 2002) | [bleu.py](bleu.py) |

## Where this differs from the paper, and why

| | Paper (base) | Here | Why |
|---|---|---|---|
| Size | d_model 512, 6 + 6 layers, 65M params | d_model 256, 4 + 4 layers, 9.4M params | One laptop GPU |
| Vocabulary | ~37k BPE, character-level | 8k BPE, byte-level (as GPT-2) | Smaller model; bytes mean Devanagari never hits an unknown token |
| Batch | ~25k source + 25k target tokens | ~12k padded tokens per side (3k × 4 accumulated) | Memory |
| Training | 100k steps, 8 × P100, 12 h | 20k steps, M1 Pro (~28k tokens/s) | Compute |
| Language pair | En → De, En → Fr (WMT 2014) | En → Hi (IITB, WAT test set) | Hindi is the point |

Label smoothing uses PyTorch's built-in, which spreads ε over all classes including padding; the paper does not
say how its implementation treats padding.

## Data

`data.py` downloads the corpus from Hugging Face (`cfilt/iitb-english-hindi`) and prepares it:

| Step | Pairs |
|---|---|
| IITB training set | 1,659,083 |
| After cleaning (drop duplicates, untranslated lines, pairs without Latin on the English side or Devanagari on the Hindi side) | 1,394,805 |
| After length filter (1–100 tokens per side, length ratio ≤ 3 : 1 + 3) | 1,359,431 (28.8M English + 29.9M Hindi tokens) |
| Dev / test (untouched) | 520 / 2,507 |

The shared 8,000-token BPE is learned on 150k random training pairs.

## Results

English → Hindi, IITB / WAT test set (2,507 sentences), last 5 checkpoints (steps 16k–20k) averaged, beam 4,
length penalty 0.6:

| Metric | Score |
|---|---|
| sacreBLEU (13a tokenizer) | **12.7** |
| chrF2 | **39.3** |
| BLEU from [bleu.py](bleu.py) (Devanagari-aware tokenization) | 13.72 |

sacreBLEU detail: n-gram precisions 47.6 / 20.7 / 9.8 / 4.8, brevity penalty 0.864 (the output is 13% shorter
than the reference). For scale, the IITB corpus paper reports 12.23 BLEU for its En → Hi NMT system; its
tokenization is not identical, so treat that as a reference point, not a like-for-like comparison.

Training curve (dev set, 520 sentences, greedy decoding of the current weights):

| Step | 2.5k | 5k | 7.5k | 10k | 12.5k | 15k | 17.5k | 20k |
|---|---|---|---|---|---|---|---|---|
| Dev perplexity | 122.4 | 54.8 | 28.7 | 18.9 | 15.4 | 13.8 | 12.8 | 12.1 |
| Dev BLEU | 0.65 | 2.77 | 5.47 | 8.88 | 10.41 | 12.00 | 12.40 | 12.88 |

Dev BLEU was still rising at 20k steps; a longer run would likely score higher.

## Correctness checks

`python check.py` tests the things that, when wrong, still train but quietly give a worse model:

- sinusoidal encoding matches the formula in §3.5
- the decoder is causal: changing target token *j* changes no prediction before *j*
- source padding changes nothing
- one embedding matrix is shared by encoder, decoder and output
- the learning rate peaks at the warmup step
- BLEU is 100 on identical text and 0 with no overlap; the Devanagari danda is split off, vowel signs are not
- BPE encodes and decodes mixed Hindi-English text losslessly
- beam search with beam 1 equals greedy decoding
- the full pipeline can overfit 8 sequence-reversal pairs

Removing the causal mask or the padding mask makes the matching check fail.

## Run it

```bash
pip install -r requirements.txt
python data.py                     # ~8 min: download, clean, learn BPE, encode
python check.py
python train.py --name base --steps 20000 --max-tokens 3000 --accum 4
python translate.py runs/base      # test set: average last 5 checkpoints, beam 4, BLEU
python translate.py runs/base --text "Where is the railway station?"
./ablations.sh                     # Table 3, on a 300k-pair subset
```

On Apple silicon the code uses the `mps` device; elsewhere pass `--device cuda` or `--device cpu`.

## References

- Vaswani et al., 2017. *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Sennrich, Haddow & Birch, 2016. *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Kunchukuttan, Mehta & Bhattacharyya, 2018. *The IIT Bombay English-Hindi Parallel Corpus.* [arXiv:1710.02855](https://arxiv.org/abs/1710.02855)
- Papineni et al., 2002. *BLEU: a Method for Automatic Evaluation of Machine Translation.* [ACL](https://aclanthology.org/P02-1040/)
- Wu et al., 2016. *Google's Neural Machine Translation System* (length penalty). [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
