#!/bin/sh
# Table 3 of the paper, scaled down: 300k training pairs, 8000 steps, one change at a time from the reference.
# Waits for the main run to finish so the two never share the GPU.
#   PY=../../own-voice-models/.venv/bin/python ./ablations.sh > runs/ablations.log 2>&1
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
while pgrep -f "train.py --name base" > /dev/null; do sleep 60; done

common="--train-pairs 300000 --steps 8000 --max-tokens 3000 --accum 2 --eval-every 4000 --save-every 4000"
$PY -u train.py --name abl-ref $common
$PY -u train.py --name abl-heads1 --heads 1 $common          # row (A): one head
$PY -u train.py --name abl-heads32 --heads 32 $common        # row (A): d_k = 8
$PY -u train.py --name abl-no-smoothing --smoothing 0 $common  # row (D): epsilon_ls = 0
$PY -u train.py --name abl-no-dropout --dropout 0 $common    # row (D): P_drop = 0
$PY -u train.py --name abl-learned-pos --learned-pos $common # row (E): learned positions
