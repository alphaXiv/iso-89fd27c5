#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --disable-pip-version-check -r requirements-repro.txt
rm -rf outputs
python prepare_base.py

seed="$(python -c 'import json; print(json.load(open("experiment_config.json"))["seed"])')"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  train_specialist.py --task sst2 --output outputs/specialist_sst2 \
  --seed "$((seed + 101))" &
pid_sst=$!
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  train_specialist.py --task qnli --output outputs/specialist_qnli \
  --seed "$((seed + 202))" &
pid_qnli=$!
wait "$pid_sst"
wait "$pid_qnli"

python reproduce.py
