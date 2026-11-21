#!/bin/bash
# Cartoon Set stack: 64px autoencoder, 32 latent, 46M teacher, then the six DMD runs.
# GPUs 1 and 2 only. GPU 0 is left alone because another user has a job there.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONUNBUFFERED=1
mkdir -p runs

echo "vae $(date -Is)"
CUDA_VISIBLE_DEVICES=1 "$PY" -u train_vae.py \
  --data data/processed_cartoon --out runs/vae_cartoon \
  --res 64 --latent-res 32 --epochs 12 --batch 128 \
  --log-every 50 --sample-every 200 --save-every 400 \
  > runs/vae_cartoon.log 2>&1
echo "vae done $(date -Is)"

echo "teacher $(date -Is)"
CUDA_VISIBLE_DEVICES=1 "$PY" -u train_latent.py \
  --data runs/vae_cartoon --vae runs/vae_cartoon/last.pt \
  --out runs/teacher_cartoon --ch 128 --mults 1,2,2,2 --attn-res 16,8 \
  --batch 128 --iters 15000 --sample-every 1000 --save-every 3000 \
  > runs/teacher_cartoon.log 2>&1
echo "teacher done $(date -Is)"

echo "pairs $(date -Is)"
CUDA_VISIBLE_DEVICES=1 "$PY" -u make_pairs.py \
  --teacher runs/teacher_cartoon/last.pt \
  --data runs/vae_cartoon \
  --vae runs/vae_cartoon/last.pt \
  --n 10000 --batch 128 --steps 20 --cfg 2 --seed 123 \
  --out runs/pairs_cartoon.pt \
  > runs/pairs_cartoon.log 2>&1
echo "pairs done $(date -Is)"

common=(
  --teacher runs/teacher_cartoon/last.pt
  --pairs runs/pairs_cartoon.pt
  --data runs/vae_cartoon
  --vae runs/vae_cartoon/last.pt
  --iters 20000 --batch 32 --reg-batch 32
  --sample-every 2000 --save-every 4000
)

launch() {
  local gpu="$1" out="$2"
  shift 2
  mkdir -p "$out"
  echo "start $out gpu $gpu $(date -Is)" >&2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_dmd.py "${common[@]}" --out "$out" "$@" \
    > "$out/train.log" 2>&1 &
  echo $!
}

p1=$(launch 1 runs/dmd_cartoon --cfg 2 --lambda-reg 0.25)
p2=$(launch 2 runs/dmd_cartoon_noreg --cfg 2 --lambda-reg 0)
wait "$p1"
wait "$p2"
echo "wave1 done $(date -Is)"

p1=$(launch 1 runs/dmd_cartoon_regonly --cfg 2 --lambda-reg 0.25 --no-dm)
p2=$(launch 2 runs/dmd_cartoon_cfg1 --cfg 1 --lambda-reg 0.25)
wait "$p1"
wait "$p2"
echo "wave2 done $(date -Is)"

p1=$(launch 1 runs/dmd_cartoon_cfg4 --cfg 4 --lambda-reg 0.25)
p2=$(launch 2 runs/dmd_cartoon_dmd2 --cfg 2 --lambda-reg 0.25 --fake-steps 4)
wait "$p1"
wait "$p2"
echo "cartoon queue done $(date -Is)"
