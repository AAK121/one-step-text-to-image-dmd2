#!/bin/bash
# 128px cartoons in parallel with the 64px DMD jobs already on GPUs 1 and 2.
# The autoencoder, teacher, and pairs use GPU 0. Distillation starts on GPUs 1 and 2
# once those 64px jobs exit, so the two resolutions do not share a card.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONUNBUFFERED=1
mkdir -p runs

echo "vae128 $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_vae.py \
  --data data/processed_cartoon128 --out runs/vae_cartoon128 \
  --res 128 --latent-res 64 --epochs 12 --batch 64 \
  --log-every 50 --sample-every 200 --save-every 400 \
  > runs/vae_cartoon128.log 2>&1
echo "vae128 done $(date -Is)"

echo "teacher128 $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_latent.py \
  --data runs/vae_cartoon128 --vae runs/vae_cartoon128/last.pt \
  --out runs/teacher_cartoon128 --ch 128 --mults 1,2,2,2 --attn-res 32,16 \
  --batch 32 --iters 15000 --sample-every 1000 --save-every 3000 \
  > runs/teacher_cartoon128.log 2>&1
echo "teacher128 done $(date -Is)"

echo "pairs128 $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u make_pairs.py \
  --teacher runs/teacher_cartoon128/last.pt \
  --data runs/vae_cartoon128 \
  --vae runs/vae_cartoon128/last.pt \
  --n 10000 --batch 32 --steps 20 --cfg 2 --seed 123 \
  --out runs/pairs_cartoon128.pt \
  > runs/pairs_cartoon128.log 2>&1
echo "pairs128 done $(date -Is)"

# 64px DMD currently holding the empty cards: full run on GPU 1, no-regression on GPU 2.
while kill -0 1895275 2>/dev/null || kill -0 1895279 2>/dev/null; do
  sleep 30
done
echo "64px wave released the gpus $(date -Is)"

common=(
  --teacher runs/teacher_cartoon128/last.pt
  --pairs runs/pairs_cartoon128.pt
  --data runs/vae_cartoon128
  --vae runs/vae_cartoon128/last.pt
  --iters 20000 --batch 16 --reg-batch 16
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

p1=$(launch 1 runs/dmd_cartoon128 --cfg 2 --lambda-reg 0.25)
p2=$(launch 2 runs/dmd_cartoon128_noreg --cfg 2 --lambda-reg 0)
wait "$p1"
wait "$p2"
echo "wave1 done $(date -Is)"

p1=$(launch 1 runs/dmd_cartoon128_regonly --cfg 2 --lambda-reg 0.25 --no-dm)
p2=$(launch 2 runs/dmd_cartoon128_cfg1 --cfg 1 --lambda-reg 0.25)
wait "$p1"
wait "$p2"
echo "wave2 done $(date -Is)"

p1=$(launch 1 runs/dmd_cartoon128_cfg4 --cfg 4 --lambda-reg 0.25)
p2=$(launch 2 runs/dmd_cartoon128_dmd2 --cfg 2 --lambda-reg 0.25 --fake-steps 4)
wait "$p1"
wait "$p2"
echo "cartoon128 queue done $(date -Is)"

# The 64px queue stopped after its first pair of runs. Finish the other four
# once the 128 distillations have released the cards.
common=(
  --teacher runs/teacher_cartoon/last.pt
  --pairs runs/pairs_cartoon.pt
  --data runs/vae_cartoon
  --vae runs/vae_cartoon/last.pt
  --iters 20000 --batch 32 --reg-batch 32
  --sample-every 2000 --save-every 4000
)
echo "resume 64px ablations $(date -Is)"
p1=$(launch 1 runs/dmd_cartoon_regonly --cfg 2 --lambda-reg 0.25 --no-dm)
p2=$(launch 2 runs/dmd_cartoon_cfg1 --cfg 1 --lambda-reg 0.25)
wait "$p1"
wait "$p2"
p1=$(launch 1 runs/dmd_cartoon_cfg4 --cfg 4 --lambda-reg 0.25)
p2=$(launch 2 runs/dmd_cartoon_dmd2 --cfg 2 --lambda-reg 0.25 --fake-steps 4)
wait "$p1"
wait "$p2"
echo "64px ablations done $(date -Is)"
