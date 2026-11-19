#!/usr/bin/env bash
# 时间块协议 + 走满路径（--direct-full）：五场景，保证首达在评测帧上判定
set -euo pipefail
E=~/miniconda3/envs/trust4dgs
export PATH=$E/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "${TCSVT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ckpt_of() { case $1 in backpack) echo "${SCENE_ROOT:-$PWD/data}/checkpoints/A_nat/ckpt_last.ckpt";; *) echo "work/$1_nat/ckpt_last.ckpt";; esac; }
data_of() { case $1 in
  backpack) echo "${SCENE_ROOT:-$PWD/data}/scenes/backpack";;
  sriracha) echo "${SCENE_ROOT:-$PWD/data}/scenes/sriracha-tree";;
  truck) echo "${SCENE_ROOT:-$PWD/data}/Truck";;
  umbrella) echo "${SCENE_ROOT:-$PWD/data}/Umbrella";;
  jumping) echo "${SCENE_ROOT:-$PWD/data}/Jumping";; esac; }
extra_of() { case $1 in truck|umbrella|jumping) echo "--dataset nvidia --factor 2";; *) echo "";; esac; }
GPU=0; pids=()
for sc in backpack sriracha truck umbrella jumping; do
  GPU=$(( (GPU + 1) % 4 ))
  python3 scripts/r4d_prefix.py --ckpt "$(ckpt_of $sc)" --data-dir "$(data_of $sc)" \
    --rates "work/rates_${sc}.json" --sens "work/sens_even_${sc}.json" \
    --frames 32 --eval-parity odd --gpu $GPU $(extra_of $sc) --split block \
    --direct --direct-full --direct-steps 44 \
    --out "work/final/block_${sc}.json" > "work/final/block_${sc}.log" 2>&1 &
  pids+=($!)
  [ ${#pids[@]} -ge 4 ] && { wait "${pids[0]}" || exit 1; pids=("${pids[@]:1}"); }
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "✗ 有失败"; exit 1; }
echo done
