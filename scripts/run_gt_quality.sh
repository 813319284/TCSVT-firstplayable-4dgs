#!/usr/bin/env bash
# 五场景 × 六种状态（16-bit / 最强固定 / full-gain / joint-gain / ours / tail-cvar）的真实 GT 绝对画质
set -euo pipefail
E=~/miniconda3/envs/trust4dgs
export PATH=$E/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "${TCSVT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU=0; pids=()
for sc in backpack sriracha truck umbrella jumping; do
  GPU=$(( (GPU + 1) % 4 ))
  python3 scripts/eval_gt_quality.py --scene "$sc" --gpu $GPU > "work/final/gtq_${sc}.log" 2>&1 &
  pids+=($!)
  [ ${#pids[@]} -ge 4 ] && { wait "${pids[0]}" || exit 1; pids=("${pids[@]:1}"); }
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "✗ 有场景失败"; exit 1; }
for sc in backpack sriracha truck umbrella jumping; do echo "== $sc"; tail -7 "work/final/gtq_${sc}.log"; done
