#!/usr/bin/env bash
# 五场景 × 五策略 × 五 trace 的真实收发回放（trace 驱动 + 真实解码 + 实测一帧渲染）
set -euo pipefail
E=~/miniconda3/envs/trust4dgs
export PATH=$E/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "${TCSVT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU=0; pids=()
for sc in "$@"; do
  GPU=$(( (GPU + 1) % 4 ))
  python3 scripts/delivery_sim.py --scene "$sc" --policies all --traces all --gpu $GPU \
    > "work/final/deliv_${sc}.log" 2>&1 &
  pids+=($!)
  [ ${#pids[@]} -ge 4 ] && { wait "${pids[0]}" || exit 1; pids=("${pids[@]:1}"); }
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "✗ 有场景失败"; exit 1; }
echo "✓ 全部完成"
