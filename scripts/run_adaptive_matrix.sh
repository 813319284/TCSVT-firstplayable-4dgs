#!/usr/bin/env bash
# run_adaptive_matrix.sh — 逐包重估的强自适应基线（强自适应基线）
#
# 三种策略只换"候选排序目标"，其余完全相同（同一 checkpoint、同一包集合、同一标定/评测帧、
# 同一真实字节账本、同一双区域接受判据、同一走满路径）：
#   deficit     = 论文式(3)，质量缺口减少量的正部/字节（Ours，回归用）
#   full_gain   = 全图边际 PSNR 增益/字节（标准逐包贪心）
#   joint_gain  = 双区域边际 PSNR 增益之和/字节（不做缺口截断）
#
# 用法： bash scripts/run_adaptive_matrix.sh [scene ...]
set -uo pipefail
E=~/miniconda3/envs/trust4dgs
export PATH=$E/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "${TCSVT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 2

SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(backpack sriracha truck umbrella jumping)

ckpt_of() {
  case $1 in
    backpack) echo "${SCENE_ROOT:-$PWD/data}/checkpoints/A_nat/ckpt_last.ckpt";;
    *) echo "work/$1_nat/ckpt_last.ckpt";;
  esac
}
data_of() {
  case $1 in
    backpack) echo "${SCENE_ROOT:-$PWD/data}/scenes/backpack";;
    sriracha) echo "${SCENE_ROOT:-$PWD/data}/scenes/sriracha-tree";;
    truck)    echo "${SCENE_ROOT:-$PWD/data}/Truck";;
    umbrella) echo "${SCENE_ROOT:-$PWD/data}/Umbrella";;
    jumping)  echo "${SCENE_ROOT:-$PWD/data}/Jumping";;
  esac
}
extra_of() { case $1 in truck|umbrella|jumping) echo "--dataset nvidia --factor 2";; *) echo "";; esac; }

GPU=0
wait_slot() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; GPU=$(( (GPU + 1) % 4 )); }
run() {  # run <scene> <policy> <ckpt> <data> <extra>
  local sc=$1 pol=$2 ck=$3 dd=$4 ex=$5
  wait_slot
  echo "[$(date +%H:%M:%S)] start $sc/$pol (gpu $GPU)"
  python3 scripts/r4d_prefix.py --ckpt "$ck" --data-dir "$dd" \
      --rates "work/rates_${sc}.json" --sens "work/sens_even_${sc}.json" \
      --frames 32 --eval-parity odd --gpu $GPU $ex \
      --direct --direct-full --direct-steps 44 --adaptive-score "$pol" \
      --out "work/final/adaptive_${pol}_${sc}.json" \
      > "work/final/adaptive_${pol}_${sc}.log" 2>&1 &
}

for sc in "${SCENES[@]}"; do
  CK=$(ckpt_of $sc); DD=$(data_of $sc); EX=$(extra_of $sc)
  for pol in deficit full_gain joint_gain; do run "$sc" "$pol" "$CK" "$DD" "$EX"; done
done
wait
echo "[$(date +%H:%M:%S)] all done"
ls -la work/final/adaptive_*.json | tail -20
