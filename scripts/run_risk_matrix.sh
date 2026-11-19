#!/usr/bin/env bash
# run_risk_matrix.sh — 尾部风险（CVaR）评分与"计算-字节"两种策略（尾部风险评分）
#
#   tail_cvar         每包重估，目标=双区域逐帧缺口的 CVaR_alpha
#   tail_cvar_r3      同一目标，但每 3 包才重估一次（计算节省）
#   joint_gain_r3     最强自适应基线的同样降频版本（保证降频不是只对 Ours 有利）
#
# 与已有矩阵共用 checkpoint / 包集合 / 标定与评测帧 / 真实字节账本 / 双区域接受判据。
# 用法： bash scripts/run_risk_matrix.sh [scene ...]
set -euo pipefail
E=~/miniconda3/envs/trust4dgs
export PATH=$E/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "${TCSVT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(backpack sriracha truck umbrella jumping)

ckpt_of() { case $1 in backpack) echo "${SCENE_ROOT:-$PWD/data}/checkpoints/A_nat/ckpt_last.ckpt";; *) echo "work/$1_nat/ckpt_last.ckpt";; esac; }
data_of() { case $1 in
  backpack) echo "${SCENE_ROOT:-$PWD/data}/scenes/backpack";;
  sriracha) echo "${SCENE_ROOT:-$PWD/data}/scenes/sriracha-tree";;
  truck) echo "${SCENE_ROOT:-$PWD/data}/Truck";;
  umbrella) echo "${SCENE_ROOT:-$PWD/data}/Umbrella";;
  jumping) echo "${SCENE_ROOT:-$PWD/data}/Jumping";; esac; }
extra_of() { case $1 in truck|umbrella|jumping) echo "--dataset nvidia --factor 2";; *) echo "";; esac; }

GPU=0
wait_slot() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; GPU=$(( (GPU + 1) % 4 )); }
pids=()
launch() {  # launch <scene> <tag> <score> <refresh>
  local sc=$1 tag=$2 score=$3 rf=$4
  wait_slot
  echo "[$(date +%H:%M:%S)] start $sc/$tag (gpu $GPU)"
  python3 scripts/r4d_prefix.py --ckpt "$(ckpt_of "$sc")" --data-dir "$(data_of "$sc")" \
      --rates "work/rates_${sc}.json" --sens "work/sens_even_${sc}.json" \
      --frames 32 --eval-parity odd --gpu $GPU $(extra_of "$sc") \
      --direct --direct-full --direct-steps 44 --adaptive-score "$score" \
      --risk-alpha 0.9 --refresh "$rf" \
      --out "work/final/risk_${tag}_${sc}.json" > "work/final/risk_${tag}_${sc}.log" 2>&1 &
  pids+=($!)
}

for sc in "${SCENES[@]}"; do
  launch "$sc" tail_cvar      tail_cvar   1
  launch "$sc" tail_cvar_r3   tail_cvar   3
  launch "$sc" joint_gain_r3  joint_gain  3
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ $fail -eq 0 ] || { echo "✗ 有子任务失败，检查 work/final/risk_*.log"; exit 1; }
echo "[$(date +%H:%M:%S)] all done"
ls -la work/final/risk_*.json | tail -20
