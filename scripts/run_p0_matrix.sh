#!/usr/bin/env bash
# run_p0_matrix.sh — 按修正后的协议重跑全部前缀实验
#
# 产出（work/final/）：
#   p0_<scene>.json                 主结果（7 个固定/半固定顺序 + direct，λ=1）
#   p0_<scene>_lam{0,05,2}.json     λ 敏感性（只跑 direct）
#   p0_<scene>_crosscam.json        跨机位校准（calib cam2 → eval cam1，多视场景）
#   p0_s43_<scene>.json             第二训练种子的复现
#
# 用法： bash scripts/run_p0_matrix.sh [scene ...]
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
    sriracha) echo "work/sriracha_nat/ckpt_last.ckpt";;
    truck)    echo "work/truck_nat/ckpt_last.ckpt";;
    umbrella) echo "work/umbrella_nat/ckpt_last.ckpt";;
    jumping)  echo "work/jumping_nat/ckpt_last.ckpt";;
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
extra_of() {
  case $1 in truck|umbrella|jumping) echo "--dataset nvidia --factor 2";; *) echo "";; esac
}
seed43_ckpt_of() {
  case $1 in
    backpack) echo "work/backpack_s43/ckpt_last.ckpt";;
    truck)    echo "work/truck_s43/ckpt_last.ckpt";;
    *) echo "";;
  esac
}

GPU=0
wait_slot() {  # 最多 4 个并发
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
  GPU=$(( (GPU + 1) % 4 ))
}

run() {  # run <tag> <ckpt> <data> <extra> <outfile> <args...>
  local tag=$1 ck=$2 dd=$3 ex=$4 out=$5; shift 5
  wait_slot
  echo "[$(date +%H:%M:%S)] start $tag (gpu $GPU)"
  python3 scripts/r4d_prefix.py --ckpt "$ck" --data-dir "$dd" --rates "work/rates_${tag%%_*}.json" \
      --sens "work/sens_even_${tag%%_*}.json" --frames 32 --eval-parity odd --gpu $GPU $ex "$@" --out "$out" \
      > "work/final/$(basename "$out" .json).log" 2>&1 &
}

for sc in "${SCENES[@]}"; do
  CK=$(ckpt_of $sc); DD=$(data_of $sc); EX=$(extra_of $sc)
  run "$sc" "$CK" "$DD" "$EX" "work/final/p0_${sc}.json" --direct --direct-steps 24
  run "$sc" "$CK" "$DD" "$EX" "work/final/p0_${sc}_lam0.json" --direct --only-direct --lam 0
  run "$sc" "$CK" "$DD" "$EX" "work/final/p0_${sc}_lam05.json" --direct --only-direct --lam 0.5
  run "$sc" "$CK" "$DD" "$EX" "work/final/p0_${sc}_lam2.json" --direct --only-direct --lam 2
  case $sc in
    truck|umbrella|jumping)
      run "$sc" "$CK" "$DD" "$EX" "work/final/p0_${sc}_crosscam.json" \
          --direct --only-direct --calib-cam 2 --eval-cam 1 ;;
  esac
  S43=$(seed43_ckpt_of $sc)
  if [ -n "$S43" ] && [ -f "$S43" ]; then
    run "$sc" "$S43" "$DD" "$EX" "work/final/p0_s43_${sc}.json" --direct --direct-steps 24
  fi
done
wait
echo "[$(date +%H:%M:%S)] all done"
ls -la work/final/p0_*.json | tail -20
