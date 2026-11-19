#!/usr/bin/env bash
# cleanroom.sh — 两道门禁的"reviewer view"入口，两种模式自动选：
#
#   模式 A（打包者，在仓库里跑）：找到 deliverable/supplementary.zip → 解到临时目录，
#            用包内脚本重跑三绿，确认"拿到 ZIP 能复现"。
#   模式 B（reviewer，在解压后的 supplementary/ 根目录跑）：没有 ZIP 就地把当前目录当待测对象，
#            同样跑三绿（README 里承诺的三条命令，一条命令等价）。
#
# 用法： bash scripts/cleanroom.sh [--keep]
# 退出码：0 = 三绿；非 0 = 有门禁红（缺失输入也会红，不静默跳过）

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TCSVT_ROOT:-${REPO:-$(dirname "$HERE")}}"
ZIP="$REPO/deliverable/supplementary.zip"

run_checks() {  # $1 = 待测根目录（含 scripts/ 与 paper/ 与 results/）
  local ROOT="$1"
  [ -f "$ROOT/scripts/check_numbers.py" ] || { echo "✗ $ROOT 下没有 scripts/check_numbers.py"; return 2; }
  [ -f "$ROOT/paper/main.tex" ]           || { echo "✗ $ROOT 下没有 paper/main.tex"; return 2; }

  echo "== 1. 数字门禁 =="
  local out line
  out=$(cd "$ROOT" && TCSVT_ROOT="$PWD" python3 scripts/check_numbers.py 2>&1) || true
  line=$(echo "$out" | grep -m1 "检查项" || true)
  echo "  ${line:-（无输出）}"
  echo "$out" | grep "读入文件" | sed 's/^/  /' || true
  case "$line" in
    *"失败 0"*) : ;;
    *) echo "✗ check_numbers 未全绿"; echo "$out" | tail -12; return 1 ;;
  esac

  echo "== 2. 独立审计 =="
  out=$(cd "$ROOT" && TCSVT_ROOT="$PWD" python3 scripts/audit_numbers.py 2>&1) || true
  line=$(echo "$out" | tail -1)
  echo "  $line"
  case "$line" in
    *"全部一致"*) : ;;
    *) echo "✗ audit_numbers 未全绿"; echo "$out" | tail -12; return 1 ;;
  esac

  echo "== 3. 从零编译 =="
  out=$(cd "$ROOT" && TCSVT_ROOT="$PWD" bash scripts/build_check.sh 2>&1) || true
  echo "$out" | tail -5 | sed 's/^/  /'
  case "$out" in
    *"build_check 通过"*) : ;;
    *) echo "✗ build_check 失败"; return 1 ;;
  esac
  echo "✓ 三绿"
}

if [ -f "$ZIP" ]; then
  TMP="$(mktemp -d /tmp/cleanroom.XXXXXX)"
  if [ "${1:-}" = "--keep" ]; then trap 'echo "  保留 $TMP"' EXIT; else trap 'rm -rf "$TMP"' EXIT; fi
  echo "== 模式 A：解包复验 $ZIP =="
  unzip -q "$ZIP" -d "$TMP"
  ROOT="$TMP/supplementary"
  [ -d "$ROOT" ] || { echo "✗ 包内没有 supplementary/ 目录"; exit 1; }
  echo "  文件数 $(find "$ROOT" -type f | wc -l)"
  run_checks "$ROOT"
  echo "✓ cleanroom 通过（reviewer view：解包后三绿）"
else
  echo "== 模式 B：就地复验（没有 $ZIP，按解压后的补充包根目录处理）=="
  run_checks "$REPO"
  echo "✓ cleanroom 通过（就地三绿）"
fi
