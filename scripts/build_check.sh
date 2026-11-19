#!/usr/bin/env bash
# build_check.sh — 从零编译校验（投稿件必须过这一关）
#
# 为什么需要：2026-11-14 发现 `\end{abstract}` 被吞掉，但 PDF 靠 LaTeX 的恢复机制
# 仍能正常出页——肉眼看不出、`tex.py build` 也不报错，只有从零编译的日志里有
# `! LaTeX Error`。这个脚本把"从零编译 + 0 错误 + 引用全解析"固化成门禁。
#
# 2026-11-19（编译门禁加固）加固：
#   · REPO 默认取脚本上级目录，可用 TCSVT_ROOT 覆盖 —— 解压到任何目录都能跑
#   · set -euo pipefail，每一趟 pdflatex / bibtex 的返回码都校验
#   · 参考页数/词数：有 paper/build/main.pdf 就比它，没有就读 paper/EXPECTED.json；
#     两者都没有 → 直接 FAIL（不再用"0 词"冒充一致）
#
# 用法： bash scripts/build_check.sh [--keep]
# 退出码：0 = 通过；非 0 = 有 LaTeX 错误 / 引用未解析 / 无法核对

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TCSVT_ROOT:-${REPO:-$(dirname "$HERE")}}"
SRC="$REPO/paper"
[ -f "$SRC/main.tex" ] || { echo "✗ 找不到 $SRC/main.tex（用 TCSVT_ROOT 指向仓库或解压包根目录）"; exit 2; }
TMP="$(mktemp -d /tmp/build_check.XXXXXX)"
if [ "${1:-}" = "--keep" ]; then
  trap 'echo "  保留 $TMP"' EXIT
else
  trap 'rm -rf "$TMP"' EXIT
fi

echo "== 1. 复制源（不含 build/ 与 main.bbl） =="
rsync -a --exclude build --exclude main.bbl --exclude '*.aux' --exclude '*.log' "$SRC"/ "$TMP"/
ls "$TMP" | tr '\n' ' '; echo

echo "== 2. 四趟编译（pdflatex → bibtex → pdflatex ×2） =="
cd "$TMP" || exit 2
pass=0
run() {  # $1 = 趟名；余下 = 命令
  local name="$1"; shift
  if ! "$@" ; then
    echo "  ✗ 第 $name 趟返回非零"
    for f in pass*.log bibtex.log; do
      [ -f "$f" ] && { echo "  --- $f 首条错误 ---"; grep -m3 -A2 '^!' "$f" || true; }
    done
    exit 1
  fi
}
pass=1; run "$pass(pdflatex)" pdflatex -interaction=nonstopmode main > pass1.log 2>&1
pass=bibtex; run "bibtex" bibtex main > bibtex.log 2>&1
pass=2; run "$pass(pdflatex)" pdflatex -interaction=nonstopmode main > pass2.log 2>&1
pass=3; run "$pass(pdflatex)" pdflatex -interaction=nonstopmode main > pass3.log 2>&1

echo "== 3. 检查 =="
errs=$(grep -c '^!' pass3.log || true)
echo "  LaTeX 错误（^!）: $errs"
if [ "$errs" != "0" ]; then
  echo "  ✗ 有错误，前 5 条："
  grep -A2 '^!' pass3.log | head -20
  exit 1
fi

undef=$(grep -c "Citation .* undefined\|Reference .* undefined" pass3.log || true)
echo "  未定义引用: $undef"
[ "$undef" != "0" ] && { grep "undefined" pass3.log | head -5; exit 1; }

info=$(pdfinfo main.pdf)
echo "  Title: $(echo "$info" | awk -F': *' '/^Title/{print $2}')"
case "$info" in
  *First-Playable*) : ;;
  *) echo "  ✗ PDF Title 元数据不是 First-Playable 标题"; exit 1 ;;
esac
case "$info" in
  *Chen\ Yang*|*"Chen Yang"*) : ;;
  *) echo "  ✗ PDF Author 元数据缺失"; exit 1 ;;
esac

q=$(pdftotext main.pdf - 2>/dev/null | grep -c '??' || true)
echo "  PDF 里的 ?? 占位: $q"
[ "$q" != "0" ] && exit 1

words_of() {  # $1 = PDF；口径固定：pdftotext（默认模式）→ python split()，并记录 poppler 版本
  pdftotext "$1" - | python3 -c "import sys; print(len(sys.stdin.read().split()))"
}
rp=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
rw=$(words_of main.pdf)
echo "  pdftotext: $(pdftotext -v 2>&1 | head -1 | cut -d' ' -f1-3)（词数口径：python split()）"
echo "  页数: $rp"
echo "  字体嵌入标记: $(pdffonts main.pdf | awk 'NR>2 && $(NF-4)!="yes"{n++} END{print (n>0?"no":"yes")}')"

echo "== 4. 与参考构建比对 =="
REF="$REPO/paper/build/main.pdf"
EXP="$REPO/paper/EXPECTED.json"
if [ -f "$REF" ]; then
  pages=$(pdfinfo "$REF" | awk '/^Pages/{print $2}')
  words=$(words_of "$REF")
  echo "  参考=paper/build/main.pdf：$pages 页 / $words 词"
elif [ -f "$EXP" ]; then
  pages=$(python3 -c "import json,sys;d=json.load(open('$EXP'));print(d['pages'])")
  words=$(python3 -c "import json,sys;d=json.load(open('$EXP'));print(d['words'])")
  echo "  参考=paper/EXPECTED.json：$pages 页 / $words 词"
else
  echo "  ✗ 既没有 paper/build/main.pdf 也没有 paper/EXPECTED.json —— 无法核对页数与词数"
  exit 1
fi
echo "  从零编译：$rp 页 / $rw 词"
if [ "$rp" != "$pages" ] || [ "$rw" != "$words" ]; then
  echo "✗ 与参考不一致（页 $rp≠$pages 或 词 $rw≠$words）"
  echo "  词数是抽取器敏感量：请确认用的是同一 pdftotext 与同一口径（本脚本：默认模式 + python split()）"
  exit 1
fi
echo "✓ build_check 通过（0 错误 / $rp 页 / $rw 词，与参考一致）"
