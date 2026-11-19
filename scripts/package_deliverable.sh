#!/usr/bin/env bash
# package_deliverable.sh — 打包交付件并自证可编译
#
# 产出：
#   deliverable/TCSVT_SI_manuscript.pdf   主稿 PDF（来自 paper/build/main.pdf）
#   deliverable/latex_source.zip          源码包（main.tex + refs.bib + main.bbl + figures/ + BUILD.md）
#   deliverable/supplementary.zip         补充包（脚本 + 结果 CSV/JSON）
#
# 自证：把 latex_source.zip 解到 /tmp 独立编译，页数/词数必须与主构建一致。
# 用法： bash scripts/package_deliverable.sh [--skip-build]
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || exit 2

# 确定性打包：zip 会把每个文件的 mtime/顺序/权限写进包，导致"内容没改、md5 也变"。
# 这里固定 mtime（SOURCE_DATE_EPOCH）、固定权限、排序入包、strip 扩展属性，
# 使 zip 的 md5 只随内容变化 —— 才可以当冻结凭据用（见 scripts/freeze.sh）。
EPOCH="${SOURCE_DATE_EPOCH:-1700000000}"
zip_det() {  # $1=包名  $2=要装进去的目录名（相对 deliverable/）
  local out="$1" dir="$2"
  ( cd "$REPO/deliverable" || exit 1
    rm -f "$out"
    find "$dir" -exec touch -h -d "@$EPOCH" {} +
    chmod -R u+rwX,go+rX "$dir"
    umask 022
    find "$dir" -print | LC_ALL=C sort | TZ=UTC zip -q -X -@ "$out" )
}

if [ "${1:-}" != "--skip-build" ]; then
  ( cd "$REPO/paper" || exit 1
    pdflatex -interaction=nonstopmode main > .pass1.log 2>&1 || { tail -20 .pass1.log; exit 1; }
    bibtex main > .bib.log 2>&1 || { tail -20 .bib.log; exit 1; }
    pdflatex -interaction=nonstopmode main > .pass2.log 2>&1 || { tail -20 .pass2.log; exit 1; }
    pdflatex -interaction=nonstopmode main > .pass3.log 2>&1 || { tail -20 .pass3.log; exit 1; }
    mkdir -p build && cp main.pdf build/main.pdf
    rm -f .pass1.log .bib.log .pass2.log .pass3.log ) || exit 1
fi

PDF="$REPO/paper/build/main.pdf"
[ -f "$PDF" ] || { echo "✗ 缺 $PDF"; exit 1; }
pages=$(pdfinfo "$PDF" | awk '/^Pages/{print $2}')
words=$(pdftotext "$PDF" - | python3 -c "import sys; print(len(sys.stdin.read().split()))")

echo "== 1. 交付 PDF（$pages 页 / $words 词）=="
mkdir -p "$REPO/work/deliv_hist"
[ -f "$REPO/deliverable/TCSVT_SI_manuscript.pdf" ] && \
  cp "$REPO/deliverable/TCSVT_SI_manuscript.pdf" \
     "$REPO/work/deliv_hist/manuscript.$(date +%Y%m%d-%H%M).pdf"
cp "$PDF" "$REPO/deliverable/TCSVT_SI_manuscript.pdf"

echo "== 2. 源码目录 =="
L="$REPO/deliverable/latex"
rm -rf "$L"; mkdir -p "$L"
cp "$REPO/paper/main.tex" "$REPO/paper/refs.bib" "$L/"
[ -f "$REPO/paper/IEEEtran.cls" ] && cp "$REPO/paper/IEEEtran.cls" "$L/"
[ -f "$REPO/paper/build/main.bbl" ] && cp "$REPO/paper/build/main.bbl" "$L/"
[ -f "$REPO/paper/EXPECTED.json" ] && cp "$REPO/paper/EXPECTED.json" "$L/"
mkdir -p "$L/figures" "$L/tables"
cp "$REPO"/paper/tables/*.tex "$L/tables/" 2>/dev/null || true
# 只带正文真正 \includegraphics 的图，未引用的旧图不进投稿包
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$REPO/paper/main.tex" \
  | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u | while read -r fig; do
    [ -f "$REPO/paper/figures/$fig" ] && cp "$REPO/paper/figures/$fig" "$L/figures/" || echo "  ⚠ 缺图: $fig"
  done
cat > "$L/BUILD.md" <<EOF
# Build instructions

\`\`\`
pdflatex main
bibtex   main
pdflatex main
pdflatex main
\`\`\`

\`main.bbl\` is included, so a single \`pdflatex main && pdflatex main\` also produces
the complete PDF with the bibliography. The class file \`IEEEtran.cls\` is
included in this directory (official IEEE Transactions template, V1.8b
2015/08/26 with the 2023 font-size update); \`IEEEtran.bst\` comes from TeX Live
(package \`ieeetran\`).

Engine: pdflatex. Output: \`main.pdf\`, $pages pages, US Letter.
\`IEEEtran.cls\` (the official IEEE Transactions template file, version V1.8b,
2015/08/26, with the 2023 font-size update) is included next to \`main.tex\`, so
the class used here is the one shipped with the paper.
EOF

echo "== 3. 打包 =="
zip_det latex_source.zip latex && echo "  latex_source.zip $(du -h deliverable/latex_source.zip | cut -f1)"

echo "== 4. 补充包同步 =="
S="$REPO/deliverable/supplementary"
rm -rf "$S"
mkdir -p "$S/scripts" "$S/results/raw" "$S/paper/tables" "$S/paper/figures" "$S/legacy"
cp "$REPO/SUPP_README.md" "$S/README.md"
for f in "$REPO"/scripts/*.py "$REPO"/scripts/*.sh; do cp "$f" "$S/scripts/"; done
cp "$REPO/paper/main.tex" "$REPO/paper/refs.bib" "$REPO/paper/IEEEtran.cls" "$S/paper/" 2>/dev/null || true
cp "$REPO"/paper/tables/*.tex "$S/paper/tables/"
for f in $(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$REPO/paper/main.tex" | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u); do
  cp "$REPO/paper/figures/$f" "$S/paper/figures/" 2>/dev/null || echo "  ⚠ 缺图 $f"
done
# 当前协议的原始结果（主表/主图全部由这些 JSON 生成）
cp "$REPO"/work/final/p0_*.json "$S/results/raw/" 2>/dev/null
cp "$REPO"/work/final/p0_alloc_*.json "$REPO"/work/final/block_*.json "$REPO"/work/motion_*.json \
   "$REPO"/work/final/p0_*_lam*.json \
   "$REPO"/work/final/p0_*_crosscam.json "$REPO"/work/final/p0_s43_*.json "$S/results/raw/" 2>/dev/null
cp "$REPO"/work/rates_*.json "$REPO"/work/sens_p0_*.json "$REPO"/work/sens_even_*.json \
   "$REPO"/work/mask_manifest_*.json "$REPO"/work/final/calib_cameras_*.json \
   "$REPO"/work/final/adaptive_*.json "$REPO"/work/final/risk_*.json \
   "$REPO"/work/final/ref_frames_*.json "$REPO"/work/final/gtq_*.json \
   "$REPO"/work/final/deliv_*.json "$REPO"/work/final/blockrisk_*.json \
   "$REPO"/work/final/ref_frames_block_*.json "$S/results/raw/" 2>/dev/null
cp "$REPO"/reports/*.csv "$S/results/" 2>/dev/null
mkdir -p "$S/legacy/results" "$S/legacy/scripts"
cp "$REPO"/reports/legacy/*.csv "$S/legacy/results/" 2>/dev/null || true
[ -f "$REPO/reports/legacy/MANIFEST.csv" ] && cp "$REPO/reports/legacy/MANIFEST.csv" "$S/legacy/results/"
cp "$REPO"/scripts/legacy/* "$S/legacy/scripts/" 2>/dev/null || true
[ -f "$REPO/paper/EXPECTED.json" ] && cp "$REPO/paper/EXPECTED.json" "$S/paper/"
# 旧协议结果单独归档，并写明不作主结果
cp "$REPO"/work/final/prefix_*.json "$REPO"/work/final/prefix4_*.json \
   "$REPO"/work/final/prefixlp_*.json "$REPO"/work/final/prefix_s43_*.json "$S/legacy/" 2>/dev/null
cat > "$S/legacy/LEGACY.md" <<'EOT'
# Legacy artefacts (not used by the paper)

Files here come from the earlier protocol: cumulative bytes were counted as the
serialized size of the current model state instead of the sum of transmitted
packets, and fixed schedules were scored at level boundaries. They are kept only
for traceability; every number in the manuscript is generated from
`results/raw/p0_*.json` and `results/raw/p0_alloc_*.json`.
EOT
zip_det supplementary.zip supplementary && echo "  supplementary.zip $(du -h deliverable/supplementary.zip | cut -f1)"

echo "== 5. 自证：从 latex_source.zip 独立编译 =="
T=$(mktemp -d /tmp/deliv_check.XXXXXX)
unzip -q "$REPO/deliverable/latex_source.zip" -d "$T"
cd "$T/latex" || exit 1
pdflatex -interaction=nonstopmode main >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main >/dev/null 2>&1
rp=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
rw=$(pdftotext main.pdf - 2>/dev/null | python3 -c "import sys; print(len(sys.stdin.read().split()))")
echo "  独立编译：$rp 页 / $rw 词（主构建 $pages 页 / $words 词）"
rm -rf "$T"
if [ "$rp" = "$pages" ] && [ "$rw" = "$words" ]; then
  echo "✓ 交付件自证通过"
  exit 0
fi
echo "✗ 交付件与主构建不一致"
exit 1
