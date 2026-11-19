#!/usr/bin/env bash
# sha256sums.sh — 给三个交付件生成完整 SHA-256 清单（完整 64 位，不用前缀）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TCSVT_ROOT:-${REPO:-$(dirname "$HERE")}}"
cd "$REPO"
OUT="deliverable/SHA256SUMS.txt"
sha256sum deliverable/TCSVT_SI_manuscript.pdf \
          deliverable/latex_source.zip \
          deliverable/supplementary.zip > "$OUT"
cat "$OUT"
echo "→ $OUT"
