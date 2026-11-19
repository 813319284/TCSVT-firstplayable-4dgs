# How each number in the paper is produced

Two kinds of work are involved. **Re-derivation** (CPU, seconds to a minute) uses the
stored JSON in `results/` and needs nothing else. **Re-running** (one GPU) regenerates
that JSON from the checkpoints and captures, see the end of this file.

All commands assume the repository root as working directory:

```bash
export TCSVT_ROOT=$PWD
```

## 1. Re-derivation from the stored results

| Command | Produces | Cross-checked by |
|---|---|---|
| `python3 scripts/make_tex_tables.py` | `paper/tables/{scenes,sens,firstplayable,scalable,adaptive}.tex` | `check_numbers.py` parses the table bodies |
| `python3 scripts/make_p0_tables.py` | `results/table_{firstplayable,lambda,seeds,crosscam}.csv` | `audit_numbers.py` |
| `python3 scripts/make_adaptive_table.py` | `results/table_adaptive_compare.csv` | `check_numbers.py` (Table VI cells and bold marks) |
| `python3 scripts/tail_metrics.py --table` | `results/table_tail.csv` (mean and 90%-frame thresholds, five scenes) | `check_numbers.py` |
| `python3 scripts/stats_matrix.py` | `results/table_{robustness,camera}.csv` (tolerance and camera splits) | `check_numbers.py` |
| `python3 scripts/make_figures_p0.py` | `paper/figures/fig_{firstplayable,scalable}.pdf` | figure existence and panel count |
| `python3 scripts/make_fig_samebudget.py` | `paper/figures/fig_samebudget_{hero,all}.pdf` | idem |
| `python3 scripts/make_fig_crossview.py` | `paper/figures/fig_crossview.pdf` (Fig. 5) | idem |

Then the two gates:

```bash
python3 scripts/check_numbers.py   # 676 checks: text, tables, semantics, packaging
python3 scripts/audit_numbers.py   # 188 checks: prefixes, BD-PSNR, matched budgets, sentence numbers
```

## 2. Re-running the schedules (GPU)

```bash
export SCENE_ROOT=/path/to/data      # scenes/…, checkpoints/…, _src/…
bash scripts/run_p0_matrix.sh        # five scenes × six fixed schedules × the adaptive order
bash scripts/run_block_full.sh       # timestamp-block protocol
bash scripts/run_risk_matrix.sh      # tail-CVaR and its equal-refresh controls
bash scripts/run_ref_frames.sh       # per-frame qualification (90%-frame thresholds)
bash scripts/run_delivery_traces.sh  # 125 sender→link→receiver replays at 1/2/5/10 Mb/s
bash scripts/run_gt_quality.sh       # PSNR/SSIM/LPIPS against the captures
bash scripts/run_adaptive_matrix.sh  # per-packet baselines (deficit / full-gain / joint-gain)
```

Each driver writes into `results/raw/`; re-running them replaces the stored JSON, after
which `check_numbers.py` must still pass unchanged — that is the reproducibility claim
of the paper.

## 3. Rebuilding the manuscript

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/EXPECTED.json` records the reference build: 12 pages, 7528 words
(`pdftotext` + `python split()`), produced with pdftotext 0.86.1.
