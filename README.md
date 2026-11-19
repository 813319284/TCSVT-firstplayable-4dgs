# First-Playable Progressive Delivery of 4D Gaussian Scenes from Real-World Video

Reproduction package for the manuscript submitted to the *IEEE Transactions on
Circuits and Systems for Video Technology* special issue on spatial intelligence.

A frozen factorised 4D Gaussian reconstruction is encoded once as a base layer plus
bit-plane refinement packets, so the scene becomes a single truncatable stream that
decodes and renders as data arrives. Ranking the packets by the shortfall they remove
from the full-scene and dynamic-region targets per transmitted byte brings the
first playable frame to the client with **16.6-50.3% fewer bytes** than the strongest
of six fixed delivery schedules, on five real captures, without retraining the model.

Everything needed to re-derive the numbers in the paper is in this repository.
Nothing here needs a GPU: the schedules, byte counts and statistics are stored as
JSON and re-checked by two independent scripts.

## Quick start (CPU only, no checkpoints)

```bash
python3 scripts/check_numbers.py    # 676 checks against the manuscript text and tables
python3 scripts/audit_numbers.py    # 188 independent re-derivations, re-run from scratch
bash    scripts/build_check.sh      # rebuilds the paper from source (LaTeX); expects 12 pages / 7528 words
```

`check_numbers.py` re-derives every number quoted in the text from `results/`;
`audit_numbers.py` recomputes the 40 qualifying prefixes, the BD-PSNR values, the
matched-budget pairs and the table cells by a different route. Both exit non-zero on
any mismatch.

## Layout

```
paper/        LaTeX source (main.tex, refs.bib, IEEEtran.cls, tables/, figures/) and the built manuscript.pdf
scripts/      every script that produced a number, table or figure in the paper
results/      raw per-run JSON (one file per scene and configuration) and the aggregated CSV tables
```

## Reproducing the tables and figures

| Item | Data | Regenerate |
|---|---|---|
| Table II (scenes, protocol, codestream) | `results/raw/p0_<scene>.json` | `python3 scripts/make_tex_tables.py` |
| Table III (component sensitivity) | `results/raw/sens_*_<scene>.json` | `python3 scripts/make_tex_tables.py` |
| Table IV (first-playable bytes, five scenes) | `results/raw/p0_<scene>.json` | `python3 scripts/make_tex_tables.py` |
| Table V (one stream against per-budget allocation) | `results/raw/p0_alloc_<scene>.json` | `python3 scripts/make_p0_tables.py` |
| Table VI (per-packet baselines) | `results/raw/adaptive_*_<scene>.json` | `python3 scripts/make_adaptive_table.py` |
| Figs. 1 and 3 | `results/raw/p0_*.json` | `python3 scripts/make_figures_p0.py`, `scripts/make_fig_samebudget.py` |
| Figs. 4-5 (qualitative, cross-view) | decoded prefixes | `python3 scripts/render_compare.py`, `python3 scripts/make_fig_crossview.py` |
| Tail metrics (Sec. V-F) | `results/raw/risk_*`, `ref_frames_*` | `python3 scripts/tail_metrics.py --table` |
| Delivery replay, 125 runs (Sec. VI) | `results/table_delivery_system.csv` | `python3 scripts/delivery_sim.py` |
| Absolute quality against the captures (Sec. V-I) | `results/table_gt_quality.csv` | `python3 scripts/eval_gt_quality.py` |
| Robustness / camera split (Sec. V) | `results/table_robustness.csv`, `table_camera.csv` | `python3 scripts/stats_matrix.py` |

`REPRODUCE.md` gives the full command line for each of them.

## Re-running the experiments (needs the captures and the weights)

The scene captures and the trained checkpoints (~10 GB) are **not** redistributed.
To re-run a schedule end to end, put them under a single root and point `SCENE_ROOT`
at it:

```
$SCENE_ROOT/scenes/backpack          $SCENE_ROOT/scenes/sriracha-tree
$SCENE_ROOT/Truck  $SCENE_ROOT/Umbrella  $SCENE_ROOT/Jumping
$SCENE_ROOT/checkpoints/A_nat/ckpt_last.ckpt
$SCENE_ROOT/_src/shape-of-motion-main        # original reconstruction code
```

Then, for example:

```bash
export TCSVT_ROOT=$PWD SCENE_ROOT=/path/to/data
python3 scripts/r4d_prefix.py --scene backpack --split parity --direct-full --direct-steps 44
bash    scripts/run_p0_matrix.sh          # all five scenes, all six fixed schedules and the adaptive order
```

`scripts/r4d_rates.py`, `sens_grid.py`, `packets.py` and `alloc_search_p0.py` run on
CPU; the rest need one CUDA GPU (all runs in the paper used a single NVIDIA L40).

## Environment

Python 3.10, PyTorch 2.7 (CUDA 12.8), `gsplat` 1.4 (JIT-compiled on first use),
`lpips`, `scipy`, `pandas`, `matplotlib`, `Pillow`. See `requirements.txt`.
LaTeX is only needed for `paper/` (pdflatex, standard IEEEtran class included).

## What this package does not contain

* The trained checkpoints and the original captures (licences and size; see above).
* Re-rendered images: every PSNR/SSIM/LPIPS value in the paper is stored per frame in
  `results/`, but re-rendering needs the weights.
* The earlier protocol's artefacts. `results/` holds only the protocol used in the paper.

## License and citation

Code: MIT (`LICENSE`). Manuscript text and figures: (c) the authors.
Citation metadata is in `CITATION.cff`; a BibTeX entry will be added once the paper
is published.
