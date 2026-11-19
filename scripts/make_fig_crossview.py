#!/usr/bin/env python3
"""make_fig_crossview.py — 跨视角组图（同时间、cam1/2/3，前缀 vs 16-bit）。"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

W = f"{_ROOT}"
SCENES = ["truck", "jumping"]
CAMS = [1, 2, 3]

fig, axes = plt.subplots(len(SCENES) * 2, len(CAMS), figsize=(7.0, 4.4))
for r, sc in enumerate(SCENES):
    d = json.load(open(f"{W}/work/final/p0_{sc}.json"))
    mb = d["summary"]["direct"]["MB"]
    for c, cam in enumerate(CAMS):
        for k, tag in enumerate(["ours", "ref"]):
            p = f"{W}/work/crossview/{sc}/cam{cam}_{tag}.png"
            ax = axes[r * 2 + k, c]
            if os.path.exists(p):
                ax.imshow(np.asarray(Image.open(p)))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and k == 0:
                ax.set_title(f"camera {cam}", fontsize=7)
        axes[r * 2, 0].set_ylabel(f"{sc} ours\n{mb:.2f} MB" if c == 0 else "", fontsize=6.5)
        axes[r * 2 + 1, 0].set_ylabel(f"{sc} 16-bit" if c == 0 else "", fontsize=6.5)
fig.tight_layout(pad=0.2)
out = f"{W}/paper/figures/fig_crossview.pdf"
fig.savefig(out, bbox_inches="tight", dpi=330)
print("wrote", out)
