#!/usr/bin/env python3
"""make_fig_samebudget.py — 同字节对照组图（新 Fig. 1 / Fig. 3）。

读 work/samebudget/<scene>/f<idx>_{best,ours,ref}.png 与 p0_<scene>.json 的字节/PSNR，
按「最强固定顺序 @ B_ours / Ours @ B_ours / 16-bit」三列 × 五场景排版；所有面板用
同一固定裁剪矩形（不按方法挑区域）。

用法： python3 scripts/make_fig_samebudget.py
"""
from __future__ import annotations

import os

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

W = os.environ.get("TCSVT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
NICE = {"backpack": "backpack", "sriracha": "sriracha-tree", "truck": "Truck",
        "umbrella": "Umbrella", "jumping": "Jumping"}


def crop(img, frac=(0.15, 0.55, 0.85, 0.95)):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = frac
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main() -> int:
    rows = []
    for sc in SCENES:
        p = f"{W}/work/samebudget/{sc}"
        meta = json.load(open(f"{p}/meta.json"))
        d = json.load(open(f"{W}/work/final/p0_{sc}.json"))
        B = meta["budget_bytes"] / 1e6
        rows.append((sc, p, meta, d, B))

    # ---------- Fig. A：两场景 + 全幅 ----------
    for name, scenes in [("fig_samebudget_hero", ["truck", "umbrella"])]:
        fig, axes = plt.subplots(len(scenes), 3, figsize=(7.2, 2.3 * len(scenes)))
        for r, sc in enumerate(scenes):
            _, p, meta, d, B = next(x for x in rows if x[0] == sc)
            f = sorted(meta["frames"])[0]
            for c, (tag, lab) in enumerate([("best", f"best fixed ({meta['best_order']}) @ {B:.2f} MB"),
                                            ("ours", f"ours @ {B:.2f} MB"),
                                            ("ref", "16-bit reference")]):
                img = np.asarray(Image.open(f"{p}/f{f}_{tag}.png"))
                ax = axes[r, c]
                ax.imshow(img)
                ax.set_title(lab, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(NICE[sc], fontsize=8)
        fig.tight_layout(pad=0.3)
        out = f"{W}/paper/figures/{name}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=330)
        print("wrote", out)

    # ---------- Fig. B：五场景三列 ----------
    fig, axes = plt.subplots(len(SCENES), 3, figsize=(7.2, 1.55 * len(SCENES)))
    for r, (sc, p, meta, d, B) in enumerate(rows):
        f = sorted(meta["frames"])[0]
        for c, tag in enumerate(["best", "ours", "ref"]):
            img = np.asarray(Image.open(f"{p}/f{f}_{tag}.png"))
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["best fixed", "ours", "16-bit"][c], fontsize=7)
        axes[r, 0].set_ylabel(f"{NICE[sc]}\n{B:.2f} MB", fontsize=7)
    fig.tight_layout(pad=0.25)
    out = f"{W}/paper/figures/fig_samebudget_all.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=330)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
