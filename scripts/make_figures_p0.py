#!/usr/bin/env python3
"""make_figures_p0.py — 重跑后的定量图（全部从 p0_*.json 生成，不手填数字）。

Fig. first-playable: 每个场景 8 条顺序的 ΔPSNR（相对 16-bit 参考）随累计传输字节的变化，
  虚线为 −0.5 dB 门槛，圆点为各顺序的达标点。
Fig. scalability: 单流与按预算优化分配在 12 个匹配预算上的差（全图 / 动态区域）。

用法： python3 scripts/make_figures_p0.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = os.environ.get("TCSVT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
NICE = {"backpack": "backpack", "sriracha": "sriracha-tree", "truck": "Truck",
        "umbrella": "Umbrella", "jumping": "Jumping"}
FIXED = ["uniform", "dynfirst", "posfirst", "roundrobin", "sens", "oneshot", "mpprefix"]
ORDERS = ["uniform", "dynfirst", "posfirst", "roundrobin", "sens", "oneshot", "mpprefix", "direct"]
SHOW = ["direct"]   # 主图只画 ours + 该场景最强固定 + uniform
STYLE = {
    "uniform":    dict(color="#9e9e9e", ls="-",  lw=1.2, label="uniform bit-plane"),
    "dynfirst":   dict(color="#ff9800", ls="-",  lw=1.1, label="dynamic-first"),
    "posfirst":   dict(color="#4caf50", ls="-",  lw=1.1, label="position-first"),
    "roundrobin": dict(color="#00acc1", ls="-",  lw=1.1, label="round-robin"),
    "sens":       dict(color="#9c27b0", ls="-",  lw=1.1, label="linear sensitivity"),
    "oneshot":    dict(color="#795548", ls="--", lw=1.1, label="frozen single score"),
    "mpprefix":   dict(color="#e91e63", ls="-.", lw=1.1, label="multi-probe fixed"),
    "direct":     dict(color="#1f4e79", ls="-",  lw=2.0, label="prefix-adaptive (ours)"),
}


def fig_prefix():
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.1), sharey=False)
    for ax, sc in zip(axes, SCENES):
        d = json.load(open(f"{W}/work/final/p0_{sc}.json"))
        ref = d["ref_psnr"]
        valid = {k: v["MB"] for k, v in d["summary"].items() if v.get("MB")}
        best = min((valid[k], k) for k in FIXED if k in valid)[1]
        show = ["direct", best, "uniform"]
        for od in show:
            rows = sorted([r for r in d["results"] if r["order"] == od], key=lambda r: r["bytes"])
            if not rows:
                continue
            x = [r["bytes"] / 1e6 for r in rows]
            y = [r["psnr_first"] - ref for r in rows]
            st = dict(STYLE[od]) if od in STYLE else dict(color="#e91e63", ls="-.", lw=1.2)
            st.pop("label", None)
            if od == best:
                st["color"], st["ls"] = "#e91e63", "-."
            ax.plot(x, y, **st)
            hit = next((r for r in rows if r["psnr_first"] >= ref - 0.5
                        and r["psnr_first_fg"] >= d["summary"]["uniform"]["ref_fg"] - 0.5), None)
            if hit is not None:
                ax.plot([hit["bytes"] / 1e6], [hit["psnr_first"] - ref], "o", ms=4,
                        color=st["color"], zorder=5)
        ax.axhline(-0.5, color="k", lw=0.8, ls=":")
        ax.set_title(NICE[sc], fontsize=8.5)
        ax.set_xlabel("transmitted MB", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(r"$\Delta$PSNR vs. 16-bit (dB)", fontsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE["direct"]["color"], lw=2.0, label="prefix-adaptive (ours)"),
               Line2D([], [], color="#e91e63", ls="-.", lw=1.2, label="strongest fixed (per scene)"),
               Line2D([], [], color="#9e9e9e", lw=1.2, label="level-synchronous")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = f"{W}/paper/figures/fig_firstplayable.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_scalable():
    xs, df, dg, names = [], [], [], []
    for sc in SCENES:
        p = f"{W}/work/final/p0_alloc_{sc}.json"
        if not os.path.exists(p):
            continue
        al = json.load(open(p)); st = json.load(open(f"{W}/work/final/p0_{sc}.json"))
        direct = [r for r in st["results"] if r["order"] == "direct"]
        seen = set()
        for b in al["budgets"]:
            B = b["bytes"]; key = (sc, round(B / 1e6, 2))
            if key in seen:
                continue
            seen.add(key)
            cand = [r for r in direct if r["bytes"] <= B]
            if not cand:
                continue
            s = cand[-1]
            xs.append(B / 1e6); df.append(s["psnr_first"] - b["eval_full"])
            dg.append(s["psnr_first_fg"] - b["eval_fg"]); names.append(sc)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(xs, df, s=22, color="#1f4e79", label="full mask", zorder=3)
    ax.scatter(xs, dg, s=22, color="#e07a2f", marker="^", label="dynamic region", zorder=3)
    ax.set_xlabel("budget (MB, transmitted)", fontsize=7)
    ax.set_ylabel(r"stream $-$ allocation (dB)", fontsize=7)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    out = f"{W}/paper/figures/fig_scalable.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, len(xs), "budgets")


if __name__ == "__main__":
    fig_prefix()
    fig_scalable()
