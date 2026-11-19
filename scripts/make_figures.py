#!/usr/bin/env python3
"""make_figures.py — 论文图 1（冷启动前缀曲线）与图 3（全序列 RD）。

输出到 paper/figures/*.pdf（矢量，pdflatex 可直接 \\includegraphics）。
用法： python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = f"{_ROOT}"
MANIFEST = json.load(open(os.path.join(W, "work", "manifest_final.json")))


def load(p):
    d = json.load(open(p if os.path.isabs(p) else os.path.join(W, p)))
    return {"MB": d["total_MB"], "full": d["full_psnr"], "fg": d.get("fg_psnr")}

FIG = os.path.join(W, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "legend.frameon": False, "figure.dpi": 200})

ORDER_STYLE = {"uniform": ("tab:gray", "s", "uniform bit-plane"),
               "dynfirst": ("tab:orange", "^", "dynamic-first"),
               "posfirst": ("tab:green", "v", "position-first"),
               "roundrobin": ("tab:cyan", "P", "round-robin"),
               "sens": ("tab:purple", "D", "measured-sensitivity"),
               "oneshot": ("tab:brown", "x", "frozen prefix-objective"),
               "direct": ("tab:blue", "o", "prefix-objective, per-step (ours)")}


def fig_prefix():
    scenes = list(MANIFEST)
    fig, axes = plt.subplots(1, len(scenes), figsize=(7.0, 1.78), sharey=False)
    for ax, scene in zip(axes, scenes):
        pr = json.load(open(os.path.join(W, MANIFEST[scene]["prefix"])))
        ref = pr["ref_psnr"]
        thr = ref - 0.5
        s = pr["summary"]
        for name, (color, marker, label) in ORDER_STYLE.items():
            rs = [r for r in pr["results"] if r["order"] == name]
            xs = np.array([r["MB"] for r in rs])
            ys = np.array([r["psnr_first"] - ref for r in rs])
            ax.plot(xs, ys, color=color, marker=marker, markersize=2.5, label=label)
            hit = next((r for r in rs if r["psnr_first"] >= thr), None)
            if hit is not None:
                ax.plot([hit["MB"]], [hit["psnr_first"] - ref], marker="*", color=color,
                        markersize=7, markeredgecolor="k", markeredgewidth=0.3)
        ax.axhline(-0.5, color="k", linestyle="--", linewidth=0.6)
        ys_all = [r["psnr_first"] - ref for r in pr["results"]]
        ax.set_ylim(max(-8.0, min(ys_all) - 0.5), 1.2)
        # 标注达阈值所需的相对节省
        base = s["uniform"]["MB"]
        ours = s["direct"]["MB"]
        if base and ours:
            ax.annotate(f"$-${100*(1-ours/base):.0f}%", xy=(ours, thr - ref),
                        xytext=(ours * 0.30, thr - ref + 0.9), fontsize=7, color="tab:blue")
        ax.set_title(scene, fontsize=8)
        ax.set_xlabel("prefix (MB)")
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7.2)
    axes[0].set_ylabel(r"$\Delta$PSNR vs. 16-bit (dB)")
    h, l = [], []
    for name, (color, marker, label) in ORDER_STYLE.items():
        h.append(matplotlib.lines.Line2D([], [], color=color, marker=marker, markersize=3))
        l.append(label)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.0, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handlelength=1.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = os.path.join(FIG, "fig1_prefix.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_rd():
    scenes = list(MANIFEST)
    fig, axes = plt.subplots(1, len(scenes), figsize=(7.0, 1.78))
    for ax, scene in zip(axes, scenes):
        m = MANIFEST[scene]
        for pol, color, marker, label in (("uniform", "tab:gray", "s", "uniform bit-plane"),
                                          ("alloc", "tab:blue", "o", "per-budget allocation"),
                                          ("nested", "tab:green", "^", "single adaptive stream (ours)")):
            if pol == "nested":
                np_ = os.path.join(W, f"work/final/nested_{SCENE_FILES[scene]}.json")
                if not os.path.exists(np_):
                    continue
                rows = json.load(open(np_))["rows"]
                ax.plot([r["MB"] for r in rows], [r["psnr_all"] for r in rows],
                        color=color, linewidth=0.9, label=label)
                continue
            pts = [load(p) for p in m[pol]]
            xs = sorted([(p["MB"], p["full"]) for p in pts])
            ax.plot([x for x, _ in xs], [y for _, y in xs], color=color, marker=marker,
                    markersize=3, label=label)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 5, 10]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(scene, fontsize=8)
        ax.set_xlabel("model size (MB)")
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7.2)
    axes[0].set_ylabel("masked PSNR (dB)")
    h_, l_ = [], []
    for color, marker, label in (("tab:gray", "s", "uniform bit-plane"),
                                 ("tab:blue", "o", "per-budget allocation"),
                                 ("tab:green", "^", "single adaptive stream (ours)")):
        h_.append(matplotlib.lines.Line2D([], [], color=color, marker=marker, markersize=3))
        l_.append(label)
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=5.6, frameon=False,
               bbox_to_anchor=(0.5, -0.07), handlelength=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(FIG, "fig3_rd.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


SCENE_FILES = {"backpack": "backpack", "sriracha-tree": "sriracha", "Truck": "truck",
               "Umbrella": "umbrella", "Jumping": "jumping"}


def fig_composition():
    """图 2：码率构成（逐场景，堆叠条形）。"""
    scenes = list(MANIFEST)
    comps = ["bg.means", "bg.quats", "bg.colors", "bg.scales", "bg.opacities",
             "fg.means", "fg.motion_coefs", "fg.others"]
    colors = plt.cm.tab20(np.linspace(0, 1, len(comps)))
    data = {}
    for scene in scenes:
        r = json.load(open(os.path.join(W, f"work/rates_{SCENE_FILES[scene]}.json")))
        c = {k.replace(".params.", "."): v["16"] for k, v in r["components"].items()}
        fg_other = sum(v for k, v in c.items() if k.startswith("fg.") and k not in ("fg.means", "fg.motion_coefs"))
        vals = [c.get("bg.means", 0), c.get("bg.quats", 0), c.get("bg.colors", 0),
                c.get("bg.scales", 0), c.get("bg.opacities", 0),
                c.get("fg.means", 0), c.get("fg.motion_coefs", 0), fg_other]
        data[scene] = np.array(vals) / 1e6
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bottom = np.zeros(len(scenes))
    for i, (name, col) in enumerate(zip(comps, colors)):
        vals = np.array([data[s][i] for s in scenes])
        ax.bar(scenes, vals, bottom=bottom, label=name, color=col, width=0.6)
        bottom += vals
    ax.set_ylabel("bytes at 16-bit (MB)")
    ax.set_xticklabels([t.get_text().replace("sriracha-tree", "sriracha\n-tree") for t in ax.get_xticklabels()], fontsize=6.5)
    ax.legend(fontsize=5.6, ncol=2, loc="upper right")
    ax.grid(alpha=0.2, axis="y", linewidth=0.4)
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_composition.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_staleness():
    """图 3：候选包排序的陈旧度（相邻两次刷新的 Kendall tau）。"""
    scenes = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for c, s in zip(plt.cm.tab10(np.linspace(0, 0.45, 5)), scenes):
        d = json.load(open(os.path.join(W, f"work/final/prefixlp_{s}.json")))
        tr = [x for x in d["staleness"]["direct"] if x["tau"] is not None]
        ax.plot([x["step"] for x in tr], [x["tau"] for x in tr], marker="o",
                markersize=2.2, color=c, label=s)
    ax.set_xlabel("packet index")
    ax.set_ylabel(r"Kendall $\tau$")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="lower right", handletextpad=0.2)
    fig.tight_layout()
    out = os.path.join(FIG, "fig6_staleness.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    fig_prefix()
    fig_rd()
    fig_composition()
    fig_staleness()
