#!/usr/bin/env python3
"""collect_rd.py — 汇总所有率失真工作点，按 **bits 向量** 自动分类（不靠文件名）。

判据：
  · 所有分量同一比特（fg==bg==coefs==bases 且每分量同值）→ `uniform@<b>`
  · 否则 → `alloc`（本方法）
  · 只有单一分量被改动（其余 16）→ `single:<comp>@<b>`（灵敏度表，不进 RD 曲线）

输出：reports/rd_table.json + 控制台表 + 每场景 BD-PSNR（PCHIP 插值，共同区间积分）。

用法： python3 scripts/collect_rd.py --out reports/rd_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SCENE_KEYS = {"backpack": ["bp", "rd_r", "rd_u"], "sriracha": ["sr"], "truck": ["tr"], "umbrella": ["umb"]}


def classify(bits: dict) -> tuple[str, str]:
    vals = [v for v in bits.values() if isinstance(v, int)]
    if not vals:
        return "unknown", ""
    if len(set(vals)) == 1:
        return "uniform", f"b{vals[0]}"
    lows = [k for k, v in bits.items() if v < 16]
    if len(lows) == 1:
        return "single", f"{lows[0]}@{bits[lows[0]]}"
    return "alloc", ""


def guess_scene(path: str, d: dict) -> str | None:
    dd = (d.get("scene") or "") + " " + (d.get("ckpt") or "")
    low = dd.lower()
    for s, keys in SCENE_KEYS.items():
        if s in low:
            return s
    f = os.path.basename(path)
    if "umm" in f or "umbrella" in f:
        return "umbrella"
    if "truck" in f or f.startswith("rd_utr") or f.startswith("rd_atr") or f.startswith("tr_"):
        return "truck"
    if "sr" in f and "bp" not in f:
        return "sriracha"
    if "bp" in f or f.startswith("rd_r") or f.startswith("rd_u"):
        return "backpack"
    return None


def pchip_interp(x, y, xs):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(x, y)(xs)


def bd_psnr(anchor, test):
    a = sorted(anchor)
    t = sorted(test)
    xa = np.log10([p[0] for p in a]); ya = np.array([p[1] for p in a])
    xt = np.log10([p[0] for p in t]); yt = np.array([p[1] for p in t])
    lo, hi = max(xa.min(), xt.min()), min(xa.max(), xt.max())
    if hi <= lo + 1e-6:
        return None
    xs = np.linspace(lo, hi, 300)
    return float(np.mean(pchip_interp(xt, yt, xs) - pchip_interp(xa, ya, xs)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", default="work/rd_*.json,work/rd2/*.json,work/sr_*.json,work/lin_*.json,work/sens/*.json,work/sens_sr/*.json,work/sens_truck/*.json,work/sens_umb/*.json,work/curve/*.json")
    ap.add_argument("--out", default="reports/rd_table.json")
    a = ap.parse_args()

    rows = []
    for g in a.globs.split(","):
        for p in glob.glob(g):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if "total_MB" not in d or d.get("full_psnr") is None:
                continue
            scene = guess_scene(p, d)
            if scene is None:
                continue
            kind, tag = classify(d.get("bits") or {})
            rows.append({"file": p, "scene": scene, "kind": kind, "tag": tag,
                         "MB": d["total_MB"], "full": d["full_psnr"], "fg": d.get("fg_psnr"),
                         "n_eval": d.get("n_eval"), "bits": d.get("bits")})

    table = {}
    for r in rows:
        table.setdefault(r["scene"], {}).setdefault(r["kind"], []).append(
            {"MB": r["MB"], "full": r["full"], "fg": r["fg"], "tag": r["tag"], "file": r["file"]})

    summary = {}
    for scene, kinds in table.items():
        print(f"\n### {scene}")
        for kind in ["uniform", "alloc"]:
            pts = sorted({(round(p["MB"], 3), round(p["full"], 3)) for p in kinds.get(kind, [])})
            print(f"  {kind:8s} n={len(pts)}  {pts}")
        uni = [(p["MB"], p["full"]) for p in kinds.get("uniform", [])]
        alo = [(p["MB"], p["full"]) for p in kinds.get("alloc", [])]
        bd = bd_psnr(uni, alo) if len(uni) >= 2 and len(alo) >= 2 else None
        if bd is not None:
            print(f"  BD-PSNR(alloc vs uniform) = {bd:+.3f} dB")
        summary[scene] = {"bd_psnr": None if bd is None else round(bd, 3),
                          "n_uniform": len(uni), "n_alloc": len(alo)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
