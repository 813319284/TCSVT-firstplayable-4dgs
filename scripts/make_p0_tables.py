#!/usr/bin/env python3
"""make_p0_tables.py — 从重跑后的 p0_*.json 汇总论文用表。

产出（reports/）：
  table_firstplayable.csv  每场景各顺序的首次达标累计传输字节（MB）+ 相对最强固定顺序的节省
  table_lambda.csv         λ ∈ {0, 0.5, 1, 2} 下 direct 的达标字节
  table_crosscam.csv       calib cam2 → eval cam1 与同机位标定的对比
  table_seeds.csv          第二训练种子的 direct / best-fixed 对比

用法： python3 scripts/make_p0_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

W = os.environ.get("TCSVT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
FIXED = ["uniform", "dynfirst", "roundrobin", "posfirst", "sens", "oneshot", "mpprefix"]


def _mb(d, key="direct"):
    v = d["summary"].get(key, {}).get("MB")
    return "" if v is None else round(v, 2)


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    os.makedirs(f"{W}/reports", exist_ok=True)

    # ---- 主表 ----
    rows = []
    for sc in SCENES:
        p = f"{W}/work/final/p0_{sc}.json"
        if not os.path.exists(p):
            print(f"  ⚠ 缺 {p}")
            continue
        d = load(p)
        s = d["summary"]
        valid = {k: v["MB"] for k, v in s.items() if v.get("MB")}
        best_fixed = min((valid[k], k) for k in FIXED if k in valid)
        ours = valid["direct"]
        rows.append({
            "scene": sc,
            "base_MB": round(d["base_bytes"] / 1e6, 2),
            "uniform_MB": round(valid["uniform"], 2),
            "roundrobin_MB": round(valid["roundrobin"], 2),
            "oneshot_MB": round(valid["oneshot"], 2),
            "mpprefix_MB": round(valid["mpprefix"], 2),
            "direct_MB": round(ours, 2),
            "best_fixed_MB": round(best_fixed[0], 2),
            "best_fixed_order": best_fixed[1],
            "save_vs_best_pct": round((1 - ours / best_fixed[0]) * 100, 1),
            "save_vs_uniform_pct": round((1 - ours / valid["uniform"]) * 100, 1),
            "calib_n": len(d.get("calib_ids", [])),
            "eval_n": len(d.get("eval_ids", [])),
            "codec_ok": all(d.get("codec_bit_exact", {}).values()),
        })
    with open(f"{W}/reports/table_firstplayable.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"table_firstplayable.csv（{len(rows)} 行）")
    for r in rows:
        print(f"  {r['scene']:9s} base {r['base_MB']:6.2f} | best fixed {r['best_fixed_MB']:6.2f}"
              f" ({r['best_fixed_order']}) | ours {r['direct_MB']:6.2f} | 省 {r['save_vs_best_pct']:4.1f}%"
              f" | vs uniform 省 {r['save_vs_uniform_pct']:4.1f}%")

    # ---- λ 敏感性 ----
    lam_rows = []
    for sc in SCENES:
        base = f"{W}/work/final/p0_{sc}.json"
        if not os.path.exists(base):
            continue
        d1 = load(base)
        row = {"scene": sc, "lam1.0_MB": _mb(d1)}
        for tag, lam in [("lam0", "0"), ("lam05", "0.5"), ("lam2", "2")]:
            p = f"{W}/work/final/p0_{sc}_{tag}.json"
            row[f"lam{lam}_MB"] = _mb(load(p)) if os.path.exists(p) else ""
        lam_rows.append(row)
    if lam_rows:
        with open(f"{W}/reports/table_lambda.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lam_rows[0]))
            w.writeheader()
            w.writerows(lam_rows)
        print(f"table_lambda.csv（{len(lam_rows)} 行）")

    # ---- 跨机位 ----
    cc_rows = []
    for sc in ["truck", "umbrella", "jumping"]:
        a = f"{W}/work/final/p0_{sc}.json"
        b = f"{W}/work/final/p0_{sc}_crosscam.json"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        da, db = load(a), load(b)
        cc_rows.append({
            "scene": sc,
            "same_cam_MB": _mb(da), "cross_cam_MB": _mb(db),
            "calib_cam": db.get("calib_cam"), "eval_cam": db.get("eval_cam"),
        })
    if cc_rows:
        with open(f"{W}/reports/table_crosscam.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cc_rows[0]))
            w.writeheader()
            w.writerows(cc_rows)
        print(f"table_crosscam.csv（{len(cc_rows)} 行）")

    # ---- 种子 ----
    seed_rows = []
    for sc in SCENES:
        p = f"{W}/work/final/p0_s43_{sc}.json"
        if not os.path.exists(p):
            continue
        d = load(p)
        s = d["summary"]
        valid = {k: v["MB"] for k, v in s.items() if v.get("MB")}
        best = min((valid[k], k) for k in FIXED if k in valid)
        seed_rows.append({"scene": sc, "oneshot_MB": round(valid.get("oneshot", 0), 2),
                          "mpprefix_MB": round(valid.get("mpprefix", 0), 2),
                          "direct_MB": round(valid["direct"], 2),
                          "best_fixed_MB": round(best[0], 2), "best_fixed_order": best[1],
                          "save_vs_best_pct": round((1 - valid["direct"] / best[0]) * 100, 1)})
    if seed_rows:
        with open(f"{W}/reports/table_seeds.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            w.writeheader()
            w.writerows(seed_rows)
        print(f"table_seeds.csv（{len(seed_rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
