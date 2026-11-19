#!/usr/bin/env python3
"""stats_matrix.py — 稳健性矩阵（阈值/相机/划分三轴），全部从已记录的 per_frame 复算。

三条轴：
  · 质量阈值 0.25 / 0.50 / 1.00 dB（各自 16-bit 逐帧参考）
  · 相机：按 camera 分组，报每相机的前缀字节（不能只看合并均值）
  · 时间块划分：parity vs block（block_*.json 已有）

产出：
  reports/table_robustness.csv   场景 × 策略 × 阈值 → B_mean / B_90%
  reports/table_camera.csv       场景 × 策略 → 每个相机的 B_mean
用法：python3 scripts/stats_matrix.py [--repo R]
"""
import argparse
import csv
import json
import os
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
POL = {"ours": "p0", "full_gain": "adaptive_full_gain",
       "joint_gain": "adaptive_joint_gain", "tail_cvar": "risk_tail_cvar"}
TOLS = [0.25, 0.50, 1.00]


def root():
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def load(path):
    return json.load(open(path))


def refs(path):
    d = load(path)
    return {f"{x['camera']}:{x['frame_id']}": (x["full"], x["fg"]) for x in d["per_frame"]}


def prefix_bytes(run, ref, tol, cam=None, frac=1.0):
    """首次达标的字节。frac=None → 用**均值**判据（与主表同口径）；
    frac=1.0 → 每帧都达标；frac=0.9 → ≥90% 帧达标。"""
    stages = [r for r in run["results"] if r["order"] == "direct"]
    for st in stages:
        pf = [x for x in (st.get("per_frame") or [])
              if (cam is None or x["camera"] == cam) and f"{x['camera']}:{x['frame_id']}" in ref]
        if not pf:
            continue
        if frac is None:
            m_full = sum(x["full"] for x in pf) / len(pf)
            m_fg = sum(x["fg"] for x in pf) / len(pf)
            r_full = sum(ref[f"{x['camera']}:{x['frame_id']}"][0] for x in pf) / len(pf)
            r_fg = sum(ref[f"{x['camera']}:{x['frame_id']}"][1] for x in pf) / len(pf)
            if m_full >= r_full - tol and m_fg >= r_fg - tol:
                return st["bytes"]
            continue
        ok = 0
        for x in pf:
            r = ref[f"{x['camera']}:{x['frame_id']}"]
            if x["full"] >= r[0] - tol and x["fg"] >= r[1] - tol:
                ok += 1
        if ok / len(pf) >= frac - 1e-12:
            return st["bytes"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=root())
    a = ap.parse_args()
    repo = a.repo

    rob, cam_rows = [], []
    for sc in SCENES:
        rp = f"{repo}/work/final/ref_frames_{sc}.json"
        if not os.path.exists(rp):
            print(f"✗ 缺 {rp}")
            continue
        ref = refs(rp)
        cams = sorted({int(k.split(":")[0]) for k in ref})
        for name, stem in POL.items():
            p = f"{repo}/work/final/{stem}_{sc}.json"
            if not os.path.exists(p):
                continue
            run = load(p)
            for tol in TOLS:
                rob.append({"scene": sc, "policy": name, "tol_dB": tol,
                            "B_mean_MB": _mb(prefix_bytes(run, ref, tol, frac=None)),
                            "B_90pct_MB": _mb(prefix_bytes(run, ref, tol, frac=0.9))})
            for c in cams:
                cam_rows.append({"scene": sc, "policy": name, "camera": c,
                                 "B_mean_MB": _mb(prefix_bytes(run, ref, 0.5, cam=c, frac=None))})

    with open(f"{repo}/reports/table_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rob[0]))
        w.writeheader(); w.writerows(rob)
    with open(f"{repo}/reports/table_camera.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cam_rows[0]))
        w.writeheader(); w.writerows(cam_rows)
    print(f"✓ table_robustness.csv（{len(rob)} 行）· table_camera.csv（{len(cam_rows)} 行）")

    for sc in SCENES:
        line = [r for r in rob if r["scene"] == sc and r["policy"] == "ours"]
        other = [r for r in rob if r["scene"] == sc and r["policy"] == "joint_gain"]
        if line:
            print(f"  {sc:9s} ours B_mean@0.25/0.5/1.0 = "
                  + "/".join(str(r["B_mean_MB"]) for r in line)
                  + " | joint " + "/".join(str(r["B_mean_MB"]) for r in other))
    return 0


def _mb(v):
    return None if v is None else round(v / 1e6, 3)


if __name__ == "__main__":
    raise SystemExit(main())
