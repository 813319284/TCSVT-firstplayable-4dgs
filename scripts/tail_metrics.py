#!/usr/bin/env python3
"""tail_metrics.py — 从已记录的逐阶段 per_frame + 16-bit 逐帧参考，复算"尾部"指标。

不重跑任何调度实验：只用
  · work/final/ref_frames_<scene>.json   （scripts/r4d_prefix.py --ref-frames-only 产出）
  · 任一 run 的 JSON（p0 / adaptive_* / risk_*）里 results[*].per_frame

对每个顺序、每个包边界复算三种门槛的**首次达标字节**：
  B_mean        评测帧平均 full/fg 均达标（与主表同口径，可追溯）
  B_90pct       至少 90% 的评测帧**各自**在 full/fg 两区都达标
  B_all_frames  每一帧都达标
"每帧达标"= 该帧 PSNR ≥ 它自己的 16-bit 参考 − tol。

用法：
  python3 scripts/tail_metrics.py --run work/final/risk_tail_cvar_truck.json --ref work/final/ref_frames_truck.json
  python3 scripts/tail_metrics.py --table            # 汇总所有 run → reports/table_tail.csv
"""
import argparse
import csv
import json
import os
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
POLICY_TAGS = {
    "p0": "best_fixed_run(direct=Ours)",
    "adaptive_deficit": "ours_regression",
    "adaptive_full_gain": "adaptive_full_gain",
    "adaptive_joint_gain": "adaptive_joint_gain",
    "risk_tail_cvar": "risk_tail_cvar",
    "risk_tail_cvar_r3": "risk_tail_cvar_refresh3",
    "risk_joint_gain_r3": "adaptive_joint_gain_refresh3",
}


def root():
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def key(x):
    return f"{x['camera']}:{x['frame_id']}"


def load_ref(path):
    d = json.load(open(path))
    return {key(x): (x["full"], x["fg"]) for x in d["per_frame"]}


def first_passes(run, ref, tol):
    """返回 {order: {B_mean, B_90pct, B_all_frames}}（字节，未达标为 None）。"""
    out = {}
    orders = sorted({r["order"] for r in run["results"]})
    for o in orders:
        b_mean = b90 = ball = None
        for st in [r for r in run["results"] if r["order"] == o]:
            pf = st.get("per_frame") or []
            if not pf:
                continue
            pairs = [(ref[key(x)], (x["full"], x["fg"])) for x in pf if key(x) in ref]
            if len(pairs) != len(pf):
                continue
            if b_mean is None:
                m_full = sum(v[0] for _, v in pairs) / len(pairs)
                m_fg = sum(v[1] for _, v in pairs) / len(pairs)
                r_full = sum(r[0] for r, _ in pairs) / len(pairs)
                r_fg = sum(r[1] for r, _ in pairs) / len(pairs)
                if m_full >= r_full - tol and m_fg >= r_fg - tol:
                    b_mean = st["bytes"]
            ok = [1 if (v[0] >= r[0] - tol and v[1] >= r[1] - tol) else 0 for r, v in pairs]
            frac = sum(ok) / len(ok)
            if b90 is None and frac >= 0.9 - 1e-12:
                b90 = st["bytes"]
            if ball is None and frac >= 1.0 - 1e-12:
                ball = st["bytes"]
        out[o] = {"B_mean": b_mean, "B_90pct": b90, "B_all_frames": ball}
    return out


def mb(v):
    return None if v is None else round(v / 1e6, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=root())
    ap.add_argument("--run")
    ap.add_argument("--ref")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", default="reports/table_tail.csv")
    a = ap.parse_args()
    repo = a.repo

    if a.run:
        ref = load_ref(a.ref)
        fp = first_passes(json.load(open(a.run)), ref, a.tol)
        print(f"{os.path.basename(a.run)}  tol={a.tol}")
        for o, v in fp.items():
            print(f"  {o:10s} B_mean={mb(v['B_mean'])}  B_90%={mb(v['B_90pct'])}  B_all={mb(v['B_all_frames'])}")
        return 0

    if not a.table:
        ap.error("要么给 --run/--ref，要么给 --table")

    rows = []
    for sc in SCENES:
        refp = f"{repo}/work/final/ref_frames_{sc}.json"
        if not os.path.exists(refp):
            print(f"✗ 缺 {refp}（先跑 --ref-frames-only）")
            continue
        ref = load_ref(refp)
        for tag in POLICY_TAGS:
            p = f"{repo}/work/final/{tag}_{sc}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            order = "direct"
            fp = first_passes(d, ref, a.tol).get(order)
            if not fp:
                continue
            rows.append({
                "scene": sc, "run": POLICY_TAGS[tag],
                "B_mean_MB": mb(fp["B_mean"]),
                "B_90pct_MB": mb(fp["B_90pct"]),
                "B_all_frames_MB": mb(fp["B_all_frames"]),
                "renders": d.get("n_renders"),
                "t_total_s": d.get("t_total_s"),
            })
    out = f"{repo}/{a.out}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "run", "B_mean_MB", "B_90pct_MB",
                                          "B_all_frames_MB", "renders", "t_total_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ {out}（{len(rows)} 行）")
    for r in rows:
        print(f"  {r['scene']:9s} {r['run']:26s} mean={r['B_mean_MB']}  p90={r['B_90pct_MB']}  all={r['B_all_frames_MB']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
