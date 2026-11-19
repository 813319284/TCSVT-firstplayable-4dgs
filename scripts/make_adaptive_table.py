#!/usr/bin/env python3
"""make_adaptive_table.py — 逐包重估强基线的对照表（强自适应基线对照）。

三种策略只换候选排序目标，其余（checkpoint / 包集合 / 标定与评测帧 / 字节账本 / 双区域接受判据 /
走满路径）完全相同：
    deficit     论文式(3)：质量缺口减少量的正部 / 新增字节        （Ours）
    full_gain   全图边际 PSNR 增益 / 新增字节                      （标准逐包贪心）
    joint_gain  双区域边际 PSNR 增益之和 / 新增字节                （不做缺口截断）

输出：reports/table_adaptive_compare.csv（五行全保留；负收益原样记负数）
用法： python3 scripts/make_adaptive_table.py [--repo R]
"""
import argparse
import csv
import hashlib
import json
import os
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
POLICIES = ["deficit", "full_gain", "joint_gain"]
FIXED = ["uniform", "roundrobin", "posfirst", "dynfirst", "oneshot", "mpprefix"]


def root() -> str:
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def find(repo, rel):
    for c in (f"{repo}/{rel}", f"{repo}/results/raw/{rel.split('work/final/')[-1]}"):
        if os.path.exists(c):
            return c
    return f"{repo}/{rel}"


def first_pass(path):
    d = json.load(open(path))
    v = d["summary"]["direct"]["bytes_to_ref_minus_0.5"]
    return None if v is None else v, d


def ids_hash(seq):
    return hashlib.sha256(",".join(str(x) for x in seq).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=root())
    a = ap.parse_args()
    repo = a.repo
    rows, missing = [], []
    for sc in SCENES:
        base_p = find(repo, f"work/final/p0_{sc}.json")
        if not os.path.exists(base_p):
            missing.append(base_p)
            continue
        base = json.load(open(base_p))
        ours, _ = first_pass(base_p)
        best_fixed = min(base["summary"][k]["bytes_to_ref_minus_0.5"] for k in FIXED)
        vals = {}
        for pol in POLICIES:
            p = find(repo, f"work/final/adaptive_{pol}_{sc}.json")
            if not os.path.exists(p):
                missing.append(p)
                vals[pol] = None
                continue
            v, d = first_pass(p)
            vals[pol] = v
            if d.get("adaptive_score") != pol:
                missing.append(f"{p}: adaptive_score={d.get('adaptive_score')} != {pol}")

        def pct(opp):
            if opp is None or ours is None:
                return ""
            return f"{100 * (opp - ours) / opp:.1f}"

        fg, jg = vals["full_gain"], vals["joint_gain"]
        best_adapt = min([v for v in (fg, jg) if v is not None], default=None)
        rows.append({
            "scene": sc,
            "ours_bytes": ours if ours is not None else "not reached",
            "fullgain_bytes": fg if fg is not None else "not reached",
            "jointgain_bytes": jg if jg is not None else "not reached",
            "best_fixed_bytes": best_fixed,
            "ours_vs_fullgain_pct": pct(fg),
            "ours_vs_jointgain_pct": pct(jg),
            "ours_vs_best_adaptive_pct": pct(best_adapt),
            "calib_ids_sha256": ids_hash(base["calib_ids"]),
            "eval_ids_sha256": ids_hash(base["eval_ids"]),
        })
    out = f"{repo}/reports/table_adaptive_compare.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"✓ {out}（{len(rows)} 行）")
        for r in rows:
            print(f"  {r['scene']:9s} ours={r['ours_bytes'] if isinstance(r['ours_bytes'], str) else round(r['ours_bytes']/1e6,2)}MB "
                  f"fullgain={r['fullgain_bytes'] if isinstance(r['fullgain_bytes'], str) else round(r['fullgain_bytes']/1e6,2)} "
                  f"jointgain={r['jointgain_bytes'] if isinstance(r['jointgain_bytes'], str) else round(r['jointgain_bytes']/1e6,2)} "
                  f"vs_best_adaptive={r['ours_vs_best_adaptive_pct']}%")
    if missing:
        print("✗ 缺输入或元数据不符：")
        for m in missing:
            print("   -", m)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
