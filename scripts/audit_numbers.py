#!/usr/bin/env python3
"""audit_numbers.py — 独立复核（不复用 make_tex_tables/check_numbers 的逻辑）。

从 work/final/*.json 与 work/sens_p0_*.json 重新推导，并与稿件表体逐项比对：
  1. 每个顺序的首次达标字节（从逐前缀记录重算，双区域判据）
  2. Table III 的节省率（对比最强固定顺序与 uniform）
  3. Table II 的灵敏度矩阵（与 paper/tables/sens.tex 的字符串比对）
  4. Table IV 的单流 − 按预算分配差（与 paper/tables/scalable.tex 比对）
  5. 码流自证：codec_bit_exact 全 True、stream_check 两数相等、calib/eval 帧不交
  6. 摘要里的区间（16.6/47.1/45.9/58.1/0.55）与实际数据一致

用法： python3 scripts/audit_numbers.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

W = os.environ.get("TCSVT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
FIXED = ["uniform", "dynfirst", "posfirst", "roundrobin", "oneshot", "mpprefix"]


def tex_table(name):
    p = f"{W}/paper/tables/{name}.tex"
    return open(p).read() if os.path.exists(p) else ""



def _p(rel):
    """仓库布局（work/final/...）或补充包布局（results/raw/...）之间自动选路。"""
    import os as _os
    cands = [f"{W}/{rel}"]
    if rel.startswith("work/final/"):
        cands.append(f"{W}/results/raw/{rel[len('work/final/'):]}")
    if rel.startswith("work/"):
        cands.append(f"{W}/results/raw/{rel[len('work/'):]}")
    cands.append(f"{W}/results/{_os.path.basename(rel)}")
    for c in cands:
        if _os.path.exists(c):
            return c
    return cands[0]


def J(rel):
    import json as _json
    return _json.load(open(_p(rel)))

def main() -> int:
    tex = open(f"{W}/paper/main.tex").read()
    bad, checks = [], 0

    # ---------- 1. 达标前缀独立重算 ----------
    prefixes = {}
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        ref = d["ref_psnr"]
        ref_fg = d["summary"]["uniform"]["ref_fg"]
        rows = [r for r in d["results"] if r["order"] == "direct"]
        for od in ["direct"] + FIXED:
            rs = sorted([r for r in d["results"] if r["order"] == od], key=lambda r: r["bytes"])
            if not rs:
                continue
            hit = next((r for r in rs if r["psnr_first"] >= ref - 0.5
                        and r["psnr_first_fg"] >= ref_fg - 0.5), None)
            got = hit["MB"] if hit else None
            pub = d["summary"].get(od, {}).get("MB")
            checks += 1
            if (got is None) != (pub is None) or (got is not None and abs(got - pub) > 0.011):
                bad.append(f"{sc}/{od}: 重算 {got} vs summary {pub}")
            if od == "direct":
                prefixes[sc] = got

    # ---------- 2. Table III 复核 ----------
    fp = tex_table("firstplayable")
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        valid = {k: v["MB"] for k, v in d["summary"].items() if v.get("MB")}
        if "direct" not in valid:
            continue
        best = min((valid[k], k) for k in FIXED if k in valid)
        save = (1 - valid["direct"] / best[0]) * 100
        checks += 1
        if f"{save:.1f}" not in fp:
            bad.append(f"Table III {sc}: 节省 {save:.1f}% 不在表里")


    # ---------- 3. Table II 灵敏度 ----------
    sens_tex = tex_table("sens")
    keys = ["bg.means", "bg.quats", "bg.scales", "bg.colors", "bg.opacities",
            "fg.means", "fg.quats", "fg.scales", "fg.colors", "fg.opacities", "fg.motion_coefs"]
    for sc in SCENES:
        g = J(f"work/sens_p0_{sc}.json")["groups"]
        for k in keys:
            v = g[k]["loss_full"]
            checks += 1
            if f"{v:.2f}" not in sens_tex:
                bad.append(f"Table II {sc}/{k}: {v:.2f} 不在表里")

    # ---------- 4. Table IV 复核 ----------
    sc_tex = tex_table("scalable")
    for sc in SCENES:
        p = _p(f"work/final/p0_alloc_{sc}.json")
        if not os.path.exists(p):
            continue
        al = json.load(open(p))
        st = J(f"work/final/p0_{sc}.json")
        direct = [r for r in st["results"] if r["order"] == "direct"]
        seen = set()
        for b in al["budgets"]:
            key = (sc, round(b["bytes"] / 1e6, 2))
            if key in seen:
                continue
            seen.add(key)
            cand = [r for r in direct if r["bytes"] <= b["bytes"]]
            if not cand:
                continue
            s = cand[-1]
            df = s["psnr_first"] - b["eval_full"]
            checks += 1
            if f"${df:+.2f}$" not in sc_tex:
                bad.append(f"Table IV {sc}@{b['budget_MB']}: Δfull {df:+.2f} 不在表里")

    # ---------- 5. 码流自证 ----------
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        checks += 1
        if not all(d.get("codec_bit_exact", {}).values()):
            bad.append(f"{sc}: codec_bit_exact 有 False")
        checks += 1
        if not set(d["calib_ids"]).isdisjoint(set(d["eval_ids"])):
            bad.append(f"{sc}: 校准帧与评测帧重叠")
        for name, chk in d.get("stream_check", {}).items():
            checks += 1
            if chk["last_stage_bytes"] != chk["serialized_stream_bytes"]:
                bad.append(f"{sc}/{name}: 末级字节 ≠ 码流字节")

    # ---------- 6. 摘要区间 ----------
    saves, uni_saves = [], []
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        valid = {k: v["MB"] for k, v in d["summary"].items() if v.get("MB")}
        best = min((valid[k], k) for k in FIXED if k in valid)
        saves.append((1 - valid["direct"] / best[0]) * 100)
        uni_saves.append((1 - valid["direct"] / valid["uniform"]) * 100)
    for lo, hi, label in [(min(saves), max(saves), "best-fixed"),
                          (min(uni_saves), max(uni_saves), "uniform")]:
        checks += 1
        if f"{lo:.1f}" not in tex or f"{hi:.1f}" not in tex:
            bad.append(f"摘要区间 {label}: {lo:.1f}–{hi:.1f}% 与正文不符")

    diffs = []
    for sc in SCENES:
        p = _p(f"work/final/p0_alloc_{sc}.json")
        if not os.path.exists(p):
            continue
        al = json.load(open(p)); st = J(f"work/final/p0_{sc}.json")
        direct = [r for r in st["results"] if r["order"] == "direct"]
        for b in al["budgets"]:
            cand = [r for r in direct if r["bytes"] <= b["bytes"]]
            if cand:
                diffs += [cand[-1]["psnr_first"] - b["eval_full"],
                          cand[-1]["psnr_first_fg"] - b["eval_fg"]]
    import math
    if not diffs:
        bad.append("没有可用的 allocation JSON（包内缺 p0_alloc_*.json）")
        diffs = [0.0]
    bound = math.ceil(max(abs(x) for x in diffs) * 100) / 100
    checks += 1
    if f"{bound:.2f}" not in tex:
        bad.append(f"最大匹配预算差上界 {bound:.2f} dB 未在正文")

    # ---------- 6b. 第二种子（五场景）----------
    for sc in SCENES:
        p = _p(f"work/final/p0_s43_{sc}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        valid = {k: v["MB"] for k, v in d["summary"].items() if v.get("MB")}
        if "direct" not in valid:
            continue
        checks += 1
        if f"{valid['direct']:.2f}" not in tex:
            bad.append(f"seed43 {sc}: {valid['direct']:.2f} MB 不在正文")

    # ---------- 7. 时间块切分 ----------
    blk_sav = []
    for sc in SCENES:
        p = _p(f"work/final/block_{sc}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        # 与正文同口径：首次达标字节，不是最后一阶段的流大小
        valid = {k: v["bytes_to_ref_minus_0.5"] for k, v in d["summary"].items()
                 if v.get("bytes_to_ref_minus_0.5")}
        if "direct" not in valid:
            continue
        best = min((valid[k], k) for k in FIXED if k in valid)
        blk_sav.append((1 - valid["direct"] / best[0]) * 100)
    if blk_sav:
        for v in blk_sav:
            checks += 1
            if f"{v:.1f}" not in tex:
                bad.append(f"时间块切分节省 {v:.1f}% 不在正文")

    # ---------- 8. 逐帧判据（每帧对自身 16-bit 值）----------
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        rows = [r for r in d["results"] if r["order"] == "direct"]
        if not rows or not rows[-1].get("per_frame"):
            continue
        ref16 = {x["frame_id"]: x for x in rows[-1]["per_frame"]}
        hit = None
        for r in rows:
            pf = r.get("per_frame") or []
            if pf and all(x["full"] >= ref16[x["frame_id"]]["full"] - 0.5
                          and x["fg"] >= ref16[x["frame_id"]]["fg"] - 0.5
                          for x in pf if x["frame_id"] in ref16):
                hit = r
                break
        if hit:
            checks += 1
            if f"{hit['MB']:.2f}" not in tex:
                bad.append(f"逐帧判据 {sc}: {hit['MB']:.2f} MB 不在正文")

    # ---------- 8b. 逐帧判据的额外字节百分比 ----------
    for sc in SCENES:
        d = J(f"work/final/p0_{sc}.json")
        rows = [r for r in d["results"] if r["order"] == "direct"]
        if not rows or not rows[-1].get("per_frame"):
            continue
        ref16 = {x["frame_id"]: x for x in rows[-1]["per_frame"]}
        mean_mb = d["summary"]["direct"]["MB"]
        hit = None
        for r in rows:
            pf = r.get("per_frame") or []
            if pf and all(x["full"] >= ref16[x["frame_id"]]["full"] - 0.5
                          and x["fg"] >= ref16[x["frame_id"]]["fg"] - 0.5
                          for x in pf if x["frame_id"] in ref16):
                hit = r
                break
        if hit:
            pct = (hit["MB"] / mean_mb - 1) * 100
            checks += 1
            if f"{pct:.1f}" not in tex:
                bad.append(f"逐帧判据额外字节 {sc}: {pct:.1f}% 不在正文")

    # ---------- 9. 运动一致性 ----------
    for sc in SCENES:
        p = _p(f"work/motion_{sc}.json")
        if not os.path.exists(p):
            continue
        mo = json.load(open(p))
        checks += 1
        if f"{mo['pearson_prefix']:.2f}" not in tex:
            bad.append(f"motion {sc}: {mo['pearson_prefix']:.2f} 不在正文")

    print(f"独立复核：{checks} 项，异常 {len(bad)}")
    for b in bad[:20]:
        print("  ✗", b)
    print("审计结果：" + ("全部一致" if not bad else "有不一致"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
