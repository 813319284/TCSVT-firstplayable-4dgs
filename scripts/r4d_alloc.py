#!/usr/bin/env python3
"""r4d_alloc.py — 灵敏度引导的码率分配（本稿的方法核心）。

输入
  · `--rates`  ：各分量在 {4,6,8,12,16} bit 下的实测字节（CPU-only，秒级）
  · `--sens`   ：各分量单独降到 4 bit 时实测的 ΔPSNR（每分量一次渲染，约 2 min，可并行）
输出
  · 给定总预算下的逐分量比特向量（贪心：每字节边际失真最小化）
  · 可直接喂给 `r4d_rd.py --bits ...` 的字符串

模型（写进论文的近似，必须照实描述）
  D_i(b) ≈ Δ_i4 * (16 - b) / 12        # 每分量失真随比特线性（以 4/16 bit 两个实测点定标）
  bytes_i(b) = 实测码率曲线插值
  目标：min Σ D_i(b_i)  s.t.  Σ bytes_i(b_i) ≤ B_target
  贪心：从 b=16 逐级降到 4，每步选「ΔD_i / Δbytes_i」最大者，直到预算用完。

用法
  python3 scripts/r4d_alloc.py --rates work/rates_backpack.json \
      --sens work/sens_backpack.json --target-mb 10.0
"""
from __future__ import annotations

import argparse
import json

LEVELS = [16, 14, 12, 10, 8, 7, 6, 5, 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True, help="json: {component: {bit: bytes}}")
    ap.add_argument("--sens", required=True, help="json: {component: delta_at_4bit}")
    ap.add_argument("--target-mb", type=float, default=0.0, help="0 = 只打印前沿，不选点")
    ap.add_argument("--fixed-mb", type=float, default=0.0, help="固定开销（相机等），计入总量")
    ap.add_argument("--shape", default="", help="json {bits: shape} 归一化失真形状（默认取自 bg.means 实测）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    raw = json.load(open(a.rates))
    # 兼容两种 schema：{comp: {bits: bytes}} 或 r4d_rates.py 的 {"components": {...}, "fixed_bytes": ...}
    if "components" in raw:
        rates = {k.replace(".params.", "."): v for k, v in raw["components"].items()}
        if not a.fixed_mb and raw.get("fixed_bytes"):
            a.fixed_mb = raw["fixed_bytes"] / 1e6
    else:
        rates = {k.replace(".params.", "."): v for k, v in raw.items()}
    sens = {k.replace(".params.", "."): v for k, v in json.load(open(a.sens)).items()}
    comps = [c for c in rates if c in sens]
    print(f"components={len(comps)}  fixed={a.fixed_mb:.3f} MB")

    # 归一化失真形状：D(b)/D(4)，默认用 bg.means 的实测两点 + 四点（有就更好）
    shape = {4: 1.0, 6: 0.43, 8: 0.18, 12: 0.01, 16: 0.0}
    if a.shape:
        shape = {int(k): float(v) for k, v in json.load(open(a.shape)).items()}
    def D(comp, b):
        return sens[comp] * shape.get(b, 0.0)

    # 逐分量码率曲线（MB）
    rb = {c: {b: rates[c][str(b)] / 1e6 for b in LEVELS if str(b) in rates[c]} for c in comps}
    for c in rb:
        if 16 not in rb[c]:  # 缺 16 bit 时用 12 bit 外推
            rb[c][16] = rb[c].get(12, max(rb[c].values())) * 1.1

    bits = {c: 16 for c in comps}
    total = sum(rb[c][16] for c in comps) + a.fixed_mb
    print(f"start: all 16-bit -> {total:.3f} MB")

    # 逐步降级：每步选 (ΔD / Δbytes) 最大
    trace = []
    while True:
        best = None
        for c in comps:
            b = bits[c]
            nxt = [x for x in LEVELS if x < b]
            if not nxt:
                continue
            nb = max(nxt)
            save = rb[c][b] - rb[c][nb]
            if save <= 0:
                continue
            lost = max(0.0, D(c, nb) - D(c, b))    # 用形状模型算边际失真（降位宽 → 失真增大）
            score = lost / save                    # dB per MB（越小越划算）
            if best is None or score < best[0]:
                best = (score, c, nb, save, lost)
        if best is None:
            break
        score, c, nb, save, lost = best
        bits[c] = nb
        total -= save
        trace.append({"component": c, "to_bits": nb, "total_MB": round(total, 4),
                      "cum_delta_dB": None, "step_save_MB": round(save, 4),
                      "step_lost_dB": round(lost, 4)})
        if a.target_mb and total <= a.target_mb:
            break

    cum = 0.0
    for t in trace:
        cum += t["step_lost_dB"]
        t["cum_delta_dB"] = round(cum, 3)

    print(f"\n{'component':22s} {'bits':>4s}  {'MB':>8s}   {'cum ΔdB':>9s}")
    for c in comps:
        print(f"{c:22s} {bits[c]:>4d}  {rb[c][bits[c]]:8.3f}")
    print(f"{'TOTAL':22s} {'':>4s}  {total:8.3f}   pred Δ={cum:+.3f} dB")

    spec = ",".join(
        f"{c}={bits[c]}" for c in comps
    )
    print(f"\n--bits '{spec}'")
    out = {"comps": comps, "bits": bits, "total_MB": round(total, 4),
           "pred_delta_dB": round(cum, 3), "bits_spec": spec, "trace": trace}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
