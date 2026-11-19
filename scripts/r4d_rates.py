#!/usr/bin/env python3
"""r4d_rates.py — 纯 CPU 的逐分量码率曲线（不需要 GPU、不需要建模型）。

直接读 ckpt 的 state_dict，对每个属性张量在 {16,12,8,6,4} bit 下做对称均匀量化 +
zlib-9 熵编码，给出字节数。这就是 `r4d_alloc.py` 需要的 `--rates`。

用法：
  python3 scripts/r4d_rates.py --ckpt <ckpt> --out work/rates_backpack.json
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch

LEVELS = [16, 14, 12, 10, 8, 7, 6, 5, 4]
FG = ["means", "scales", "colors", "quats", "opacities", "motion_coefs"]
BG = ["means", "scales", "colors", "quats", "opacities"]


def qbytes(t: torch.Tensor, bits: int) -> int:
    x = t.detach().float()
    fin = torch.isfinite(x)
    amax = x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0)
    amax = float(amax.clamp_min(1e-6))
    x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax), -amax, amax)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    if bits > 16:
        return q.numel() * 4
    if bits > 8:
        arr = q.to(torch.int16).numpy()
    else:
        arr = q.to(torch.int8).clamp(-127, 127).numpy()
    return len(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    out, fixed = {}, 0
    for grp, names in (("fg", FG), ("bg", BG)):
        for n in names:
            k = f"{grp}.params.{n}"
            if k not in sd:
                continue
            out[k] = {str(b): qbytes(sd[k], b) for b in LEVELS}
    for n in ["rots", "transls"]:
        k = f"motion_bases.params.{n}"
        if k in sd:
            out[k] = {str(b): qbytes(sd[k], b) for b in LEVELS}
    for n in ["Rs", "ts"]:
        k = f"camera_poses.params.{n}"
        if k in sd:
            fixed += sd[k].numel() * 4

    tot16 = sum(v["16"] for v in out.values()) + fixed
    tot4 = sum(v["4"] for v in out.values()) + fixed
    print(f"{'component':32s} " + " ".join(f"{b:>9d}b" for b in LEVELS))
    for k, v in out.items():
        print(f"{k:32s} " + " ".join(f"{v[str(b)]/1e6:10.3f}" for b in LEVELS))
    print(f"{'TOTAL (MB)':32s} " + " ".join(
        f"{(sum(x[str(b)] for x in out.values())+fixed)/1e6:10.3f}" for b in LEVELS))
    json.dump({"components": out, "fixed_bytes": fixed,
               "total_MB": {str(b): (sum(x[str(b)] for x in out.values()) + fixed) / 1e6
                            for b in LEVELS}},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  (16bit {tot16/1e6:.3f} MB → 4bit {tot4/1e6:.3f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
