#!/usr/bin/env python3
"""measure_decode.py — 诚实测一次「编码 / 解码」的墙钟时间。

为什么需要：`r4d_prefix.py` 的包序列是"预量化缓存"，其 `t_decode` 只反映最后一个
分量的反量化，不能代表真实解码。这里对给定比特向量做**真的**熵编码 + 熵解码，
给出每个阶段的时间与字节数，供正文写"启动时间以传输为主、解码可忽略"。

用法：
  python3 scripts/measure_decode.py --ckpt <ckpt> --bits '...' --reps 5
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import torch
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))


def q_encode(x: torch.Tensor, bits: int):
    x = x.detach().float()
    fin = torch.isfinite(x)
    amax = float((x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0)).clamp_min(1e-6))
    xc = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax), -amax, amax)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(xc / scale), -qmax - 1, qmax)
    if bits > 16:
        raw = q.to(torch.int32).numpy().tobytes()
    elif bits > 8:
        raw = q.to(torch.int16).numpy().tobytes()
    else:
        raw = q.to(torch.int8).clamp(-127, 127).numpy().tobytes()
    return raw, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bits", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    bits = {}
    for kv in a.bits.split(","):
        k, v = kv.split("=")
        bits[k.strip()] = int(v)

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    tensors = {}
    for grp in ["fg", "bg"]:
        for name, t in sd.items():
            if t is None or not hasattr(t, "shape"):
                continue
            if name.startswith(f"{grp}.params."):
                key = f"{grp}.{name.split('.')[-1]}"
                tensors[key] = t
    for name in ["rots", "transls"]:
        k = f"motion_bases.params.{name}"
        if k in sd:
            tensors[f"bases.{name}"] = sd[k]

    def bits_for(k):
        if k in bits:
            return bits[k]
        grp = k.split(".")[0]
        if k.startswith("fg.motion_coefs"):
            return bits.get("coefs", bits.get("fg", 16))
        if k.startswith("bases."):
            return bits.get("bases", 16)
        return bits.get(grp, 16)

    # 编码
    t0 = time.time()
    streams, meta = {}, {}
    nbytes = 0
    for k, t in tensors.items():
        raw, scale = q_encode(t, bits_for(k))
        blob = zlib.compress(raw, 9)
        streams[k] = blob
        meta[k] = (t.shape, t.dtype, scale, bits_for(k))
        nbytes += len(blob)
    t_enc = time.time() - t0

    # 解码（重复 reps 次取均值）
    t0 = time.time()
    for _ in range(a.reps):
        for k, blob in streams.items():
            shape, dtype, scale, b = meta[k]
            raw = zlib.decompress(blob)
            npdt = np.int32 if b > 16 else (np.int16 if b > 8 else np.int8)
            arr = np.frombuffer(raw, dtype=npdt).astype(np.float32) * scale
            out = torch.from_numpy(arr).reshape(shape).to(dtype)
    t_dec = (time.time() - t0) / a.reps

    print(f"encode(all)   = {t_enc*1e3:8.1f} ms")
    print(f"decode(all)   = {t_dec*1e3:8.1f} ms   ({len(tensors)} tensors, {nbytes/1e6:.3f} MB)")
    print(f"transfer@2Mb/s= {nbytes*8/2e6*1e3:8.1f} ms")
    print(f"decode/transfer = {100*t_dec/(nbytes*8/2e6):.2f}%")
    json.dump({"bytes": nbytes, "t_encode_s": t_enc, "t_decode_s": t_dec,
               "bits": bits}, open(f"{_ROOT}/work/decode_timing.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
