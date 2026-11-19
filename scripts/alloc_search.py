#!/usr/bin/env python3
"""alloc_search.py — 独立于投递顺序的"按预算优化分配器"（用于对照单流码率可分级性）。

做法：以模型驱动的分配（r4d_alloc 的解）为起点，在**标定帧**上做实测爬山：
每一步尝试所有"某分量升一档 / 某分量降一档"的单步移动，保留预算内且实测
画质最高的解，直到没有改进。这样得到的是**直接优化画质**的分配，而不是
按阈值缺口排序的结果。

用法：
  python3 scripts/alloc_search.py --ckpt <ckpt> --data-dir <scene> \
      --rates work/rates9_backpack.json --sens work/sens_even_backpack.json \
      --shape work/shape_backpack.json --budget-mb 11.7 --frames 12 \
      --out work/final/search_backpack_11.7.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

TRUST = f"{_DATA}"
sys.path.insert(0, f"{TRUST}/METHOD01_CAPE4D/scripts")
sys.path.insert(0, f"{TRUST}/JOURNAL_DATA/_src/shape-of-motion-main")

LEVELS = [4, 5, 6, 7, 8, 10, 12, 14, 16]
COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--rates", required=True, help="r4d_rates.py 输出（含多档位宽）")
    ap.add_argument("--sens", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--budget-mb", type=float, required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--dataset", choices=["iphone", "nvidia"], default="iphone")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start-bits", default="", help="起始比特向量（如 'fg.means=6,bg.means=16,...'）；留空用模型驱动分配")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image
    import m1_eval as M

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/as_{os.getpid()}")
    model = t.model

    raw = json.load(open(a.rates))
    rates = {k.replace(".params.", "."): {int(b): v for b, v in d.items()}
             for k, d in raw["components"].items()}
    sens = {k.replace(".params.", "."): v for k, v in json.load(open(a.sens)).items()}
    shape = {int(k): v for k, v in json.load(open(a.shape)).items()}

    # 模型驱动起点
    bits = {c: 16 for c in COMPS}
    def R(c, b):
        return rates[c].get(b, rates[c][min(rates[c], key=lambda x: abs(x - b))])
    def D(c, b):
        return sens.get(c, 0.0) * shape.get(b, 0.0)
    total = sum(R(c, 16) for c in COMPS)
    while total > a.budget_mb * 1e6:
        best = None
        for c in COMPS:
            nxt = [x for x in LEVELS if x < bits[c]]
            if not nxt:
                continue
            nb = max(nxt)
            save = R(c, bits[c]) - R(c, nb)
            lost = max(0.0, D(c, nb) - D(c, bits[c]))
            sc = lost / max(save, 1)
            if best is None or sc < best[0]:
                best = (sc, c, nb, save)
        if best is None:
            break
        _, c, nb, save = best
        bits[c] = nb
        total -= save

    if a.start_bits:
        for kv in a.start_bits.split(","):
            k, v = kv.split("=")
            if k.strip() in bits:
                bits[k.strip()] = int(v)

    orig = {}
    for grp in ["fg", "bg"]:
        for n, p in getattr(model, grp).params.items():
            orig[f"{grp}.{n}"] = p.data.clone()
    for n in ["rots", "transls"]:
        if n in model.motion_bases.params:
            orig[f"bases.{n}"] = model.motion_bases.params[n].data.clone()

    def quant(x, b):
        x = x.detach().float()
        fin = torch.isfinite(x)
        amax = float((x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0)).clamp_min(1e-6))
        xc = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax), -amax, amax)
        qmax = 2 ** (b - 1) - 1
        s = amax / qmax
        return torch.clamp(torch.round(xc / s), -qmax - 1, qmax).float() * s

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)

    def score(bits_):
        for k, v in orig.items():
            if k.startswith("bases."):
                n = k.split(".")[1]
                p = model.motion_bases.params[n]
                p.data = quant(v, bits_.get(n, bits_.get("bases", 16))).to(p.dtype)
            else:
                grp, n = k.split(".")
                p = getattr(model, grp).params[n]
                p.data = quant(v, bits_.get(f"{grp}.{n}", bits_.get(grp, 16))).to(p.dtype)
        ps = []
        with torch.no_grad():
            used = 0
            for i in range(len(val)):
                if i % 2 == 0:
                    continue          # 只用标定半边（偶数索引）
                if used >= a.frames:
                    break
                used += 1
                fn = val.frame_names[i]
                w2c = val.w2cs[i:i + 1].to(dev); K = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, K, (Wd, H))
                pred = out["img"][0].detach().float().clamp(0, 1)
                gt = val.imgs[i].to(dev).float()[..., :3]
                mp = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
                if not osp.exists(mp):
                    mp = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
                if not osp.exists(mp):
                    vis = torch.ones((H, Wd), device=dev)
                else:
                    m_ = np.array(Image.open(mp).convert("L"))
                    if (m_.shape[0], m_.shape[1]) != (H, Wd):
                        m_ = np.array(Image.open(mp).convert("L").resize((Wd, H), Image.NEAREST))
                    vis = (torch.from_numpy(m_).to(dev).float() / 255.0 > 0.5).float()
                ps.append(M._psnr(pred, gt, vis))
        return float(np.mean(ps)) if ps else float("nan")

    cur = dict(bits)
    cur_bytes = sum(R(c, cur[c]) for c in COMPS)
    cur_s = score(cur)
    trace = [{"iter": 0, "MB": cur_bytes / 1e6, "psnr_calib": cur_s, "bits": dict(cur)}]
    print(f"start {cur_bytes/1e6:.2f} MB  calib {cur_s:.3f} dB")
    for it in range(1, a.iters + 1):
        best = (cur_s, None, None, None)
        for c in COMPS:
            for nb in LEVELS:
                if nb == cur[c]:
                    continue
                cand = dict(cur); cand[c] = nb
                cb = sum(R(cc, cand[cc]) for cc in COMPS)
                if cb > a.budget_mb * 1e6:
                    continue
                s = score(cand)
                if s > best[0] + 1e-4:
                    best = (s, c, nb, cb)
        if best[1] is None:
            print(f"  iter {it}: no improving move")
            break
        cur_s, c, nb, cur_bytes = best
        cur[c] = nb
        trace.append({"iter": it, "MB": cur_bytes / 1e6, "psnr_calib": cur_s, "bits": dict(cur)})
        print(f"  iter {it}: {c}->{nb}  {cur_bytes/1e6:.2f} MB  calib {cur_s:.3f} dB")
    json.dump({"budget_mb": a.budget_mb, "final": {"MB": cur_bytes / 1e6, "bits": cur},
               "psnr_calib": cur_s, "trace": trace}, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
