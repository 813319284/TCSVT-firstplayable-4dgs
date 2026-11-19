#!/usr/bin/env python3
"""render_compare.py — 定性对照图：同一留出帧在「16-bit 参考 / uniform 阈值前缀 / 我们阈值前缀」下的渲染。

用法：
  python3 scripts/render_compare.py --ckpt work/truck_nat/ckpt_last.ckpt \
      --data-dir DATA/Truck --dataset nvidia --factor 2 \
      --frame 3 --outdir work/vis_truck
"""
from __future__ import annotations

import argparse
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

CONFIGS = {
    "uniform_thr": "fg.means=16,fg.scales=4,fg.colors=4,fg.quats=4,fg.opacities=12,fg.motion_coefs=16,bg.means=16,bg.scales=8,bg.colors=4,bg.quats=4,bg.opacities=4",
    "ours_thr": "fg.means=12,fg.scales=4,fg.colors=4,fg.quats=4,fg.opacities=4,fg.motion_coefs=8,bg.means=12,bg.scales=8,bg.colors=8,bg.quats=4,bg.opacities=4",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--bits", default="")
    ap.add_argument("--tag", default="render")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/rc_{os.getpid()}")
    model = t.model

    bits = {}
    if a.bits:
        for kv in a.bits.split(","):
            k, v = kv.split("=")
            bits[k.strip()] = int(v)

    def q(x, b):
        x = x.detach().float()
        fin = torch.isfinite(x)
        amax = float((x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0)).clamp_min(1e-6))
        xc = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax), -amax, amax)
        qmax = 2 ** (b - 1) - 1
        s = amax / qmax
        return (torch.clamp(torch.round(xc / s), -qmax - 1, qmax).float() * s)

    for grp in ["fg", "bg"]:
        mod = getattr(model, grp)
        for n, p in mod.params.items():
            k = f"{grp}.{n}"
            b = bits.get(k)
            if b is None:
                b = bits.get("coefs", bits.get("fg", 16)) if n == "motion_coefs" else bits.get(grp, 16)
            p.data = q(p.data, b).to(p.dtype)
    for n in ["rots", "transls"]:
        if n in model.motion_bases.params:
            p = model.motion_bases.params[n]
            p.data = q(p.data, bits.get(f"bases.{n}", bits.get("bases", 16))).to(p.dtype)

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)
    i = a.frame
    with torch.no_grad():
        w2c = val.w2cs[i:i + 1].to(dev)
        K = val.Ks[i:i + 1].to(dev)
        tid = int(val.time_ids[i].item())
        H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
        out = model.render(tid, w2c, K, (Wd, H))
        pred = out["img"][0].detach().float().clamp(0, 1).cpu().numpy()
        gt = val.imgs[i].cpu().numpy()[..., :3]
    os.makedirs(a.outdir, exist_ok=True)
    Image.fromarray((pred * 255).astype(np.uint8)).save(osp.join(a.outdir, f"{a.tag}_pred.png"))
    Image.fromarray((gt * 255).astype(np.uint8)).save(osp.join(a.outdir, f"{a.tag}_gt.png"))
    print("wrote", a.outdir, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
