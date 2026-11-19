#!/usr/bin/env python3
"""nested_rd.py — 把"自适应投递顺序"当成一个**可分级码流**来评估。

问题：如果客户端可以停在任意前缀，那么同一条流在**所有**码率上都应接近
"为该码率单独分配比特"的最优模型。前者是 rate-scalable，后者是 per-budget optimal。

做法：读 `prefix_<scene>.json` 里 direct 顺序的每一级比特向量，逐级应用后
在**全部评测帧**（奇偶切分的评测半边）上渲染评测，得到 (MB, PSNR) 曲线，
再与 per-budget 分配曲线（r4d_rd 的工作点）比较。

用法：
  python3 scripts/nested_rd.py --ckpt <ckpt> --data-dir <scene> \
      --prefix work/final/prefix_backpack.json --out work/final/nested_backpack.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--frames", type=int, default=64, help="评测帧上限（奇偶切分后取一半）")
    ap.add_argument("--dataset", choices=["iphone", "nvidia"], default="iphone")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
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
                                        OptimizerConfig(), work_dir=f"/tmp/nrd_{os.getpid()}")
    model = t.model

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)

    pr = json.load(open(a.prefix))
    stages = [r for r in pr["results"] if r["order"] == "direct"]

    orig = {}
    for grp in ["fg", "bg"]:
        mod = getattr(model, grp)
        for n, p in mod.params.items():
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

    rows = []
    for st in stages:
        bits = st["bits"]
        for k, v in orig.items():
            if k.startswith("bases."):
                n = k.split(".")[1]
                p = model.motion_bases.params[n]
                p.data = quant(v, bits.get(k, bits.get("bases", 16))).to(p.dtype)
            else:
                grp, n = k.split(".")
                p = getattr(model, grp).params[n]
                b = bits.get(k)
                if b is None:
                    b = bits.get("coefs", bits.get("fg", 16)) if n == "motion_coefs" else bits.get(grp, 16)
                p.data = quant(v, b).to(p.dtype)
        ps, pfg = [], []
        with torch.no_grad():
            for i in range(min(a.frames, len(val))):
                if i % 2 == 0:
                    continue
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
        rows.append({"stage": st["stage"], "MB": st["MB"], "bytes": st["bytes"],
                     "psnr_all": round(float(np.mean(ps)), 4) if ps else None, "n": len(ps)})
        print(f"stage{st['stage']:02d} {st['MB']:8.3f} MB  psnr_all={rows[-1]['psnr_all']}")
    json.dump({"rows": rows, "scene": osp.basename(ddir)}, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
