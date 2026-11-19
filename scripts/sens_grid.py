#!/usr/bin/env python3
"""sens_grid.py — 与位平面码流同一网格上的单组灵敏度（校准帧）。

对每个场景：16-bit 网格上把**一组**属性截断到 4 bit，其余保持 16 bit，渲染校准帧，
记录全图与动态区域 PSNR 的损失。与 packets.py 的口径一致（同一 16-bit 网格），
因此可以和交付码流的数字并排放。

输出：work/sens_p0_<scene>.json
用法： python3 scripts/sens_grid.py --ckpt ... --data-dir ... [--dataset nvidia --factor 2] --out ...
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
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--calib-span", type=int, default=12)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image
    import m1_eval as M
    from packets import Plane, truncate, dequant

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/sg_{os.getpid()}")
    model = t.model

    src = {}
    for c in COMPS:
        g, n = c.split(".")
        src[c] = (getattr(model, g).params[n].data.clone())

    planes = {c: Plane(c, src[c].float().cpu().numpy()) for c in COMPS}

    def set_bits(bits):
        for c in COMPS:
            g, n = c.split(".")
            pl = planes[c]
            q = truncate(pl.q, bits.get(c, 16))
            p = getattr(model, g).params[n]
            p.data = torch.from_numpy(dequant(q, pl.scale).reshape(pl.shape)).to(device=p.device, dtype=p.dtype)

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)
    ids = [i for i in range(len(val)) if i % 2 == 0][:a.calib_span]

    def evaluate():
        ps, pfg = [], []
        with torch.no_grad():
            for i in ids:
                fn = val.frame_names[i]
                w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, Kk, (Wd, H))
                pred = out["img"][0].detach().float().clamp(0, 1)
                gt = val.imgs[i].to(dev).float()[..., :3]
                mp = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
                if not osp.exists(mp):
                    mp = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
                m = np.array(Image.open(mp).convert("L")) if osp.exists(mp) else None
                if m is None:
                    vis = torch.ones((H, Wd), device=dev)
                else:
                    if (m.shape[0], m.shape[1]) != (H, Wd):
                        m = np.array(Image.open(mp).convert("L").resize((Wd, H), Image.NEAREST))
                    vis = (torch.from_numpy(m).to(dev).float() / 255.0 > 0.5).float()
                ps.append(M._psnr(pred, gt, vis))
                fgp = osp.join(ddir, f"flow3d_preprocessed/track_anything/{a.factor}x", fn + ".png")
                if osp.exists(fgp):
                    fa = np.array(Image.open(fgp).convert("L"))
                    if (fa.shape[0], fa.shape[1]) != (H, Wd):
                        fa = np.array(Image.open(fgp).convert("L").resize((Wd, H), Image.NEAREST))
                    fgm = vis * torch.from_numpy((fa > 127).astype(np.float32)).to(dev)
                    if float(fgm.sum()) >= 1:
                        pfg.append(M._psnr(pred, gt, fgm))
        return (float(np.mean(ps)) if ps else float("nan"),
                float(np.mean(pfg)) if pfg else float("nan"))

    set_bits({})
    ref_full, ref_fg = evaluate()
    out = {"ref_full": round(ref_full, 4), "ref_fg": round(ref_fg, 4), "calib_ids": ids, "groups": {}}
    print(f"16-bit calib: full={ref_full:.3f} fg={ref_fg:.3f}")
    for c in COMPS:
        set_bits({c: 4})
        f, g = evaluate()
        out["groups"][c] = {"loss_full": round(ref_full - f, 4), "loss_fg": round(ref_fg - g, 4)}
        print(f"  {c:16s} 4bit loss full={ref_full - f:6.2f} dB  fg={ref_fg - g:6.2f} dB")
    set_bits({})
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
