#!/usr/bin/env python3
"""render_gallery.py — 渐进交付的视觉证据（真实解码结果，不用示意图）。

对每个场景：从 base layer 到 16-bit 参考，取 5 个码流前缀（base / 早期 / 首次达标 /
达标后 2× / 16-bit），用与码流一致的位平面解码渲染同一留出帧，拼成一张图；
每个面板角标真实累计传输字节与 PSNR。

用法：
  python3 scripts/render_gallery.py --scene truck --ckpt ... --data-dir ... \
      --dataset nvidia --factor 2 --frame 3 --gpu 0 --out paper/figures/gallery_truck.pdf
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

from packets import Plane, truncate, dequant  # noqa: E402

COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/gal_{os.getpid()}")
    model = t.model
    src = {c: getattr(model, c.split(".")[0]).params[c.split(".")[1]].data.clone() for c in COMPS}
    planes = {c: Plane(c, x.float().cpu().numpy()) for c, x in src.items()}

    def set_bits(bits):
        for c in COMPS:
            g, n = c.split(".")
            pl = planes[c]
            p = getattr(model, g).params[n]
            p.data = torch.from_numpy(dequant(truncate(pl.q, bits.get(c, 16)), pl.scale)
                                      .reshape(pl.shape)).to(device=p.device, dtype=p.dtype)

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)

    d = json.load(open(f"{_ROOT}/work/final/p0_{a.scene}.json"))
    ref_psnr = d["ref_psnr"]
    rows = [r for r in d["results"] if r["order"] == "direct"]
    q = next((r for r in rows if r["psnr_first"] >= ref_psnr - 0.5
              and r["psnr_first_fg"] >= d["summary"]["uniform"]["ref_fg"] - 0.5), rows[-1])
    base = rows[0]
    early = next((r for r in rows if r["bytes"] >= base["bytes"] + 0.25 * (q["bytes"] - base["bytes"])), q)
    mid = next((r for r in rows if r["bytes"] >= 2 * q["bytes"]), rows[-1])
    states = [("base layer", base), ("early prefix", early), ("first playable", q),
              ("16-bit reference", None)]

    i = a.frame
    with torch.no_grad():
        w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
        tid = int(val.time_ids[i].item())
        H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
        panels = []
        for name, r in states:
            if r is None:
                set_bits({})
                mb = float(json.load(open(f"{_ROOT}/work/rates_{a.scene}.json"))["total_MB"]["16"])
                pf = d["results"][-1].get("per_frame") or []
                ps = next((x["full"] for x in pf if x["frame_id"] == i), ref_psnr)
            else:
                set_bits(r["bits"]); mb = r["bytes"] / 1e6
                pf = r.get("per_frame") or []
                ps = next((x["full"] for x in pf if x["frame_id"] == i), r["psnr_first"])
            out = model.render(tid, w2c, Kk, (Wd, H))
            img = out["img"][0].detach().float().clamp(0, 1).cpu().numpy()
            panels.append((name, img, mb, ps))
        gt = val.imgs[i].cpu().numpy()[..., :3]

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
    for ax, (name, img, mb, ps) in zip(axes, panels):
        ax.imshow(img)
        cam_i = int(val.frame_names[i].split("_")[0])
        ax.set_title(f"{name}\n{mb:.2f} MB · {ps:.2f} dB (cam {cam_i}, frame {i})", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, bbox_inches="tight", dpi=330)
    print("wrote", a.out, "frame", i, "of", a.scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
