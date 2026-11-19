#!/usr/bin/env python3
"""render_crossview.py — 跨视角一致性组图素材（同一时间、不同相机）。

在与首播前缀相同的字节状态下，渲染同一时刻的 camera 1/2/3，并给出 16-bit 参考。

用法： python3 scripts/render_crossview.py --scene truck --ckpt ... --data-dir ... \
        --dataset nvidia --factor 2 --gpu 0 --outdir work/crossview/truck
"""
from __future__ import annotations

import argparse, json, os, os.path as osp, sys
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
    ap.add_argument("--scene", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--cams", default="1,2,3"); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image
    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(), CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(), OptimizerConfig(),
                                        work_dir=f"/tmp/cv_{os.getpid()}")
    model = t.model
    src = {c: getattr(model, c.split(".")[0]).params[c.split(".")[1]].data.clone() for c in COMPS}
    planes = {c: Plane(c, x.float().cpu().numpy()) for c, x in src.items()}

    def set_bits(bits):
        for c in COMPS:
            g, n = c.split(".")
            p = getattr(model, g).params[n]
            p.data = torch.from_numpy(dequant(truncate(planes[c].q, bits.get(c, 16)), planes[c].scale)
                                      .reshape(planes[c].shape)).to(device=p.device, dtype=p.dtype)

    DS = __import__("flow3d.data." + ("iphone_dataset" if a.dataset == "iphone" else "nvidia_dataset"),
                    fromlist=["x"])
    DS = DS.iPhoneDataset if a.dataset == "iphone" else DS.NvidiaDataset
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)
    d = json.load(open(f"{_ROOT}/work/final/p0_{a.scene}.json"))
    hit = next(r for r in d["results"] if r["order"] == "direct"
               and r["bytes"] >= d["summary"]["direct"]["bytes_to_ref_minus_0.5"])
    os.makedirs(a.outdir, exist_ok=True)
    cams = [int(x) for x in a.cams.split(",")]
    for cam in cams:
        ids = [i for i in range(len(val)) if int(val.frame_names[i].split("_")[0]) == cam]
        if not ids: continue
        i = ids[len(ids) // 2]
        for tag, bits in [("ours", hit["bits"]), ("ref", None)]:
            set_bits({} if bits is None else bits)
            with torch.no_grad():
                w2c = val.w2cs[i:i+1].to(dev); Kk = val.Ks[i:i+1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, Kk, (Wd, H))
                img = out["img"][0].detach().float().clamp(0, 1).cpu().numpy()
            Image.fromarray((img * 255).astype(np.uint8)).save(osp.join(a.outdir, f"cam{cam}_{tag}.png"))
        print(f"  cam {cam} frame {i}")
    print("wrote", a.outdir, "at", round(hit["bytes"]/1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
