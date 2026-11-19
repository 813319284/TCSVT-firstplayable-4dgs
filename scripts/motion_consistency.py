#!/usr/bin/env python3
"""motion_consistency.py — 首播前缀下的运动一致性（P1-1 下游代理指标）。

在首次达标前缀与 16-bit 参考下，渲染同一相机的连续帧，计算相邻帧差分图，
再与 GT 的差分图在动态掩膜上做相关（Pearson / Spearman）。差分图相关高表示
"运动在哪里发生"被保留，可作为动态场景理解/跟踪类下游任务的代理证据。

用法： python3 scripts/motion_consistency.py --scene truck --ckpt ... --data-dir ... \
        --dataset nvidia --factor 2 --frames 6 --gpu 0 --out work/motion_truck.json
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
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image
    from scipy.stats import pearsonr, spearmanr

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/mc_{os.getpid()}")
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
    hit = next(r for r in d["results"] if r["order"] == "direct"
               and r["bytes"] >= d["summary"]["direct"]["bytes_to_ref_minus_0.5"])
    # 同一相机、连续时间戳：取评测集里第一条相机的前 N 帧
    cam = int(val.frame_names[d["eval_ids"][0]].split("_")[0])
    ids = [i for i in range(len(val)) if int(val.frame_names[i].split("_")[0]) == cam][: a.frames + 1]

    def diff_corr(bits):
        set_bits(bits)
        preds, gts = [], []
        with torch.no_grad():
            for i in ids:
                fn = val.frame_names[i]
                w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, Kk, (Wd, H))
                preds.append(out["img"][0].detach().float().clamp(0, 1).cpu().numpy())
                gts.append(val.imgs[i].cpu().numpy()[..., :3].astype(np.float32))
        p = np.stack([np.abs(preds[i + 1] - preds[i]).mean(-1) for i in range(len(preds) - 1)])
        g = np.stack([np.abs(gts[i + 1] - gts[i]).mean(-1) for i in range(len(gts) - 1)])
        return pearsonr(p.ravel(), g.ravel()).statistic, spearmanr(p.ravel(), g.ravel()).statistic

    r_hit, s_hit = diff_corr(hit["bits"])
    r_ref, s_ref = diff_corr({})
    out = {"scene": a.scene, "camera": cam, "n_frames": len(ids),
           "prefix_MB": round(hit["bytes"] / 1e6, 2),
           "pearson_prefix": round(float(r_hit), 4), "spearman_prefix": round(float(s_hit), 4),
           "pearson_ref": round(float(r_ref), 4), "spearman_ref": round(float(s_ref), 4)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
