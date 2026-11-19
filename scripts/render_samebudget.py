#!/usr/bin/env python3
"""render_samebudget.py — 同字节对照渲染（同字节对照图的素材）。

对一个场景：取 Ours 的首次达标字节 B_ours，分别解码
  (a) 逐场景最强固定顺序在该字节预算内最后一个完整包边界的状态
  (b) Ours 在 B_ours 的状态
  (c) 16-bit 参考
渲染同一评测帧（含动态区域 ROI 裁剪），输出 PNG 供排版。

用法：
  python3 scripts/render_samebudget.py --scene truck --ckpt ... --data-dir ... \
     --dataset nvidia --factor 2 --frames 3,7,11 --gpu 0 --outdir work/samebudget/truck
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
FIXED = ["uniform", "dynfirst", "posfirst", "roundrobin", "sens", "oneshot", "mpprefix"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--frames", default="3", help="逗号分隔的 val 帧索引")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/sb_{os.getpid()}")
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
    summary = d["summary"]
    B = summary["direct"]["bytes_to_ref_minus_0.5"]
    valid = {k: v["MB"] for k, v in summary.items() if v.get("MB")}
    best_order = min((valid[k], k) for k in FIXED if k in valid)[1]
    best_rows = sorted([r for r in d["results"] if r["order"] == best_order], key=lambda r: r["bytes"])
    best_state = max((r for r in best_rows if r["bytes"] <= B), key=lambda r: r["bytes"])
    ours_rows = [r for r in d["results"] if r["order"] == "direct"]
    ours_state = next(r for r in ours_rows if r["bytes"] >= B)

    os.makedirs(a.outdir, exist_ok=True)
    meta = {"scene": a.scene, "budget_bytes": B, "best_order": best_order,
            "best_bytes": best_state["bytes"], "frames": {}}
    idxs = [int(x) for x in a.frames.split(",")]
    for i in idxs:
        rows = []
        for tag, bits in [("best", best_state["bits"]), ("ours", ours_state["bits"]), ("ref", None)]:
            set_bits({} if bits is None else bits)
            with torch.no_grad():
                w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, Kk, (Wd, H))
                img = out["img"][0].detach().float().clamp(0, 1).cpu().numpy()
            Image.fromarray((img * 255).astype(np.uint8)).save(
                osp.join(a.outdir, f"f{i}_{tag}.png"))
            rows.append(tag)
        meta["frames"][str(i)] = rows
        print(f"  frame {i}: {rows}")
    json.dump(meta, open(osp.join(a.outdir, "meta.json"), "w"), indent=1)
    print("wrote", a.outdir, "budget", round(B / 1e6, 2), "MB, best order", best_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
