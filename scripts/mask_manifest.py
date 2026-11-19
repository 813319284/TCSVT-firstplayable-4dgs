#!/usr/bin/env python3
"""mask_manifest.py — 记录参与评测的每个帧的掩膜存在性/覆盖率（P0-2）。

对 p0_<scene>.json 里的 eval_ids，检查 co-visibility mask 与 dynamic mask 是否存在、
尺寸是否匹配、有效像素比例；任何缺失直接报错。输出 work/mask_manifest_<scene>.json。

用法： python3 scripts/mask_manifest.py --scene truck --data-dir ... --factor 2 [--dataset nvidia]
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    W = os.environ.get("TCSVT_ROOT") or osp.dirname(osp.dirname(osp.abspath(__file__)))
    d = json.load(open(f"{W}/work/final/p0_{a.scene}.json"))
    ddir = osp.abspath(a.data_dir)

    import torch
    sys.path.insert(0, f"{_DATA}/_src/METHOD01_CAPE4D/scripts")
    sys.path.insert(0, f"{_DATA}/_src/shape-of-motion-main")
    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)

    out = {"scene": a.scene, "n_eval": len(d["eval_ids"]), "frames": []}
    for i in d["eval_ids"]:
        fn = val.frame_names[i]
        cam = int(fn.split("_")[0])
        cov = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
        if not osp.exists(cov):
            cov = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
        dyn = osp.join(ddir, f"flow3d_preprocessed/track_anything/{a.factor}x", fn + ".png")
        for name, p in [("covisible", cov), ("dynamic", dyn)]:
            if not osp.exists(p):
                raise SystemExit(f"缺 {name} mask: {p}（P0-2）")
        c = np.array(Image.open(cov).convert("L")) > 127
        g = np.array(Image.open(dyn).convert("L")) > 127
        if c.shape != g.shape:
            raise SystemExit(f"掩膜尺寸不一致: {c.shape} vs {g.shape}")
        inter = float((c & g).sum())
        if c.sum() == 0 or g.sum() == 0 or inter == 0:
            raise SystemExit(f"掩膜为空或不相交: {fn}")
        out["frames"].append({"frame_id": i, "name": fn, "camera": cam,
                              "covisible_frac": round(float(c.mean()), 4),
                              "dynamic_frac": round(float(g.mean()), 4),
                              "intersection_frac": round(inter / c.size, 4)})
    dst = a.out or f"{W}/work/mask_manifest_{a.scene}.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"{a.scene}: {len(out['frames'])} frames, mask coverage 100%, wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
