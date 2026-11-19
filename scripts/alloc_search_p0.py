#!/usr/bin/env python3
"""alloc_search_p0.py — 按字节预算独立优化分配（新字节账本下的对照）。

对每个预算：
  1) 起点 A：单流在该预算下的位宽向量（来自 p0_<scene>.json 的 direct 轨迹）；
  2) 起点 B：全 4-bit；
  3) 在标定帧上做爬山：每步试所有"某组升一档 / 降一档"的单步移动，
     只保留总字节 ≤ 预算且实测双区域质量更高的解，直到无改进或到迭代上限；
  4) 用**评测帧**（与标定不相交）报最终质量。

字节一律按 packets.py 的真实序列化长度计（与主实验同一账本）。

用法： python3 scripts/alloc_search_p0.py --ckpt ... --data-dir ... --scene backpack \
        --budgets 8,11,14 --out work/final/p0_alloc_backpack.json
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

LEVELS = [4, 6, 8, 12, 16]
COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", default="iphone", choices=["iphone", "nvidia"])
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--budgets", required=True, help="逗号分隔的 MB 预算")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig,
                                BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig)
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
    fixed = json.load(open(f"{_ROOT}/work/rates_{a.scene}.json"))["fixed_bytes"] \
        if osp.exists(f"{_ROOT}/work/rates_{a.scene}.json") else 6912

    src = {c: getattr(model, c.split(".")[0]).params[c.split(".")[1]].data.clone() for c in COMPS}
    bases = {f"bases.{n}": model.motion_bases.params[n].data.clone()
             for n in ("rots", "transls") if n in model.motion_bases.params}
    planes = {c: Plane(c, x.float().cpu().numpy()) for c, x in list(src.items()) + list(bases.items())}

    def nbytes(c, b):
        pl = planes[c]
        n = len(pl.base_packet(LEVELS[0]))
        for i in range(1, LEVELS.index(b) + 1):
            n += len(pl.refine_packet(LEVELS[i - 1], LEVELS[i]))
        return n

    def total_bytes(bits):
        return (fixed + sum(nbytes(c, bits[c]) for c in COMPS)
                + sum(nbytes(c, LEVELS[-1]) for c in bases))

    def set_bits(bits):
        for c in COMPS:
            g, n = c.split(".")
            pl = planes[c]
            p = getattr(model, g).params[n]
            p.data = torch.from_numpy(dequant(truncate(pl.q, bits[c]), pl.scale).reshape(pl.shape)).to(device=p.device, dtype=p.dtype)

    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset as DS
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset as DS
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = DS(data_dir=ddir, factor=a.factor, split="val", scene_norm_dict=snd, load_from_cache=True)
    # G4：与主实验共用同一帧清单（不从本脚本重新构造）
    main_json = f"{_ROOT}/work/final/p0_{a.scene}.json"
    mj = json.load(open(main_json))
    calib, evals = mj["calib_ids"], mj["eval_ids"]
    print(f"  shared ids: calib {len(calib)}, eval {len(evals)}, 交集="
          f"{len(set(calib) & set(evals))}")

    def score(ids):
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

    # 单流在该预算下的状态（作为起点 A）
    stream_states = []
    p0 = f"{_ROOT}/work/final/p0_{a.scene}.json"
    if osp.exists(p0):
        d = json.load(open(p0))
        for r in d["results"]:
            if r["order"] == "direct":
                stream_states.append((r["bytes"], r["bits"]))

    out = {"scene": a.scene, "budgets": [], "fixed_bytes": fixed,
           "calib_ids": calib, "eval_ids": evals,
           "search": "hill climb, single-step moves, started from base layer and from "
                     "the stream state at that budget (locally optimised)"}
    for mb in [float(x) for x in a.budgets.split(",")]:
        budget = int(mb * 1e6)
        starts = [dict((c, LEVELS[0]) for c in COMPS)]
        cand = [b for (by, b) in stream_states if by <= budget]
        if cand:
            starts.append(dict(cand[-1]))
        best = None
        for st in starts:
            cur = dict(st)
            set_bits(cur)
            f0, g0 = score(calib)
            cur_score = f0 + g0
            for _ in range(a.iters):
                improved = False
                for c in COMPS:
                    for b in LEVELS:
                        if b == cur[c]:
                            continue
                        nxt = dict(cur); nxt[c] = b
                        if total_bytes(nxt) > budget:
                            continue
                        set_bits(nxt)
                        f, g = score(calib)
                        s = f + g
                        if s > cur_score + 1e-6:
                            cur, cur_score = nxt, s
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            if best is None or cur_score > best[0]:
                best = (cur_score, dict(cur))
        _, bits = best
        set_bits(bits)
        fe, ge = score(evals)
        row = {"budget_MB": mb, "bytes": total_bytes(bits), "eval_full": round(fe, 3),
               "eval_fg": round(ge, 3), "bits": bits}
        out["budgets"].append(row)
        print(f"  budget {mb} MB → {row['bytes']/1e6:.2f} MB, eval full={fe:.2f} fg={ge:.2f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
