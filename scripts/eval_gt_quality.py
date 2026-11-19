#!/usr/bin/env python3
"""eval_gt_quality.py — 对**真实 held-out 视频 GT**（不只是各自的 16-bit 参考）测绝对画质。

动机（可视画质层）：论文现有判据是"与该模型自己的 16-bit 参考差 ≤0.5 dB"，
读者有理由问：接近一个 13 dB 的内部参考，是否等于可以播放？这里给出三层里的"可视画质"层：
  · 与 GT 的绝对 PSNR（全图 co-visibility 掩膜 / 动态区掩膜）
  · SSIM（全图）
  · LPIPS（全图，AlexNet 权重，输入 [-1,1]）
对同一场景、同一批评测帧、同一包字节预算，比较：16-bit 参考、最强固定顺序、full-gain、
joint-gain、Ours、tail-cvar 各自**首次达标前缀**的解码状态。

用法：
  python3 scripts/eval_gt_quality.py --scene truck --gpu 0 [--limit 30]
产出：work/final/gtq_<scene>.json
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

TRUST = f"{_DATA}"
sys.path.insert(0, f"{TRUST}/METHOD01_CAPE4D/scripts")
sys.path.insert(0, f"{TRUST}/JOURNAL_DATA/_src/shape-of-motion-main")

LEVELS = [4, 6, 8, 12, 16]
COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]
TAGS = [("best_fixed", "p0"), ("ours", "p0"), ("full_gain", "adaptive_full_gain"),
        ("joint_gain", "adaptive_joint_gain"), ("tail_cvar", "risk_tail_cvar")]


def psnr(pred, gt, mask):
    m = (mask > 0.5)
    if int(m.sum()) < 1:
        return float("nan")
    err = ((pred - gt) ** 2)[m].mean().item()
    return float(10.0 * np.log10(1.0 / max(err, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--data-dir")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--rates")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    sc = a.scene
    root = f"{_ROOT}"
    ck = a.ckpt or (f"{_DATA}/checkpoints/A_nat/ckpt_last.ckpt"
                    if sc == "backpack" else f"{root}/work/{sc}_nat/ckpt_last.ckpt")
    dd = a.data_dir or {
        "backpack": f"{_DATA}/scenes/backpack",
        "sriracha": f"{_DATA}/scenes/sriracha-tree",
        "truck": f"{_DATA}/Truck",
        "umbrella": f"{_DATA}/Umbrella",
        "jumping": f"{_DATA}/Jumping",
    }[sc]
    ds = a.dataset or ("nvidia" if sc in ("truck", "umbrella", "jumping") else "iphone")
    fac = a.factor if ds == "nvidia" else 1
    rates = a.rates or f"{root}/work/rates_{sc}.json"
    out = a.out or f"{root}/work/final/gtq_{sc}.json"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, f"{root}/scripts")
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from packets import Plane, truncate, dequant

    dev = torch.device("cuda:0")
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(ck, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/gtq_{os.getpid()}")
    model = t.model

    raw = json.load(open(rates))
    fixed = raw.get("fixed_bytes", 0)

    orig, bases = {}, {}
    for c in COMPS:
        g, n = c.split(".")
        if g == "fg" and n == "motion_coefs":
            orig[c] = model.fg.params["motion_coefs"].data.clone()
        elif g in ("fg", "bg") and n in getattr(model, g).params:
            orig[c] = getattr(model, g).params[n].data.clone()
    for n in ("rots", "transls"):
        if n in model.motion_bases.params:
            bases[f"bases.{n}"] = model.motion_bases.params[n].data.clone()

    planes = {c: Plane(c, x.detach().float().cpu().numpy())
              for c, x in list(orig.items()) + list(bases.items())}
    cache = {}
    for c, pl in planes.items():
        for b in LEVELS:
            cache[(c, b)] = torch.from_numpy(dequant(truncate(pl.q, b), pl.scale).reshape(pl.shape)).to(dev)

    # 数据集
    ddir = osp.abspath(dd)
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    if ds == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset
        val = iPhoneDataset(data_dir=ddir, factor=fac, split="val",
                            scene_norm_dict=snd, load_from_cache=True)
        cams = None
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset
        val = NvidiaDataset(data_dir=ddir, factor=fac, split="val",
                            scene_norm_dict=snd, load_from_cache=True)
        cams = [1, 2, 3]

    n = len(val)
    eval_ids = []
    if cams:
        quota = max(1, a.frames // len(cams))
        for c_ in cams:
            got = 0
            for i in range(n):
                fid_cam = int(val.frame_names[i].split("_")[0])
                if fid_cam != c_ or i % 2 == 0:
                    continue
                if got < quota:
                    eval_ids.append(i); got += 1
    else:
        for i in range(n):
            if i % 2 == 1 and len(eval_ids) < a.frames:
                eval_ids.append(i)
    if a.limit:
        eval_ids = eval_ids[:a.limit]
    print(f"{sc}: eval frames {len(eval_ids)} (cams={cams})")

    def apply_bits(bits):
        for c, x in orig.items():
            g, nm = c.split(".")
            dq = cache[(c, bits[c])]
            if g == "fg" and nm == "motion_coefs":
                model.fg.params["motion_coefs"].data = dq.to(model.fg.params["motion_coefs"].dtype)
            else:
                p = getattr(model, g).params[nm]
                p.data = dq.to(p.dtype)
        for c in bases:
            nm = c.split(".")[1]
            p = model.motion_bases.params[nm]
            p.data = cache[(c, bits.get(c, 16))].to(p.dtype)

    import lpips as lpips_lib
    from skimage.metrics import structural_similarity as ssim_fn
    lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev)

    def masks(i):
        fn = val.frame_names[i]
        H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
        mp = osp.join(ddir, f"flow3d_preprocessed/covisible/{fac}x/val", fn + ".png")
        if not osp.exists(mp):
            mp = osp.join(ddir, f"covisible/{fac}x/val", fn + ".png")
        m_ = np.array(Image.open(mp).convert("L"))
        if (m_.shape[0], m_.shape[1]) != (H, Wd):
            m_ = np.array(Image.open(mp).convert("L").resize((Wd, H), Image.NEAREST))
        vis = (torch.from_numpy(m_).to(dev).float() / 255.0 > 0.5).float()
        fgp = osp.join(ddir, f"flow3d_preprocessed/track_anything/{fac}x", fn + ".png")
        fa = np.array(Image.open(fgp).convert("L"))
        if (fa.shape[0], fa.shape[1]) != (H, Wd):
            fa = np.array(Image.open(fgp).convert("L").resize((Wd, H), Image.NEAREST))
        fgm = vis * torch.from_numpy((fa > 127).astype(np.float32)).to(dev)
        return vis, fgm

    def measure(bits):
        apply_bits(bits)
        rows = []
        with torch.no_grad():
            for i in eval_ids:
                w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out_ = model.render(tid, w2c, Kk, (Wd, H))
                pred = out_["img"][0].detach().float().clamp(0, 1)
                gt = val.imgs[i].to(dev).float()[..., :3]
                vis, fgm = masks(i)
                p_full = psnr(pred, gt, vis)
                p_fg = psnr(pred, gt, fgm)
                pn = pred.cpu().numpy(); gn = gt.cpu().numpy()
                ssim = float(ssim_fn(gn, pn, channel_axis=2, data_range=1.0))
                l1 = (pred * 2 - 1).permute(2, 0, 1).unsqueeze(0)
                l2 = (gt * 2 - 1).permute(2, 0, 1).unsqueeze(0)
                lp_v = float(lp(l1, l2).item())
                rows.append({"frame_id": int(val.frame_names[i].split("_")[-1]),
                             "camera": int(val.frame_names[i].split("_")[0]),
                             "psnr_full": round(p_full, 4), "psnr_fg": round(p_fg, 4),
                             "ssim": round(ssim, 4), "lpips": round(lp_v, 4)})
        return rows

    def state_at(tag, run_json):
        d = json.load(open(run_json))
        order, fp = "direct", d["summary"]["direct"]["bytes_to_ref_minus_0.5"]
        if tag == "best_fixed":
            fixed_orders = ["uniform", "roundrobin", "posfirst", "dynfirst", "oneshot", "mpprefix"]
            order = min(fixed_orders, key=lambda k: d["summary"][k]["bytes_to_ref_minus_0.5"] or 1e18)
            fp = d["summary"][order]["bytes_to_ref_minus_0.5"]
        st = [r for r in d["results"] if r["order"] == order and r["bytes"] >= fp]
        bits = st[0]["bits"] if st else [r for r in d["results"] if r["order"] == order][-1]["bits"]
        return bits, fp, order

    res = {"scene": sc, "n_frames": len(eval_ids), "dataset": ds, "rows": {}}
    res["rows"]["ref16"] = measure(dict((c, 16) for c in COMPS))
    for tag, stem in TAGS:
        rj = f"{root}/work/final/{stem}_{sc}.json"
        if not os.path.exists(rj):
            print(f"  ⚠ 缺 {rj}")
            continue
        bits, fp, order_used = state_at(tag, rj)
        res["rows"][tag] = measure(bits)
        res["rows"][tag][0]["_first_pass_MB"] = round(fp / 1e6, 3) if fp else None
        res["rows"][tag][0]["_order"] = order_used

    summ = {}
    for k, rows in res["rows"].items():
        fpv = rows[0].pop("_first_pass_MB", None)
        ordv = rows[0].pop("_order", None)
        agg = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in ("psnr_full", "psnr_fg", "ssim", "lpips")}
        agg["first_pass_MB"] = fpv
        agg["order"] = ordv
        summ[k] = agg
    res["summary"] = summ
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")
    for k, v in summ.items():
        print(f"  {k:10s} PSNR {v['psnr_full']:6.2f}/{v['psnr_fg']:6.2f}  SSIM {v['ssim']:.4f}  LPIPS {v['lpips']:.4f}  ({v['first_pass_MB']} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
