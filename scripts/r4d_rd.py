#!/usr/bin/env python3
"""r4d_rd.py — 4D 表示（SoM/flow3d 系）的码率-失真实验台。

为什么需要它：本窗口的候选主干是「时间维 vs 空间维的码率分配」。要判断它是不是
真问题，先要能量出来：同一个 ckpt 在**不同分配方案**下的 (bytes, PSNR) 两个点。

三个动作：
  1) `--mode rate`  只做码率分解：各分量在 16/8/6/4 bit 下的实际字节数（zstd 熵编码）。
  2) `--mode alloc` 按给定比特方案量化→反量化→渲染评测，出 (bytes, PSNR)。
  3) `--mode sweep` 同一脚本内跑一组方案（便于 8 卡并行分片）。

编码模型（写进论文时要照实描述）：
  每个属性张量 → 逐张量对称均匀量化到 b bit → uint 序列 → zstd(level=19)。
  不写算术编码器；zstd 会**高估**真实码率，因此报告的码率是偏保守的上界。

用法：
  python3 scripts/r4d_rd.py --mode rate --ckpt <ckpt> --data-dir <scene>
  python3 scripts/r4d_rd.py --mode alloc --ckpt <ckpt> --data-dir <scene> \
      --bits 'fg=8,bg=8,bases=16,coefs=8' --frames 12 --out work/rd_x.json
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
import zlib

import numpy as np
import torch
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

TRUST = f"{_DATA}"
sys.path.insert(0, f"{TRUST}/METHOD01_CAPE4D/scripts")
sys.path.insert(0, f"{TRUST}/JOURNAL_DATA/_src/shape-of-motion-main")

ATTRS_FG = ["means", "scales", "colors", "quats", "opacities", "motion_coefs"]
ATTRS_BG = ["means", "scales", "colors", "quats", "opacities"]


def zstd_bytes(arr: np.ndarray) -> int:
    """uint8/uint16 numpy 数组的熵编码字节数（走 zlib-9 作为 zstd 的保守替身）。"""
    return len(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9))


def quantize(t: torch.Tensor, bits: int) -> tuple[torch.Tensor, dict]:
    """对称均匀量化 + 反量化。返回 (dequant tensor, codec meta)。"""
    if bits >= 32:
        return t.clone(), {"bits": 32, "raw": True}
    x = t.detach().float()
    fin = torch.isfinite(x)
    # colors 是 logit，训练后含 +-inf（sigmoid 饱和）；用有限值的 99.99 分位做鲁棒尺度，
    # 饱和项在反量化后仍保持饱和（sigmoid(±amax)≈0/1），不引入可见误差。
    amax = x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0, device=x.device)
    amax = amax.clamp_min(1e-6)
    x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=float(amax), neginf=-float(amax)), -amax, amax)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int16 if bits <= 16 else torch.int32)
    deq = q.float() * scale
    return deq, {"bits": bits, "scale": float(scale), "q": q}


def coded_bytes(q: torch.Tensor, bits: int) -> int:
    if bits > 16:
        return q.numel() * 4
    if bits > 8:
        return zstd_bytes(q.to(torch.int16).cpu().numpy())
    return zstd_bytes(q.to(torch.int8).clamp(-127, 127).cpu().numpy())


def parse_bits(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rate", "alloc"], default="rate")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bits", default="fg=12,bg=12,bases=16,coefs=12")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", choices=["iphone", "nvidia"], default="iphone")
    ap.add_argument("--cam", type=int, default=-1, help="只保留该相机 id 的帧（多视场景的跨机位标定/测试）")
    ap.add_argument("--frame-start", type=int, default=0, help="从第几个 val 帧开始（时间块切分）")
    ap.add_argument("--index-parity", choices=["all", "even", "odd"], default="all",
                    help="标定/评测帧的奇偶划分（even 标定、odd 评测，两者不相交）")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="灵敏度标定用 train（与评测帧不重叠，避免泄漏）")
    ap.add_argument("--knot", type=int, default=1,
                    help="运动基表的时间抽稀步长：1=原始逐帧；s>1 表示只保留每 s 帧一个结点并线性插值回全帧")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    import m1_eval as M

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    work = f"/tmp/r4d_reload_{os.getpid()}"
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=work)
    model = t.model
    bits = parse_bits(a.bits)

    report = {"ckpt": a.ckpt, "scene": osp.basename(ddir), "mode": a.mode}

    # ---------- 码率分解 ----------
    comp = {}
    total = 0
    for grp, attrs in (("fg", ATTRS_FG), ("bg", ATTRS_BG)):
        if not hasattr(model, grp):
            continue
        for name in attrs:
            if name not in getattr(model, grp, None).params:
                continue
            p = getattr(model, grp).params[name].detach()
            b = bits.get(f"{grp}.{name}")
            if b is None:
                b = bits.get("coefs", bits.get("fg", 12)) if name == "motion_coefs" else bits.get(grp, bits.get(name, 12))
            _, meta = quantize(p, b)
            if meta.get("raw"):
                nbytes = p.numel() * 4
            else:
                nbytes = coded_bytes(meta["q"], b)
            comp[f"{grp}.{name}"] = {"bits": b, "numel": int(p.numel()), "bytes": int(nbytes)}
            total += nbytes
    for name in ["rots", "transls"]:
        if name in model.motion_bases.params:
            p = model.motion_bases.params[name].detach()
            if a.knot > 1:
                nk = len(range(0, p.shape[1], a.knot))
                p = p[:, :max(nk, 2), :]
            b = bits.get(f"bases.{name}", bits.get("bases", 16))
            _, meta = quantize(p, b)
            nbytes = coded_bytes(meta["q"], b) if not meta.get("raw") else p.numel() * 4
            comp[f"motion_bases.{name}"] = {"bits": b, "numel": int(p.numel()), "bytes": int(nbytes)}
            total += nbytes
    for name in ["Rs", "ts"]:
        if hasattr(model, "camera_poses") and name in model.camera_poses.params:
            p = model.camera_poses.params[name].detach()
            nbytes = p.numel() * 4
            comp[f"camera_poses.{name}"] = {"bits": 32, "numel": int(p.numel()), "bytes": int(nbytes)}
            total += nbytes

    report["components"] = comp
    report["total_bytes"] = int(total)
    report["total_MB"] = round(total / 1e6, 3)
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]["bytes"])[:8]:
        print(f"  {k:26s} {v['bits']:>2d}b  {v['numel']:>9d}  {v['bytes']/1e6:8.3f} MB  ({100*v['bytes']/max(total,1):5.1f}%)")
    print(f"  {'TOTAL':26s}      {'':>9s}  {total/1e6:8.3f} MB")

    if a.mode == "rate":
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)
            print(f"wrote {a.out}")
        return 0

    # ---------- 时间结点抽稀（仅影响运动基表，T 轴）----------
    if a.knot > 1:
        for name in ["rots", "transls"]:
            p_ = model.motion_bases.params[name].data          # (K, T, d)
            K, T, d = p_.shape
            idx = torch.arange(0, T, a.knot, device=p_.device)
            if idx[-1] != T - 1:
                idx = torch.cat([idx, torch.tensor([T - 1], device=p_.device)])
            sub = p_[:, idx, :]                                # (K, Tk, d)
            # 线性插值回 T
            xs = idx.float() / (T - 1)
            xt = torch.linspace(0, 1, T, device=p_.device)
            j = torch.searchsorted(xs, xt.clamp(0, 1), right=True) - 1
            j = j.clamp(0, len(idx) - 2)
            w = ((xt - xs[j]) / (xs[j + 1] - xs[j]).clamp_min(1e-8)).clamp(0, 1)
            recon = sub[:, j, :] * (1 - w)[None, :, None] + sub[:, j + 1, :] * w[None, :, None]
            model.motion_bases.params[name].data = recon.to(p_.dtype)
        report["knot"] = a.knot

    # ---------- 量化 + 渲染评测 ----------
    for grp, attrs in (("fg", ATTRS_FG), ("bg", ATTRS_BG)):
        mod = getattr(model, grp, None)
        if mod is None:
            continue
        for name in attrs:
            if name not in mod.params:
                continue
            b = bits.get(f"{grp}.{name}")
            if b is None:
                if name == "motion_coefs":
                    b = bits.get("coefs", bits.get("fg", 12))
                else:
                    b = bits.get(grp, bits.get(name, 12))
            deq, meta = quantize(mod.params[name].data, b)
            mod.params[name].data = deq.to(mod.params[name].dtype)

    for name in ["rots", "transls"]:
        if name in model.motion_bases.params:
            b = bits.get("bases", 16)
            deq, _ = quantize(model.motion_bases.params[name].data, b)
            model.motion_bases.params[name].data = deq.to(model.motion_bases.params[name].dtype)

    if a.dataset == "nvidia":
        from flow3d.data.nvidia_dataset import NvidiaDataset as _DS
    else:
        from flow3d.data.iphone_dataset import iPhoneDataset as _DS
    from PIL import Image
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    val = _DS(data_dir=ddir, factor=a.factor, split=a.split,
                        scene_norm_dict=snd, load_from_cache=True)
    n = len(val) if a.cam >= 0 else min(a.frames, len(val))
    psnrs, fulls, rows = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            if i < a.frame_start:
                continue
            if a.cam >= 0 and int(val.frame_names[i].split("_")[0]) != a.cam:
                continue
            if a.index_parity != "all" and (i % 2 == 0) != (a.index_parity == "even"):
                continue
            fn = val.frame_names[i]
            w2c = val.w2cs[i:i + 1].to(dev)
            K = val.Ks[i:i + 1].to(dev)
            tid = int(val.time_ids[i].item())
            H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
            out = model.render(tid, w2c, K, (Wd, H))
            pred = out["img"][0].detach().float().clamp(0, 1)
            gt = val.imgs[i].to(dev).float()[..., :3]
            mpath = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/{a.split}", fn + ".png")
            if not osp.exists(mpath):
                mpath = osp.join(ddir, f"covisible/{a.factor}x/{a.split}", fn + ".png")
            if not osp.exists(mpath) and a.split != "val":
                mpath = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
            if not osp.exists(mpath) and a.split != "val":
                mpath = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
            if not osp.exists(mpath):
                # 无共视掩膜（train 划分常见）：退化为全图掩膜。灵敏度只需相对失真。
                vis = torch.ones((H, Wd), device=dev, dtype=torch.float32)
            else:
                m = np.array(Image.open(mpath).convert("L"))
                if (m.shape[0], m.shape[1]) != (H, Wd):
                    m = np.array(Image.open(mpath).convert("L").resize((Wd, H), Image.NEAREST))
                vis = (torch.from_numpy(m).to(dev).float() / 255.0 > 0.5).float()
            fgp = osp.join(ddir, f"flow3d_preprocessed/track_anything/{a.factor}x", fn + ".png")
            if osp.exists(fgp):
                fa = np.array(Image.open(fgp).convert("L"))
                if (fa.shape[0], fa.shape[1]) != (H, Wd):
                    fa = np.array(Image.open(fgp).convert("L").resize((Wd, H), Image.NEAREST))
                fg = torch.from_numpy((fa > 127).astype(np.float32)).to(dev)
            else:
                fg = torch.ones_like(vis)
            fgm = vis * fg
            if float(fgm.sum()) < 1:
                continue
            psnrs.append(M._psnr(pred, gt, fgm))
            fulls.append(M._psnr(pred, gt, vis))
            rows.append({"frame": fn, "val_index": i, "fg_psnr": float(psnrs[-1]),
                         "full_psnr": float(fulls[-1]), "n_fg_vis": float(fgm.sum())})
    report.update({
        "bits": bits, "n_eval": len(psnrs),
        "fg_psnr": round(float(np.mean(psnrs)), 4) if psnrs else None,
        "full_psnr": round(float(np.mean(fulls)), 4) if fulls else None,
        "render_s": round(time.time() - t0, 2),
        "per_frame": rows,
    })
    print(f"RESULT fg_psnr={report['fg_psnr']} full_psnr={report['full_psnr']} "
          f"bytes={report['total_MB']} MB  (n={len(psnrs)}, {report['render_s']}s)")
    if a.out:
        os.makedirs(osp.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
