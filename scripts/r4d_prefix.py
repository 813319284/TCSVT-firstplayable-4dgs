#!/usr/bin/env python3
"""r4d_prefix.py — 冷启动（first-playable）实验台：前缀字节 → 首段画质。

问题：给定同一个冻结 ckpt，把模型切成"可按字节前缀解码"的包序列，**升级顺序**不同，
首段可播放画质（前 K 个 val 帧的全图掩膜 PSNR）随已收字节怎么变？

三种顺序（都是同一套量化器、同一批帧、同一掩膜）：
  1. `uniform`        ：所有分量同步从 4→6→8→12→16 bit 升级（空间位平面顺序）
  2. `dynfirst`       ：先把 fg+运动基+coefs 升到 16 bit（"先发运动"的渐进基线），再升 bg
  3. `sens`           ：按实测 dB/MB 贪心顺序逐个分量升级（本稿的分配器）
  4. `prefixsens`     ：按"**前 K 帧**实测 dB/MB"贪心（前缀目标灵敏度，需要 sens_prefix 表）

输出：每个顺序的 (cum_bytes, psnr_first_K) 曲线，以及达到 "参考 − 0.5 dB" 所需字节。

用法：
  python3 scripts/r4d_prefix.py --ckpt <ckpt> --data-dir <scene> \
      --rates work/rates_backpack.json --sens work/sens_backpack.json \
      --frames 16 --out work/prefix_backpack.json
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

LEVELS = [4, 6, 8, 12, 16]
COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]


def zbytes(q: torch.Tensor, bits: int) -> int:
    if bits > 16:
        return q.numel() * 4
    if bits > 8:
        return len(zlib.compress(q.to(torch.int16).cpu().numpy().tobytes(), 9))
    return len(zlib.compress(q.to(torch.int8).clamp(-127, 127).cpu().numpy().tobytes(), 9))


def quant(x: torch.Tensor, bits: int):
    x = x.detach().float()
    fin = torch.isfinite(x)
    amax = x[fin].abs().quantile(0.9999) if fin.any() else torch.tensor(1.0, device=x.device)
    amax = float(amax.clamp_min(1e-6))
    x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax), -amax, amax)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return (q.float() * scale), q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--sens", default="")
    ap.add_argument("--sens-prefix", default="")
    ap.add_argument("--frames", type=int, default=16, help="前缀画质评测的前 K 个 val 帧")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--cam", type=int, default=-1, help="只保留该相机 id 的帧（多视场景的跨机位标定/测试）")
    ap.add_argument("--frame-start", type=int, default=0, help="从第几个 val 帧开始（时间块切分）")
    ap.add_argument("--index-parity", choices=["all", "even", "odd"], default="all",
                    help="兼容旧用法：同时作为校准与评测的奇偶划分（不推荐）")
    ap.add_argument("--calib-parity", choices=["all", "even", "odd"], default=None,
                    help="校准帧的奇偶划分（默认 even）")
    ap.add_argument("--eval-parity", choices=["all", "even", "odd"], default=None,
                    help="评测帧的奇偶划分（默认取 --index-parity）")
    ap.add_argument("--calib-cam", type=int, default=None,
                    help="校准用的相机 id（默认同 --cam）")
    ap.add_argument("--eval-cams", default="",
                    help="逗号分隔的评测相机列表（多视场景默认 1,2,3；显式排除训练相机 0）")
    ap.add_argument("--eval-cam", type=int, default=None,
                    help="评测用的相机 id（默认取 --cam）")
    ap.add_argument("--tol", type=float, default=0.5, help="双区域达标容差（dB）")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="双区域缺口里前景项的权重 λ")
    ap.add_argument("--dataset", choices=["iphone", "nvidia"], default="iphone")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--direct", action="store_true", help="直接前缀目标贪心（在标定窗口上重渲染候选包）")
    ap.add_argument("--direct-ref", type=float, default=0.0, help="直接贪心用的参考 PSNR（通常填 16-bit 前 K 帧）")
    ap.add_argument("--direct-steps", type=int, default=24)
    ap.add_argument("--lpips", action="store_true", help="前缀评测同时计算前景 LPIPS")
    ap.add_argument("--refresh-early", type=int, default=0,
                    help="前 N 个包每包重估，之后按 --refresh 的间隔重估（N=0 关闭）")
    ap.add_argument("--leaky", action="store_true",
                    help="诊断用：标定窗口与评测窗口重合（给出'全信息贪心'的上界）")
    ap.add_argument("--only-direct", action="store_true", help="只算 direct 顺序（用于精确计量标定成本）")
    ap.add_argument("--policy", choices=["step", "uniform", "adaptive"], default="step",
                    help="重估节奏：step=每包；uniform=每 --refresh 包；adaptive=τ 触发降频、停滞则恢复")
    ap.add_argument("--tau-hi", type=float, default=0.99)
    ap.add_argument("--stall-eps", type=float, default=0.02, help="降频期内缺口下降低于该值视为停滞")
    ap.add_argument("--objective", choices=["dual", "mean"], default="dual",
                    help="dual=两口径缺口之和（默认）；mean=只看全图 PSNR 的提升")
    ap.add_argument("--tau-freeze", type=float, default=0.0,
                    help=">0 时：在刷新点计算 Kendall tau，tau>=该值后冻结（触发式冻结策略）")
    ap.add_argument("--probe-n", type=int, default=4, help="mpprefix 基线用的探测前缀个数")
    ap.add_argument("--probe-every", type=int, default=8, help="mpprefix 探测前缀之间的包间隔")
    ap.add_argument("--refresh", type=int, default=1,
                    help="每多少个包刷新一次打分（1=每步重估；k>1=每 k 包重估一次；0=只在初始重估一次）")
    ap.add_argument("--direct-full", action="store_true", help="不在达标后停，继续走完 steps（用于阈值敏感性分析）")
    ap.add_argument("--ref-frames-only", action="store_true",
                    help="只算评测帧的 16-bit 逐帧参考（full/fg + 相机/帧号），供尾部指标复算用")
    ap.add_argument("--risk-alpha", type=float, default=0.9,
                    help="tail_cvar 评分用的 CVaR 水平（只取最差 (1-alpha) 比例的校准帧）")
    ap.add_argument("--adaptive-score", choices=["deficit", "full_gain", "joint_gain", "tail_cvar"], default="deficit",
                    help="逐包候选排序目标：deficit=论文式(3)（质量缺口正部/字节，默认）; "
                         "full_gain=全图边际 PSNR 增益/字节; joint_gain=双区域边际 PSNR 增益之和/字节")
    ap.add_argument("--stop-mb", type=float, default=0.0,
                    help=">0 时：贪心走到该字节预算即停止（用于生成'按预算优化'的分配）")
    ap.add_argument("--split", choices=["parity", "block"], default="parity",
                    help="parity=奇偶帧；block=时间块（标定用前 1/3，评测用后 1/3）")
    ap.add_argument("--calib-span", type=int, default=12, help="标定窗口覆盖的 val 帧数上限")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.eval_parity is None:
        a.eval_parity = a.index_parity
    if a.calib_parity is None:
        a.calib_parity = "even"
    if a.eval_cam is None:
        a.eval_cam = a.cam
    if a.calib_cam is None:
        a.calib_cam = a.eval_cam
    a.cam, a.index_parity = a.eval_cam, a.eval_parity   # 旧代码路径沿用评测设置

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    from PIL import Image
    import m1_eval as M

    raw = json.load(open(a.rates))
    rates = {k.replace(".params.", "."): v for k, v in raw["components"].items()}
    fixed = raw.get("fixed_bytes", 0)

    dev = torch.device("cuda:0")
    ddir = osp.abspath(a.data_dir)
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(a.ckpt, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/pf_{os.getpid()}")
    model = t.model

    # 原始 float 值 + 每档位宽的 (dequant, bytes) 缓存
    orig = {}
    for c in COMPS:
        g, n = c.split(".")
        if g == "fg" and n == "motion_coefs":
            orig[c] = model.fg.params["motion_coefs"].data.clone()
        elif g in ("fg", "bg") and n in getattr(model, g).params:
            orig[c] = getattr(model, g).params[n].data.clone()
    bases = {}
    for n in ["rots", "transls"]:
        if n in model.motion_bases.params:
            bases[f"bases.{n}"] = model.motion_bases.params[n].data.clone()

    # ---- 真实位平面包（P0-1）：基础包 + 增量升级包，字节按实际序列化长度计 ----
    from packets import Plane, truncate, dequant
    planes = {c: Plane(c, x.detach().float().cpu().numpy())
              for c, x in list(orig.items()) + list(bases.items())}

    def _nbytes(c, b):
        """该张量到 b 位的累计包字节（基础包 + 逐级升级包）。"""
        pl = planes[c]
        n = len(pl.base_packet(LEVELS[0]))
        for i in range(1, LEVELS.index(b) + 1):
            n += len(pl.refine_packet(LEVELS[i - 1], LEVELS[i]))
        return n

    def _delta_bytes(c, b_from, b_to):
        """从 b_from 升到 b_to 实际要发的包字节。"""
        pl = planes[c]
        if b_from == b_to:
            return 0
        n, lo = 0, b_from
        if lo < LEVELS[0]:
            n += len(pl.base_packet(LEVELS[0]))
            lo = LEVELS[0]
        for i in range(LEVELS.index(lo), LEVELS.index(b_to)):
            n += len(pl.refine_packet(LEVELS[i], LEVELS[i + 1]))
        return n

    # G3：基础层字节 = 相机 blob + 每个张量真实发送的**基础包**（bases 用 16-bit 基础包）
    base_packets = ([planes[c].base_packet(LEVELS[0]) for c in COMPS]
                    + [planes[c].base_packet(LEVELS[-1]) for c in bases])
    base_total = fixed + sum(len(k) for k in base_packets)

    cache = {}
    for c, pl in planes.items():
        for b in LEVELS:
            dq = torch.from_numpy(dequant(truncate(pl.q, b), pl.scale).reshape(pl.shape)).to(dev)
            cache[(c, b)] = (dq, _nbytes(c, b))

    # 评测集
    if a.dataset == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset
        snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                         map_location="cpu", weights_only=False)
        val = iPhoneDataset(data_dir=ddir, factor=a.factor, split="val",
                            scene_norm_dict=snd, load_from_cache=True)
        names = val.frame_names
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset
        snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                         map_location="cpu", weights_only=False)
        val = NvidiaDataset(data_dir=ddir, factor=a.factor, split="val",
                            scene_norm_dict=snd, load_from_cache=True)
        names = val.frame_names

    cams = [int(x) for x in a.eval_cams.split(",")] if a.eval_cams else None
    if cams is None and a.dataset == "nvidia":
        cams = [1, 2, 3]          # P0-4：训练相机 0 不进入评测
    if cams and a.dataset == "nvidia":
        assert 0 not in cams, "训练相机 0 不得进入 held-out evaluation"

    # ---- 时间块划分：按 time_id 切，不按全局数组下标（NVIDIA 数据按相机成段排列）----
    _cams_all = [int(nm.split("_")[0]) for nm in val.frame_names]
    _tids_all = [int(t) for t in val.time_ids.tolist()]
    _uniq_times = sorted(set(_tids_all))
    _front_t = set(_uniq_times[:len(_uniq_times) // 3])
    _back_t = set(_uniq_times[-len(_uniq_times) // 3:])
    assert _front_t.isdisjoint(_back_t), "时间块前/后 1/3 有重叠"

    def _frame_ids(parity, cam, cam_list=None, block=None):
        """parity 划分用帧号奇偶；block 划分用 time_id 的前/后 1/3（与相机索引无关）。"""
        out = []
        n = len(val)
        for i in range(n):
            fid_cam = int(val.frame_names[i].split("_")[0])
            if cam_list is not None and fid_cam not in cam_list:
                continue
            if cam >= 0 and fid_cam != cam:
                continue
            if block == "calib" and _tids_all[i] not in _front_t:
                continue
            if block == "eval" and _tids_all[i] not in _back_t:
                continue
            if parity != "all" and (i % 2 == 0) != (parity == "even"):
                continue
            out.append(i)
        return out

    _blk = (lambda which: which) if a.split == "block" else (lambda which: None)
    if a.split == "block":
        # 时间块：校准取前 1/3 的 camera-1 帧（固定取前 calib_span 个时间点），
        # 评测取后 1/3 每个相机等量帧；三相机场景要求每相机帧数一致。
        _calib_cam = 1 if a.dataset == "nvidia" else 0   # 单目场景只有 camera 0
        _cal_pool = _frame_ids("all", _calib_cam, block="calib")
        calib_ids = _cal_pool[:a.calib_span]
        _cams_eval = cams if cams else sorted({_cams_all[i] for i in _frame_ids("all", -1, block="eval")})
        _per = max(1, a.frames // len(_cams_eval))
        eval_ids, _per_actual = [], None
        for c_ in _cams_eval:
            pool = _frame_ids("all", c_, block="eval")
            take = min(_per, len(pool))          # 单目场景后 1/3 帧数本来就少，按实际数量报
            step = max(1, len(pool) // take)
            eval_ids.extend(pool[::step][:take])
            _per_actual = take if _per_actual is None else min(_per_actual, take)
        if _per_actual != _per:
            print(f"  ⚠ 时间块评测帧按实际可得数量取：每相机 {_per_actual} 帧（请求 {_per}）")
        assert set(_tids_all[i] for i in calib_ids).isdisjoint(
            set(_tids_all[i] for i in eval_ids)), "时间块下校准/评测的 time_id 重叠"
        if a.dataset == "nvidia":
            assert all(_cams_all[i] != 0 for i in eval_ids), "训练相机 0 不得进入评测"
        from collections import Counter as _C
        cnt = _C(_cams_all[i] for i in eval_ids)
        if len(_cams_eval) > 1:
            assert len(set(cnt.values())) == 1, f"时间块各相机帧数不一致: {dict(cnt)}"
        _cr = (f"{min(_tids_all[i] for i in calib_ids)}–{max(_tids_all[i] for i in calib_ids)}"
               if calib_ids else "n/a")
        _er = (f"{min(_tids_all[i] for i in eval_ids)}–{max(_tids_all[i] for i in eval_ids)}"
               if eval_ids else "n/a")
        print(f"  block split（按 time_id）: calib {len(calib_ids)} 帧(前1/3), "
              f"eval {len(eval_ids)} 帧(后1/3, per-cam {dict(cnt)}), time 范围 calib {_cr} / eval {_er}")
    else:
        calib_ids = _frame_ids(a.calib_parity, a.calib_cam, block=_blk("calib"))[:a.calib_span]
        eval_ids = []
        if cams:
            quota = max(1, a.frames // len(cams))
            for c_ in cams:
                got = 0
                for i in _frame_ids(a.eval_parity, a.eval_cam, [c_], block=_blk("eval")):
                    if i >= a.frame_start and got < quota:
                        eval_ids.append(i); got += 1
        else:
            for i in _frame_ids(a.eval_parity, a.eval_cam, block=_blk("eval")):
                if i >= a.frame_start and len(eval_ids) < a.frames:
                    eval_ids.append(i)
    if not a.leaky:
        assert not (set(calib_ids) & set(eval_ids)), (
            f"校准帧与评测帧重叠: {sorted(set(calib_ids) & set(eval_ids))[:8]}")
    per_cam = {}
    for i in eval_ids:
        c_ = int(val.frame_names[i].split("_")[0])
        per_cam[c_] = per_cam.get(c_, 0) + 1
    print(f"  frames: calib {len(calib_ids)} 个 (parity={a.calib_parity}, cam={a.calib_cam}), "
          f"eval {len(eval_ids)} 个 (parity={a.eval_parity}, cams={cams or 'all'}, per-cam {per_cam}), "
          f"交集={len(set(calib_ids) & set(eval_ids))}")
    for i in eval_ids:
        assert int(val.frame_names[i].split("_")[0]) == (a.eval_cam if a.eval_cam >= 0 else int(val.frame_names[i].split("_")[0]))

    def apply_bits(bits: dict):
        for c, x in orig.items():
            g, n = c.split(".")
            dq = cache[(c, bits[c])][0]
            if g == "fg" and n == "motion_coefs":
                model.fg.params["motion_coefs"].data = dq.to(model.fg.params["motion_coefs"].dtype)
            else:
                p = getattr(model, g).params[n]
                p.data = dq.to(p.dtype)
        for c, x in bases.items():
            n = c.split(".")[1]
            p = model.motion_bases.params[n]
            p.data = cache[(c, bits.get(c, 16))][0].to(p.dtype)

    N_RENDERS = [0]
    RENDERS_BY = {}
    CUR = ["misc"]
    LP = None
    if a.lpips:
        import m1_eval as _M
        LP = _M._lpips_fn(dev)

    SCORED = {"ids": [], "n_full": 0, "n_fg": 0, "per_frame": []}

    def eval_prefix(K: int):
        ps, pfg, pl = [], [], []
        SCORED["per_frame"] = []
        ids = eval_ids[:K] if K > 0 else eval_ids
        with torch.no_grad():
            for i in ids:
                fn = val.frame_names[i]
                w2c = val.w2cs[i:i + 1].to(dev)
                Kk = val.Ks[i:i + 1].to(dev)
                tid = int(val.time_ids[i].item())
                H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                out = model.render(tid, w2c, Kk, (Wd, H))
                pred = out["img"][0].detach().float().clamp(0, 1)
                gt = val.imgs[i].to(dev).float()[..., :3]
                mp = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
                if not osp.exists(mp):
                    mp = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
                if not osp.exists(mp):
                    raise SystemExit(f"缺 co-visibility mask: {fn}（P0-2 不允许跳过）")
                m = np.array(Image.open(mp).convert("L"))
                if (m.shape[0], m.shape[1]) != (H, Wd):
                    m = np.array(Image.open(mp).convert("L").resize((Wd, H), Image.NEAREST))
                vis = (torch.from_numpy(m).to(dev).float() / 255.0 > 0.5).float()
                if float(vis.sum()) < 1:
                    continue
                ps.append(M._psnr(pred, gt, vis))
                SCORED["n_full"] += 1
                fgp = osp.join(ddir, f"flow3d_preprocessed/track_anything/{a.factor}x", fn + ".png")
                if not osp.exists(fgp):
                    raise SystemExit(f"缺 dynamic mask: {fn}（P0-2 不允许跳过）")
                fa = np.array(Image.open(fgp).convert("L"))
                if (fa.shape[0], fa.shape[1]) != (H, Wd):
                    fa = np.array(Image.open(fgp).convert("L").resize((Wd, H), Image.NEAREST))
                fgm = vis * torch.from_numpy((fa > 127).astype(np.float32)).to(dev)
                if float(fgm.sum()) < 1:
                    raise SystemExit(f"dynamic mask 为空: {fn}（P0-2）")
                v_fg = M._psnr(pred, gt, fgm)
                pfg.append(v_fg)
                SCORED["n_fg"] += 1
                SCORED["per_frame"].append({"frame_id": i,
                                            "camera": int(val.frame_names[i].split("_")[0]),
                                            "full": round(float(ps[-1]), 4),
                                            "fg": round(float(v_fg), 4)})
                if LP is not None:
                    pl.append(M._lp(LP, pred, gt, fgm))
        return (float(np.mean(ps)) if ps else float("nan"),
                float(np.mean(pfg)) if pfg else float("nan"),
                float(np.mean(pl)) if pl else float("nan"))

    def expand(seq):
        """把任意顺序展开成"每包一步"：跨多档时逐档插入中间包（P0-1）。"""
        cur = dict((c, LEVELS[0]) for c in COMPS)
        out = []
        for bits in seq:
            for c in COMPS:
                while bits[c] > cur[c]:
                    nxt = LEVELS[LEVELS.index(cur[c]) + 1]
                    cur = dict(cur); cur[c] = nxt
                    out.append(dict(cur))
        return out

    STALE = {}
    PROBE_SCORES = {}
    STREAM_CHECK = {}

    if a.ref_frames_only:
        # 只渲染评测帧的 16-bit 参考，记录逐帧值（含相机与帧号）。不参与任何评分，只供尾部指标复算。
        apply_bits(dict((c, LEVELS[-1]) for c in COMPS))
        p16, p16fg, _ = eval_prefix(a.frames)
        json.dump({"scene": osp.basename(ddir), "ref_psnr": round(float(p16), 4),
                   "ref_fg": round(float(p16fg), 4),
                   "tol": a.tol, "frames": a.frames,
                   "per_frame": SCORED["per_frame"][:]},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}（{len(SCORED['per_frame'])} 帧 16-bit 参考）")
        return 0

    # 顺序定义
    orders = {}
    if a.only_direct:
        orders = {}
    if not a.only_direct:
        # 逐包阶梯：位深按 4→6→8→12→16 推进，每个 stage 只升一个组，
        # 与 round-robin 使用完全相同的物理包序列（G1：不再按"整轮"粗粒度评测）
        uni_seq = []
        cur_u = dict((c, LEVELS[0]) for c in COMPS)
        for lvl in LEVELS[1:]:
            for c in COMPS:
                cur_u = dict(cur_u); cur_u[c] = lvl
                uni_seq.append(dict(cur_u))
        orders["uniform"] = uni_seq
    dyn = [dict((c, 4) for c in COMPS)]
    step1 = dict((c, 4) for c in COMPS); step1.update({c: 16 for c in COMPS if c.startswith("fg.")})
    dyn.append(step1)
    for b in [6, 8, 12, 16]:
        s = dict(step1); s.update({c: b for c in COMPS if c.startswith("bg.")})
        dyn.append(s)
    orders["dynfirst"] = expand([dict(s) for s in dyn])

    # 轮转基线：无任何打分，按位宽档位逐轮把所有分量升一档（粒度与 direct 相同）
    rr = [dict((c, 4) for c in COMPS)]
    seq_rr = [("all4", dict((c, 4) for c in COMPS))]
    cur_rr = dict((c, 4) for c in COMPS)
    for lvl in [6, 8, 12, 16]:
        for c in COMPS:
            cur_rr = dict(cur_rr); cur_rr[c] = lvl
            seq_rr.append((c, dict(cur_rr)))
    orders["roundrobin"] = [dict(t[1]) for t in seq_rr]

    # 位置优先基线：所有分量 4 bit 起，先把 means（fg 再 bg）升到 16，再升其余属性
    pos = [dict((c, 4) for c in COMPS)]
    order_attr = ["fg.means", "bg.means", "fg.scales", "bg.scales", "fg.motion_coefs",
                  "fg.colors", "bg.colors", "fg.opacities", "bg.opacities", "fg.quats", "bg.quats"]
    cur = dict((c, 4) for c in COMPS)
    for c in order_attr:
        for b in [8, 12, 16]:
            cur = dict(cur); cur[c] = b
            pos.append(dict(cur))
    orders["posfirst"] = expand([dict(s_) for s_ in pos])

    sens = {k.replace(".params.", "."): v for k, v in json.load(open(a.sens)).items()} if a.sens else {}
    if sens:
        bits = {c: 4 for c in COMPS}
        seq = [("all4", dict(bits))]
        # 贪心：每次把某分量升一档，按 dB/MB 最大
        cur = dict(bits); cum = sum(cache[(c, 4)][1] for c in COMPS)
        trace = []
        while True:
            best = None
            for c in COMPS:
                b = cur[c]
                nxt = [x for x in LEVELS if x > b]
                if not nxt:
                    continue
                nb = min(nxt)
                add = _delta_bytes(c, b, nb)
                if add <= 0:
                    continue
                gain = sens.get(c, 0.0) * (nb - b) / 12.0
                sc = gain / (add / 1e6)
                if best is None or sc > best[0]:
                    best = (sc, c, nb, add, gain)
            if best is None:
                break
            _, c, nb, add, gain = best
            cur[c] = nb; cum += add
            trace.append({"comp": c, "bits": nb, "cum_bytes": cum, "gain": round(gain, 3)})
            seq.append((c, dict(cur)))
        orders["sens"] = expand([dict(s) for _c, s in seq])

    # ---- 直接前缀目标贪心：在标定窗口（even）上，每步选"两口径缺口之和下降最多/字节"的包 ----
    if a.direct:
        thr_full = None
        calib_cache = {}
        calib_frames = {}   # 逐帧校准值，仅用于尾部风险评分
        def quality(bits_dict):
            """校准帧上的双区域质量 {full, fg}（校准/评测已分离）。"""
            key = tuple(sorted(bits_dict.items()))
            if key in calib_cache:
                return calib_cache[key]
            apply_bits(bits_dict)
            ps, pfg = [], []
            with torch.no_grad():
                for i in calib_ids:
                    fn = val.frame_names[i]
                    w2c = val.w2cs[i:i + 1].to(dev); Kk = val.Ks[i:i + 1].to(dev)
                    tid = int(val.time_ids[i].item())
                    H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
                    out = model.render(tid, w2c, Kk, (Wd, H))
                    N_RENDERS[0] += 1
                    RENDERS_BY[CUR[0]] = RENDERS_BY.get(CUR[0], 0) + 1
                    pred = out["img"][0].detach().float().clamp(0, 1)
                    gt = val.imgs[i].to(dev).float()[..., :3]
                    mp = osp.join(ddir, f"flow3d_preprocessed/covisible/{a.factor}x/val", fn + ".png")
                    if not osp.exists(mp):
                        mp = osp.join(ddir, f"covisible/{a.factor}x/val", fn + ".png")
                    if not osp.exists(mp):
                        raise SystemExit(f"缺 co-visibility mask（标定）: {fn}")
                    else:
                        m_ = np.array(Image.open(mp).convert("L"))
                        if (m_.shape[0], m_.shape[1]) != (H, Wd):
                            m_ = np.array(Image.open(mp).convert("L").resize((Wd, H), Image.NEAREST))
                        vis = (torch.from_numpy(m_).to(dev).float() / 255.0 > 0.5).float()
                    ps.append(M._psnr(pred, gt, vis))
                    fgp = osp.join(ddir, f"flow3d_preprocessed/track_anything/{a.factor}x", fn + ".png")
                    if not osp.exists(fgp):
                        raise SystemExit(f"缺 dynamic mask（标定）: {fn}")
                    fa = np.array(Image.open(fgp).convert("L"))
                    if (fa.shape[0], fa.shape[1]) != (H, Wd):
                        fa = np.array(Image.open(fgp).convert("L").resize((Wd, H), Image.NEAREST))
                    fgm = vis * torch.from_numpy((fa > 127).astype(np.float32)).to(dev)
                    if float(fgm.sum()) < 1:
                        raise SystemExit(f"dynamic mask 为空（标定）: {fn}")
                    pfg.append(M._psnr(pred, gt, fgm))
            out_ = {"full": float(np.mean(ps)) if ps else float("nan"),
                    "fg": float(np.mean(pfg)) if pfg else float("nan")}
            calib_cache[key] = out_
            calib_frames[key] = (list(ps), list(pfg))
            return out_

        ref_bits = dict((c, 16) for c in COMPS)
        ref_calib = quality(ref_bits)
        REF_FRAMES = (calib_frames[tuple(sorted(ref_bits.items()))][0],
                      calib_frames[tuple(sorted(ref_bits.items()))][1])
        if a.direct_ref <= 0:
            a.direct_ref = ref_calib["full"]
            print(f"  direct: 校准窗口 16-bit 参考 full={ref_calib['full']:.3f} dB, "
                  f"fg={ref_calib['fg']:.3f} dB")
        thr_full = a.direct_ref - a.tol
        thr_fg = ref_calib["fg"] - a.tol

        def _cvar_worst(e, al):
            """经验 CVaR_alpha（分数权重）：最差 (1-alpha)·n 帧的加权均值。

            n=12、alpha=0.9 时权重和 = 1.2 → 1 个最差帧全权 + 0.2 个次差帧，再除以 1.2。
            样本很少时也不至于退化成"只看 1 帧"。
            """
            n = len(e)
            mass = (1.0 - al) * n
            if mass <= 0:
                return 0.0
            srt = sorted(e, reverse=True)
            acc, used = 0.0, 0.0
            for v in srt:
                take = min(1.0, mass - used)
                if take <= 0:
                    break
                acc += take * v
                used += take
            return acc / max(used, 1e-12)

        def tail_risk(bits_dict):
            """逐帧尾部风险 R_alpha(P)=sum_r w_r CVaR_alpha({[tau_{r,i}-Q_{r,i}]_+})。

            tau_{r,i} = 该帧 16-bit 参考 − tol（与"每帧达标"用同一个门槛，因此评分目标
            与报告口径一致）。只用校准帧；评测帧不参与任何评分。
            """
            q = quality(bits_dict)
            ps, pfg = calib_frames[tuple(sorted(bits_dict.items()))]
            ref_full, ref_fg = REF_FRAMES
            al = float(a.risk_alpha)
            tot = 0.0
            for vals, refs, w in ((ps, ref_full, 1.0), (pfg, ref_fg, a.lam)):
                if not vals or refs is None:
                    continue
                e = [max(0.0, r - a.tol - v) for v, r in zip(vals, refs)]
                tot += w * _cvar_worst(e, al)
            return tot

        def shortfall(bits_dict):
            """双区域缺口 S(P)=max(0,τf−Qf)+λ·max(0,τd−Qd)（P0-3）。"""
            q = quality(bits_dict)
            s_full = max(0.0, thr_full - q["full"]) if thr_full else 0.0
            s_fg = max(0.0, thr_fg - q["fg"]) if np.isfinite(thr_fg) else 0.0
            return s_full + a.lam * s_fg
        cur = dict((c, 4) for c in COMPS)
        seq = [("all4", dict(cur))]
        cur_p = shortfall(cur)
        cur_bytes = base_total
        score_cache = {}
        prev_cache = {}
        frozen_flag = [False]
        low_rate = [False]
        tau_hits = [0]
        psnr_hist = [(0, None)]
        stale_trace = []   # 每次刷新时：{step, kendall_tau_vs_prev}
        for step in range(a.direct_steps):
            if a.policy == "adaptive":
                k = a.refresh if low_rate[0] else 1
                if k < 1:
                    k = 1
                refresh_now = (step % k == 0)
            elif a.tau_freeze > 0:
                refresh_now = not frozen_flag[0]
                k = 1
            elif a.refresh_early and step < a.refresh_early:
                k = 1
                refresh_now = True
            else:
                k = max(1, a.refresh) if a.refresh > 0 else 10**9
                refresh_now = (step % k == 0) if k > 1 else True
            best = None
            sec_cache = {}
            for c in COMPS:
                nxt = [x for x in LEVELS if x > cur[c]]
                if not nxt:
                    continue
                nb = min(nxt)
                cand = dict(cur); cand[c] = nb
                add = _delta_bytes(c, cur[c], nb)
                if add <= 0:
                    continue
                if refresh_now:
                    score_after = shortfall(cand)
                    qa = quality(cand)
                    qb = quality(cur)
                    if a.adaptive_score == "tail_cvar":
                        # 尾部风险下降量 / 字节：只看最差那一小撮校准帧的缺口
                        primary = max(0.0, tail_risk(cur) - tail_risk(cand))
                    elif a.adaptive_score == "full_gain":
                        # 只看全图边际画质增益（标准的逐包 PSNR 贪心）
                        primary = max(0.0, qa["full"] - qb["full"])
                    elif a.adaptive_score == "joint_gain":
                        # 双区域边际增益之和，但**不做缺口截断**
                        primary = max(0.0, (qa["full"] + qa["fg"]) - (qb["full"] + qb["fg"]))
                    else:
                        # 论文式(3)：质量缺口减少量的正部（与旧实现逐位一致）
                        primary = max(0.0, cur_p - score_after)
                    score_cache[(c, nb)] = primary / (add / 1e6)
                    gpb = score_cache[(c, nb)]
                    sec_cache[(c, nb)] = ((qa["full"] + a.lam * qa["fg"])
                                          - (qb["full"] + a.lam * qb["fg"])) / (add / 1e6)
                    score_after = score_after if refresh_now else cur_p
                else:
                    gpb = score_cache.get((c, nb), -1e9)
                    score_after = cur_p
                key = (gpb, sec_cache.get((c, nb), -1e18))
                if best is None or key > best[0]:
                    best = (key, c, nb, add, score_after)
            if best is None:
                break
            if refresh_now:
                from scipy.stats import kendalltau as _kt
                keys = [k for k in score_cache if k in prev_cache]
                if len(keys) >= 4:
                    try:
                        tau = _kt([prev_cache[k] for k in keys], [score_cache[k] for k in keys]).statistic
                    except Exception:
                        tau = float("nan")
                else:
                    tau = float("nan")
                stale_trace.append({"step": step, "tau": None if tau != tau else round(float(tau), 4)})
                if a.tau_freeze > 0 and tau == tau and tau >= a.tau_freeze and step > 0:
                    frozen_flag[0] = True   # 排序已稳定 → 之后冻结
                if a.policy == "adaptive" and tau == tau and step > 0:
                    if tau >= a.tau_hi:
                        tau_hits[0] += 1
                    else:
                        tau_hits[0] = 0
                    if (not low_rate[0]) and tau_hits[0] >= 2:
                        low_rate[0] = True      # 排序饱和 → 降频
                    elif low_rate[0]:
                        # 停滞检测：与上一个刷新点相比缺口是否还在下降
                        prev = [x for x in psnr_hist if x[1] is not None]
                        if len(prev) >= 2 and (prev[-1][1] - prev[0][1]) < a.stall_eps:
                            low_rate[0] = False     # 停滞 → 恢复逐包
                            psnr_hist = [(step, cur_p)]
                psnr_hist.append((step, cur_p))
            prev_cache = dict(score_cache)
            _, c, nb, add, score_after = best
            cur[c] = nb
            cur_bytes += add
            # 非刷新步：只渲染被选中的那个状态（真实成本：1 次渲染/步）
            cur_p = score_after if refresh_now else shortfall(cur)
            seq.append((c, dict(cur)))
            print(f"  direct step{step:02d}: {c}->{nb} bit  calib_psnr={cur_p:.3f}")
            if a.stop_mb > 0 and cur_bytes / 1e6 >= a.stop_mb:
                break
            # P0-3：停止条件只看"双区域分别达标"，与 λ 无关
            if not a.direct_full:
                _q = quality(cur)
                if _q["full"] >= thr_full and _q["fg"] >= thr_fg:
                    break
        orders["direct"] = [dict(s_) for _c, s_ in seq]
        STALE["direct"] = list(stale_trace)

        # ---- oneshot：只在初始（全 4-bit）状态上给每个候选包打分一次，然后固定顺序 ----
        base_state = dict((c, 4) for c in COMPS)
        p0 = shortfall(base_state)
        scored = []
        for c in COMPS:
            for nb in [x for x in LEVELS if x > 4]:
                cand = dict(base_state); cand[c] = nb
                add = _delta_bytes(c, 4, nb)
                if add <= 0:
                    continue
                after = shortfall(cand)
                gain = max(0.0, p0 - after)
                scored.append((gain / (add / 1e6), c, nb))
        scored.sort(reverse=True)
        cur2 = dict(base_state); seq2 = [("all4", dict(cur2))]
        applied = set()
        for gpb, c, nb in scored:
            if (c, nb) in applied:
                continue
            # 只按档位顺序推进：要求该分量当前位宽低于 nb
            if cur2[c] >= nb:
                continue
            cur2 = dict(cur2); cur2[c] = nb
            applied.add((c, nb))
            seq2.append((c, dict(cur2)))
        orders["oneshot"] = expand([dict(s_) for _c, s_ in seq2])

        # ---- mpprefix：在多个代表性前缀上给每个候选包打分并取均值，然后冻结顺序 ----
        probe_states = [dict((c, 4) for c in COMPS)]
        st = dict((c, 4) for c in COMPS)
        k_ = 0
        for lvl in [6, 8, 12]:
            for c in COMPS:
                st = dict(st)
                nxt = [x for x in LEVELS if x > st[c]]
                st[c] = min(nxt) if nxt else 16
                k_ += 1
                if k_ % a.probe_every == 0 and len(probe_states) < a.probe_n:
                    probe_states.append(dict(st))
        acc = {}
        for ps in probe_states:
            p_s = shortfall(ps)
            for c in COMPS:
                for nb in [x for x in LEVELS if x > ps[c]]:
                    cand = dict(ps); cand[c] = nb
                    add = _delta_bytes(c, ps[c], nb)
                    if add <= 0:
                        continue
                    after = shortfall(cand)
                    gain = max(0.0, p_s - after)
                    acc.setdefault((c, nb), []).append(gain / (add / 1e6))
        PROBE_SCORES.update({f"{c}|{nb}": [round(x, 4) for x in v] for (c, nb), v in acc.items()})
        order_mp = sorted(((sum(v) / len(v), c, nb) for (c, nb), v in acc.items()), reverse=True)
        cur3 = dict((c, 4) for c in COMPS); seq3 = [("all4", dict(cur3))]; applied3 = set()
        for gpb, c, nb in order_mp:
            if (c, nb) in applied3 or cur3[c] >= nb:
                continue
            cur3 = dict(cur3); cur3[c] = nb; applied3.add((c, nb))
            seq3.append((c, dict(cur3)))
        orders["mpprefix"] = expand([dict(t[1]) for t in seq3])
        apply_bits(dict((c, 4) for c in COMPS))

    results = []
    t_start = time.time()
    t0 = time.time()
    for name, stages in orders.items():
        CUR[0] = name
        t_order = time.time()
        # 冷客户端：基础层包 + 逐个真实序列化的升级包，cum 就是码流字节偏移
        stream = [planes[c].base_packet(LEVELS[0]) for c in COMPS]
        stream += [planes[c].base_packet(LEVELS[-1]) for c in bases]
        cum = fixed + sum(len(k) for k in stream)
        assert cum == base_total, (cum, base_total)
        prev_bits = dict((c, LEVELS[0]) for c in COMPS)
        order_offsets = [cum]
        for idx, bits in enumerate(stages):
            new_pk = []
            for c in COMPS:
                b0, b1 = prev_bits[c], int(bits[c])
                if b1 <= b0:
                    continue
                lo = b0 if b0 >= LEVELS[0] else LEVELS[0]
                for i in range(LEVELS.index(lo), LEVELS.index(b1)):
                    new_pk.append(planes[c].refine_packet(LEVELS[i], LEVELS[i + 1]))
            stream += new_pk
            cum += sum(len(k) for k in new_pk)
            assert cum == fixed + sum(len(k) for k in stream)   # 逐级字节偏移自证
            order_offsets.append(cum)
            prev_bits = dict((c, int(bits[c])) for c in COMPS)
            t_dec0 = time.time()
            apply_bits(bits)
            t_dec = time.time() - t_dec0
            t_r0 = time.time()
            p, pfg, plp = eval_prefix(a.frames)
            t_ren = time.time() - t_r0
            results.append({"order": name, "stage": idx, "bytes": int(cum),
                            "MB": round(cum / 1e6, 4), "psnr_first": round(p, 4),
                            "psnr_first_fg": round(pfg, 4),
                            "lpips_first_fg": round(plp, 4),
                            "t_decode_s": round(t_dec, 4), "t_render_s": round(t_ren, 4),
                            "per_frame": SCORED["per_frame"][:], "bits": dict(bits)})
            print(f"{name:9s} stage{idx:02d}  {cum/1e6:8.3f} MB  first{a.frames}={p:7.3f}/{pfg:7.3f} dB (full/fg)")
        # 冷客户端自证：最后一个 stage 的 cum 必须等于真实码流拼接后的字节数
        STREAM_CHECK[name] = {"last_stage_bytes": int(cum),
                              "serialized_stream_bytes": int(fixed + sum(len(k) for k in stream))}
        assert STREAM_CHECK[name]["last_stage_bytes"] == STREAM_CHECK[name]["serialized_stream_bytes"], STREAM_CHECK[name]
        print(f"--- {name}: {time.time()-t0:.1f}s elapsed, renders={RENDERS_BY.get(name,0)}, "
              f"wall={time.time()-t_order:.1f}s, stream={len(stream)} pkts/{cum/1e6:.2f} MB")

    # 参考：16 bit 全量（only_direct 模式也要有参考，否则达标判定全是 NaN）
    if a.only_direct:
        apply_bits(dict((c, LEVELS[-1]) for c in COMPS))
        p16, p16fg, _ = eval_prefix(a.frames)
        ref_psnr, ref_fg = float(p16), float(p16fg)
    else:
        # 参考 = uniform 顺序的最后一个 stage（逐包展开后 stage 数不再是 len(LEVELS)）
        ref = [r for r in results if r["order"] == "uniform"][-1]
        ref_psnr = ref["psnr_first"]
        ref_fg = ref.get("psnr_first_fg", float("nan"))
    # P1-3：first-playable 是"所选 K 帧的平均"达标；附逐帧分布
    import statistics as _stat

    def _stats(vals, thr):
        if not vals:
            return {}
        v = sorted(vals)
        return {"mean": round(float(_stat.mean(v)), 3),
                "std": round(float(_stat.pstdev(v)), 3),
                "p10": round(float(v[max(0, int(0.1 * len(v)) - 1)]), 3),
                "frac_below": round(sum(1 for x in v if x < thr) / len(v), 3)}

    summary = {}
    for name in orders:
        rs = [r for r in results if r["order"] == name]
        def ok(r):
            a1 = r["psnr_first"] >= ref_psnr - a.tol
            a2 = (not np.isfinite(ref_fg)) or (r.get("psnr_first_fg", -1e9) >= ref_fg - a.tol)
            return a1 and a2
        need = next((r for r in rs if ok(r)), None)
        _pf = need.get("per_frame", []) if need else []
        _full = _stats([x["full"] for x in _pf], ref_psnr - a.tol)
        _fgst = _stats([x["fg"] for x in _pf], (ref_fg - a.tol) if np.isfinite(ref_fg) else -1e9)
        summary[name] = {"frame_stats": {"full": _full, "fg": _fgst},
                         "ref_psnr": ref_psnr, "ref_fg": None if not np.isfinite(ref_fg) else round(ref_fg, 3), "bytes_to_ref_minus_0.5": need["bytes"] if need else None,
                         "MB": need["MB"] if need else None,
                         "t_transfer_s_at_2Mbps": round(need["bytes"] * 8 / 2e6, 3) if need else None,
                         "t_decode_s": need["t_decode_s"] if need else None,
                         "t_render_s": need["t_render_s"] if need else None,
                         "t_startup_s": (round(need["bytes"] * 8 / 2e6 + need["t_decode_s"] + need["t_render_s"], 3)
                                         if need else None)}
        print(f"{name:9s} → first-{a.frames} 双口径(-0.5dB) 需 "
              f"{need['MB'] if need else float('nan'):.3f} MB")
    # 码流自证：从包流严格解码，必须与缓存张量逐元素相等（P0-1）
    from packets import decode_stream
    codec_ok = {}
    for c, pl in planes.items():
        pkts = [pl.base_packet(LEVELS[0])] + [pl.refine_packet(LEVELS[i], LEVELS[i + 1])
                                               for i in range(len(LEVELS) - 1)]
        q16, bits_, scale_, _, shape_ = decode_stream(pkts, LEVELS[-1])
        dq = dequant(q16, scale_).reshape(shape_)
        codec_ok[c] = bool(np.array_equal(dq.ravel(), cache[(c, LEVELS[-1])][0].detach().cpu().numpy().ravel()))
    assert all(codec_ok.values()), f"码流重建不一致: {[k for k, v in codec_ok.items() if not v]}"
    print(f"  codec: {len(codec_ok)} 个张量从包流重建一致（bit-exact）")

    json.dump({"results": results, "summary": summary, "ref_psnr": ref_psnr,
               "K": a.frames, "n_renders": N_RENDERS[0],
               "renders_by_order": dict(RENDERS_BY), "staleness": STALE, "probe_scores": PROBE_SCORES,
               "base_bytes": int(base_total), "camera_bytes": int(fixed),
               "n_base_packets": len(base_packets), "calib_ids": calib_ids,
               "eval_ids": eval_ids, "scored": SCORED,
               "split_mode": ("timestamp_block" if a.split == "block" else "parity"),
               "calib_parity": (None if a.split == "block" else a.calib_parity),
               "calib_cam": (None if a.split == "block" else a.calib_cam),
               "eval_parity": (None if a.split == "block" else a.eval_parity),
               "eval_cam": (None if a.split == "block" else a.eval_cam),
               "effective_calib_camera": (
                   sorted({_cams_all[i] for i in calib_ids})[0]
                   if (a.split == "block" and calib_ids) else None
               ),
               "calib_time_ids": (sorted({_tids_all[i] for i in calib_ids})
                                  if a.split == "block" else None),
               "eval_time_ids": (sorted({_tids_all[i] for i in eval_ids})
                                 if a.split == "block" else None),
               "calib_frame_manifest": ([{"index": int(i), "frame_id": int(val.frame_names[i].split("_")[-1]),
                                          "camera": _cams_all[i], "time_id": _tids_all[i]}
                                         for i in calib_ids] if a.split == "block" else None),
               "eval_frame_manifest": ([{"index": int(i), "frame_id": int(val.frame_names[i].split("_")[-1]),
                                         "camera": _cams_all[i], "time_id": _tids_all[i]}
                                        for i in eval_ids] if a.split == "block" else None),
               "lam": a.lam, "tol": a.tol,
               "adaptive_score": a.adaptive_score,
               "risk_alpha": a.risk_alpha,
               "packet_model": "16-bit grid, bit-plane refinement packets, zlib",
               "stream_check": STREAM_CHECK,
               "codec_bit_exact": codec_ok,
               "t_total_s": round(time.time() - t_start, 2)}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
