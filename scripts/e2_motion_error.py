#!/usr/bin/env python3
"""e2_motion_error.py — 时间轴上「运动能量 vs 渲染误差」的第一手测量。

目的：验证本窗口的核心科学假设——
  4D 单目重建的误差在时间上强非均匀，且与图像空间运动能量相关（而非与可见像素数相关）。
若成立，则「按运动能量分配时间维容量」有机制依据；若不成立，则这条主干要改。

数据：NeRF-DS backpack（180 帧，720x960，fg mask 7.2% 像素）
度量：
  m_t  = 相邻帧 Farneback 光流在 fg mask 上的平均模长（像素/帧），窗口 ±2 帧取均值
  err_t= EVAL json 中每个 val 帧的 fg_psnr / full_psnr
  vis_t= 该帧可见前景像素数（混淆变量，必须控）
输出：work/e2.json + 控制台 Spearman/Pearson 表
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

D = f"{_DATA}/scenes/backpack"
OUT = f"{_ROOT}/work/e2.json"
SCALE = 0.5


def load_motion() -> tuple[list[int], np.ndarray]:
    rgbs = sorted(glob.glob(f"{D}/rgb/1x/*.png"))
    masks = sorted(glob.glob(f"{D}/flow3d_preprocessed/track_anything/1x/*.png"))
    assert len(rgbs) == len(masks), (len(rgbs), len(masks))
    prev_gray = None
    prev_mask = None
    mags = []
    for i, (rp, mp) in enumerate(zip(rgbs, masks)):
        img = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        if prev_gray is None:
            mags.append(np.nan)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, img, None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            mag = np.linalg.norm(flow, axis=-1)
            m = msk | prev_mask
            mags.append(float(mag[m].mean()) if m.sum() > 0 else 0.0)
        prev_gray, prev_mask = img, msk
    mags = np.array(mags, dtype=np.float64)
    mags[0] = mags[1]
    return list(range(len(rgbs))), mags


def main() -> int:
    n, mags = load_motion()
    ev = json.load(open(f"{_ROOT}/work/e1_anat_repro.json"))
    rows = ev["per_frame"]
    fidx, psnr, vis = [], [], []
    for r in rows:
        f = int(r["frame"].split("_")[-1])
        fidx.append(f)
        psnr.append(r["fg_psnr"])
        vis.append(r["n_fg_vis"])
    fidx = np.array(fidx)
    psnr = np.array(psnr, dtype=np.float64)
    vis = np.array(vis, dtype=np.float64)

    # 窗口 ±2 帧的运动能量（val 帧每 5 帧采一次，故窗口覆盖 1/2 采样间隔）
    m_win = np.array([np.nanmean(mags[max(0, f - 2):f + 3]) for f in fidx])

    def rep(name, x, y):
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        print(f"{name:34s} pearson r={r_p:+.3f} (p={p_p:.2e})  spearman ρ={r_s:+.3f} (p={p_s:.2e})")
        return {"pearson_r": round(float(r_p), 4), "pearson_p": float(p_p),
                "spearman_rho": round(float(r_s), 4), "spearman_p": float(p_s)}

    print(f"n_val={len(fidx)}  motion range [{m_win.min():.3f}, {m_win.max():.3f}] px/frame")
    print(f"fg_psnr range [{psnr.min():.2f}, {psnr.max():.2f}] dB   spread={psnr.max()-psnr.min():.2f} dB")
    out = {"n_val": int(len(fidx)), "scene": "backpack",
           "fg_psnr_min": round(float(psnr.min()), 3), "fg_psnr_max": round(float(psnr.max()), 3),
           "fg_psnr_mean": round(float(psnr.mean()), 3),
           "motion_mean": round(float(m_win.mean()), 4)}
    out["corr_motion_vs_psnr"] = rep("motion vs fg_psnr", m_win, psnr)
    out["corr_vis_vs_psnr"] = rep("visibility vs fg_psnr", vis, psnr)
    out["corr_motion_vs_vis"] = rep("motion vs visibility (confound)", m_win, vis)

    # 偏相关：在控制可见像素数后，运动能量与误差是否仍相关
    A = np.c_[np.ones_like(vis), vis]
    res_m = m_win - A @ np.linalg.lstsq(A, m_win, rcond=None)[0]
    res_p = psnr - A @ np.linalg.lstsq(A, psnr, rcond=None)[0]
    r_pc, p_pc = pearsonr(res_m, res_p)
    print(f"{'partial(motion|visibility) vs psnr':34s} r={r_pc:+.3f} (p={p_pc:.2e})")
    out["partial_motion_given_vis"] = {"r": round(float(r_pc), 4), "p": float(p_pc)}

    out["per_frame"] = [
        {"frame": int(f), "motion": round(float(m), 4), "fg_psnr": round(float(p), 4),
         "n_fg_vis": float(v)}
        for f, m, p, v in zip(fidx, m_win, psnr, vis)
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
