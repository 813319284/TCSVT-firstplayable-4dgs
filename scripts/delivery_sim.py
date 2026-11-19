#!/usr/bin/env python3
"""delivery_sim.py — 真实码流的发送/接收回放（trace 驱动），测"测得的时间"而不是换算。

链路（与论文的字节账本完全一致）：
  冻结 ckpt → 量化平面 → 实际序列化包（包头 magic/length/zlib-9 全部计入）
  → 发送端按 policy 的包序、按 bandwidth trace 逐个到达
  → 接收端逐包**真实解码**（packets.decode_*，带前序位深依赖检查）
  → 到达首次达标字节后在 GPU 上渲染一帧，记录 "解码完成 + 首帧渲染" 的时刻

明确边界（写进日志与表）：
  · 传输是 **trace 驱动模拟**（恒速 1/2/5/10 Mbps + 固定种子的抖动 trace），不是真实 WAN 实测；
  · 应用层字节 = 论文的 B(P)；不额外计 TCP/UDP 头（wire_bytes 另列，恒等于应用层）；
  · 渲染只测"达标前缀下渲染一帧"的耗时，不预测播放端 FPS。

用法：
  python3 scripts/delivery_sim.py --scene truck --policy ours --trace 2mbps [--gpu 0]
产出：work/final/deliv_<scene>_<policy>_<trace>.json
"""
import argparse
import json
import os
import os.path as osp
import hashlib
import random
import sys
import time

import numpy as np
import torch
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

TRUST = f"{_DATA}"
sys.path.insert(0, f"{TRUST}/METHOD01_CAPE4D/scripts")
sys.path.insert(0, f"{TRUST}/JOURNAL_DATA/_src/shape-of-motion-main")
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

LEVELS = [4, 6, 8, 12, 16]
COMPS = ["fg.means", "fg.scales", "fg.colors", "fg.quats", "fg.opacities", "fg.motion_coefs",
         "bg.means", "bg.scales", "bg.colors", "bg.quats", "bg.opacities"]
POLICY_RUN = {
    "ours": "p0", "full_gain": "adaptive_full_gain",
    "joint_gain": "adaptive_joint_gain", "tail_cvar": "risk_tail_cvar",
    "best_fixed": "p0",
}
TRACES = {
    "1mbps": dict(kind="cbr", mbps=1.0),
    "2mbps": dict(kind="cbr", mbps=2.0),
    "5mbps": dict(kind="cbr", mbps=5.0),
    "10mbps": dict(kind="cbr", mbps=10.0),
    "jitter": dict(kind="jitter", mbps=2.0, seed=7, spread=0.6),
}


SEG_S = 0.2       # 抖动轨迹的分段长度（秒）


def trace_rate(trace, t):
    """**按绝对时间索引**的链路速率（Mb/s）。与包大小、包序号、策略无关——
    所有策略看到的是同一条带宽轨迹，这才是公平的对照。"""
    if trace["kind"] == "cbr":
        return trace["mbps"]
    seg = int(t // SEG_S)
    digest = hashlib.blake2b(f"{trace['seed']}:{seg}".encode(), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    factor = 1.0 + trace["spread"] * (rng.random() * 2 - 1)     # ±spread
    return trace["mbps"] * factor


def transfer_time(trace, nbytes, t0=0.0):
    """把 nbytes 放进这条时间索引轨迹里，返回 (完成时刻, 本次耗时)。"""
    if trace["kind"] == "cbr":
        return t0 + 8.0 * nbytes / (trace["mbps"] * 1e6), 8.0 * nbytes / (trace["mbps"] * 1e6)
    t, remain = t0, float(nbytes)
    guard = 0
    while remain > 1e-9:
        seg = int(t // SEG_S)
        seg_end = (seg + 1) * SEG_S
        if seg_end <= t:                 # 浮点落在边界上时确保段有正长度
            seg_end = t + SEG_S
        rate = trace_rate(trace, t) / 8.0 * 1e6        # bytes/s
        if rate <= 0:
            rate = 1.0
        capacity = rate * (seg_end - t)
        if capacity <= 0:
            t = seg_end
            continue
        take = min(remain, capacity)
        t += take / rate
        remain -= take
        guard += 1
        if guard > 10_000_000:
            raise RuntimeError(f"transfer_time 未收敛: nbytes={nbytes} t={t} remain={remain}")
    return t, t - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--policy", default="ours", choices=list(POLICY_RUN))
    ap.add_argument("--trace", default="2mbps", choices=list(TRACES))
    ap.add_argument("--policies", default=None, help="all = 五种策略全跑")
    ap.add_argument("--traces", default=None, help="all = 五条 trace 全跑")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    root = f"{_ROOT}"
    sc = a.scene
    if a.traces is not None:
        a.traces = a.traces
    else:
        a.traces = a.trace
    if a.policies is None:
        a.policies = a.policy
    # 真实包流：基础包 + 按 order 的升级包（与论文账本同源）
    from packets import Plane, decode_base, apply_refine
    rates = json.load(open(f"{root}/work/rates_{sc}.json"))
    fixed_bytes = rates.get("fixed_bytes", 0)
    planes = {}
    for c, v in rates["components"].items():
        cc = c.replace(".params.", ".")
        planes[cc] = None
    # 平面要用真实权重，所以读 ckpt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from flow3d.configs import (LossesConfig, OptimizerConfig, SceneLRConfig,
                                FGLRConfig, BGLRConfig, MotionLRConfig,
                                CameraPoseLRConfig, CameraScalesLRConfig)
    from flow3d.trainer import Trainer
    ck = (f"{_DATA}/checkpoints/A_nat/ckpt_last.ckpt"
          if sc == "backpack" else f"{root}/work/{sc}_nat/ckpt_last.ckpt")
    dev = torch.device("cuda:0")
    lr = SceneLRConfig(FGLRConfig(), BGLRConfig(), MotionLRConfig(),
                       CameraPoseLRConfig(), CameraScalesLRConfig())
    t, _ = Trainer.init_from_checkpoint(ck, dev, False, lr, LossesConfig(),
                                        OptimizerConfig(), work_dir=f"/tmp/ds_{os.getpid()}")
    model = t.model
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

    def assemble(policy):
        run_json = f"{root}/work/final/{POLICY_RUN[policy]}_{sc}.json"
        d = json.load(open(run_json))
        order = "direct"
        if policy == "best_fixed":
            fixed = ["uniform", "roundrobin", "posfirst", "dynfirst", "oneshot", "mpprefix"]
            order = min(fixed, key=lambda k: d["summary"][k]["bytes_to_ref_minus_0.5"] or 1e18)
        b_target = d["summary"][order]["bytes_to_ref_minus_0.5"]
        stages = [r for r in d["results"] if r["order"] == order]
        hit = [r for r in stages if r["bytes"] >= b_target][0]
        packets = []
        for c in COMPS:
            packets.append((c, planes[c].base_packet(LEVELS[0]), "base", 0, LEVELS[0]))
        for c in bases:
            packets.append((c, planes[c].base_packet(LEVELS[-1]), "base", 0, LEVELS[-1]))
        cur = dict((c, LEVELS[0]) for c in COMPS)
        cur.update({c: LEVELS[-1] for c in bases})
        for st in stages[1:]:
            nb = st["bits"]
            for c in nb:
                if nb[c] > cur.get(c, 0):
                    packets.append((c, planes[c].refine_packet(cur[c], nb[c]), "refine", cur[c], nb[c]))
                    cur = dict(cur); cur[c] = nb[c]
            if st["bytes"] >= hit["bytes"]:
                break
        offs, cum = [], fixed_bytes
        for _n, pkt, _k, _p, _b in packets:
            cum += len(pkt)
            offs.append(cum)
        return dict(order=order, target=int(b_target), bits=hit["bits"], packets=packets,
                    offs=offs, camera_bytes=int(fixed_bytes))

    def replay(case, trace_name):
        """把包流按 trace 送进接收端：相机 blob 先到，之后逐包到达；
        解码在包到达后立刻开始（与上一包解码可重叠），达标包的**解码完成时刻**才算可渲染。"""
        tr = TRACES[trace_name]
        t_net, _ = transfer_time(tr, case["camera_bytes"], 0.0)   # 相机 blob 从 t=0 开始传
        t_busy = 0.0
        t_dec_finish = 0.0
        q = {}
        arrival, decode_ms = [], []
        hit = None
        for i, (name, pkt, kind, pb, b) in enumerate(case["packets"]):
            t_net, _ = transfer_time(tr, len(pkt), t_net)
            t0 = time.perf_counter()
            if kind == "base":
                qq, _bits, _sc, _nm, _sh = decode_base(pkt)
                q[name] = qq
            else:
                qq, _bits = apply_refine(q[name], pb, pkt)
                q[name] = qq
            dt = time.perf_counter() - t0
            t_busy += dt
            t_dec_finish = max(t_net, t_dec_finish) + dt      # 到达后排队解码
            decode_ms.append(dt * 1000.0)
            arrival.append({"i": i, "name": name, "bytes": len(pkt), "cum": case["offs"][i],
                            "t_arrive_s": round(t_net, 4),
                            "t_decode_finish_s": round(t_dec_finish, 4),
                            "t_decode_busy_s": round(t_busy, 4)})
            if case["offs"][i] >= case["target"]:
                hit = i
                break
        return arrival, decode_ms, hit, t_dec_finish

    def apply_target(bits):
        for c in planes:
            g, nm = c.split(".")
            if c.startswith("bases."):
                p = model.motion_bases.params[nm]
                p.data = torch.from_numpy(dequant(truncate(planes[c].q, bits.get(c, 16)), planes[c].scale)
                                          .reshape(planes[c].shape)).to(dev).to(p.dtype)
            else:
                dq = torch.from_numpy(dequant(truncate(planes[c].q, bits[c]), planes[c].scale)
                                      .reshape(planes[c].shape)).to(dev)
                if g == "fg" and nm == "motion_coefs":
                    model.fg.params["motion_coefs"].data = dq
                else:
                    getattr(model, g).params[nm].data = dq

    # 数据集（只为达标前缀下实测一帧渲染耗时）
    ddir = {
        "backpack": f"{_DATA}/scenes/backpack",
        "sriracha": f"{_DATA}/scenes/sriracha-tree",
        "truck": f"{_DATA}/Truck",
        "umbrella": f"{_DATA}/Umbrella",
        "jumping": f"{_DATA}/Jumping",
    }[sc]
    ds = "nvidia" if sc in ("truck", "umbrella", "jumping") else "iphone"
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    if ds == "iphone":
        from flow3d.data.iphone_dataset import iPhoneDataset
        val = iPhoneDataset(data_dir=ddir, factor=1, split="val", scene_norm_dict=snd, load_from_cache=True)
    else:
        from flow3d.data.nvidia_dataset import NvidiaDataset
        val = NvidiaDataset(data_dir=ddir, factor=2, split="val", scene_norm_dict=snd, load_from_cache=True)
    idx = 1

    from packets import dequant, truncate
    policies = list(POLICY_RUN) if a.policies == "all" else [a.policies]
    traces = list(TRACES) if a.traces == "all" else [a.traces]
    results = []
    for pol in policies:
        case = assemble(pol)
        apply_target(case["bits"])
        with torch.no_grad():
            w2c = val.w2cs[idx:idx + 1].to(dev); Kk = val.Ks[idx:idx + 1].to(dev)
            tid = int(val.time_ids[idx].item())
            H, Wd = int(val.imgs.shape[1]), int(val.imgs.shape[2])
            model.render(tid, w2c, Kk, (Wd, H))
            torch.cuda.synchronize()
            r0 = time.perf_counter()
            model.render(tid, w2c, Kk, (Wd, H))
            torch.cuda.synchronize()
            t_render = time.perf_counter() - r0
        for tr_name in traces:
            arrival, decode_ms, hit, t_dec_finish = replay(case, tr_name)
            rec = {
                "scene": sc, "policy": pol, "order": case["order"], "trace": tr_name,
                "trace_cfg": TRACES[tr_name], "n_packets": len(arrival),
                "camera_bytes": case["camera_bytes"],
                "target_bytes": int(case["target"]), "app_bytes": int(arrival[-1]["cum"]),
                "wire_bytes": int(arrival[-1]["cum"]),
                "t_transfer_s": round(arrival[-1]["t_arrive_s"], 4),
                "t_decode_finish_s": round(t_dec_finish, 4),
                "t_decode_busy_s": round(arrival[-1]["t_decode_busy_s"], 4),
                "t_render_s": round(t_render, 4),
                "t_startup_s": round(t_dec_finish + t_render, 4),
                "decode_ms_mean": round(float(np.mean(decode_ms)), 3),
                "decode_ms_p95": round(float(np.percentile(decode_ms, 95)), 3),
                "script": "delivery_sim.py v2 (camera blob + pipelined decode)",
            }
            results.append(rec)
            # 增量落盘：即便进程在退出阶段卡住，已完成的 trace 也不会丢
            with open(a.out or f"{root}/work/final/deliv_{sc}.json", "w") as _f:
                json.dump({"scene": sc, "rows": results}, _f, indent=1)
            print(f"{sc:9s} {pol:10s} {tr_name:7s} {rec['target_bytes']/1e6:6.2f}MB "
                  f"transfer={rec['t_transfer_s']:7.2f}s decode={rec['t_decode_busy_s']:6.3f}s "
                  f"startup={rec['t_startup_s']:7.2f}s")
    out = a.out or f"{root}/work/final/deliv_{sc}.json"
    json.dump({"scene": sc, "rows": results}, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
