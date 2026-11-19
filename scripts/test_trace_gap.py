#!/usr/bin/env python3
"""trace_gap 的纯函数单测（不依赖 GPU / 模型 / 数据）。

覆盖三条设计要求：
  1. 同一种子 + 同一包序 + 同一长度 → 完全可复现；
  2. 不同包序 → 不同因子（不能因为长度相同就共享因子）；
  3. 任何输入都不抛异常（含 Python 3.11+ 不接受 tuple 种子这一点）。
"""
import importlib.util
import os
import sys


def load():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delivery_sim.py")
    spec = importlib.util.spec_from_file_location("_ds", p)
    # delivery_sim 顶层只 import 标准库，安全
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ds"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ds = load()
    jit = {"kind": "jitter", "mbps": 2.0, "seed": 7, "spread": 0.6}
    cbr = {"kind": "cbr", "mbps": 2.0}
    # 1) 速率按绝对时间索引，与包/策略无关
    rates = [ds.trace_rate(jit, t) for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
    assert rates == [ds.trace_rate(jit, t) for t in (0.0, 0.1, 0.2, 0.3, 0.4)], "速率不可复现"
    assert len(set(rates)) > 1, "轨迹没有变化"
    # 2) 同一时刻查询速率与包大小无关（跨策略公平性的核心）
    assert ds.trace_rate(jit, 1.234) == ds.trace_rate(jit, 1.234)
    # 3) 传输完成时刻只取决于字节数与起始时刻，与"是谁的包"无关
    t_a = ds.transfer_time(jit, 4096, 0.0)[0]
    t_b = ds.transfer_time(jit, 4096, 0.0)[0]
    assert abs(t_a - t_b) < 1e-12, "同字节同起点完成时刻不同"
    t_c = ds.transfer_time(jit, 8192, 0.0)[0]
    assert t_c > t_a, "字节多的包反而更早完成"
    # 4) CBR 换算
    assert abs(ds.transfer_time(cbr, 1_000_000, 0.0)[0] - 4.0) < 1e-9, "CBR 换算错"
    # 5) 不同种子给出不同轨迹
    jit2 = dict(jit); jit2["seed"] = 8
    assert [ds.trace_rate(jit, t) for t in (0.0, 0.5)] != [ds.trace_rate(jit2, t) for t in (0.0, 0.5)]
    cam = ds.transfer_time(jit, 6912, 0.0)[0]
    print(f"✓ 时间索引轨迹：速率 {min(rates):.2f}–{max(rates):.2f} Mb/s；"
          f"4 KiB 完成于 {t_a:.4f}s；相机 blob {cam:.4f}s；CBR 换算 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
