#!/usr/bin/env python3
"""packets.py — 真实可解码的位平面增量包（serializer / decoder / byte ledger）。

设计要点：
  * 每个属性张量先量化到 **16 bit 参考网格**：q16 = round(x / scale)，scale 由
    有限值的 99.99 分位决定，q16 ∈ [-32767, 32767]。
  * b bit 的模型 = q16 的高 b 位（算术右移 16-b 位再左移回来），因此任意 b
    都是同一网格上的截断，**可以增量重建**。
  * 升级包 b → b′ 只携带新增的 (b′−b) 位，按最小 dtype 打包后 zlib 压缩；
    包头是 JSON（张量名、形状、位宽、scale、版本），**计入字节**。
  * 解码器只吃包里有的东西：`decode_base()` → `apply_refine()` 逐步重建，
    不需要原始浮点张量。

自检（`selftest`）：基包 + 全部升级包重建出的整数必须与 q16 的高 16 位完全相等
（bit-exact），并且与直接截断的结果逐元素相同。

用法（自检）：
    python3 scripts/packets.py --selftest
"""
from __future__ import annotations

import json
import struct
import sys
import zlib

import numpy as np

VERSION = 1
MAGIC = b"R4DP"
LEVELS = [4, 6, 8, 12, 16]


# --------------------------------------------------------------------- 量化
def quantize16(x: np.ndarray):
    """float 张量 → (q16 int64, scale float)。与旧口径一致的 99.99 分位鲁棒尺度。"""
    x = np.asarray(x, dtype=np.float32)
    fin = np.isfinite(x)
    if fin.any():
        amax = float(np.quantile(np.abs(x[fin]), 0.9999))
    else:
        amax = 1.0
    amax = max(amax, 1e-6)
    scale = amax / 32767.0
    q = np.clip(np.round(np.nan_to_num(x, nan=0.0, posinf=amax, neginf=-amax) / scale),
                -32767, 32767).astype(np.int64)
    return q, scale


def truncate(q: np.ndarray, bits: int) -> np.ndarray:
    """取 16 bit 网格的高 bits 位（算术移位，保持符号）。"""
    s = 16 - int(bits)
    return (q >> s) << s


def dequant(q: np.ndarray, scale: float) -> np.ndarray:
    return (q.astype(np.float32) * np.float32(scale))


# --------------------------------------------------------------------- 包头
def _header(kind: str, name: str, shape, bits: int, prev_bits: int, scale: float) -> bytes:
    h = {"v": VERSION, "k": kind, "n": name, "s": list(shape), "b": int(bits),
         "p": int(prev_bits), "sc": float(scale)}
    return json.dumps(h, separators=(",", ":")).encode("utf-8")


def _payload_bits(resid: np.ndarray, width: int) -> bytes:
    """把 width 位的残差按最小 dtype 存下再压缩。"""
    if width <= 8:
        arr = resid.astype(np.uint8)
    elif width <= 16:
        arr = resid.astype(np.uint16)
    else:
        arr = resid.astype(np.uint32)
    return zlib.compress(arr.tobytes(), 9)


# --------------------------------------------------------------------- 编码
class Plane:
    """一个属性张量的位平面编码器（16 bit 参考网格上的可截断表示）。"""

    def __init__(self, name: str, x: np.ndarray):
        self.name = name
        self.shape = tuple(np.shape(x))
        self.q, self.scale = quantize16(x)
        self._cache = {}

    def base_packet(self, bits: int = 4) -> bytes:
        """基础包：该张量在 bits 位的粗量化值（完整张量，无符号网格）。"""
        key = ("base", bits)
        if key in self._cache:
            return self._cache[key]
        s = 16 - bits
        coarse = ((self.q >> s) + (1 << (bits - 1))).astype(np.int64)
        head = _header("base", self.name, self.shape, bits, 0, self.scale)
        pkt = MAGIC + struct.pack("<I", len(head)) + head + _payload_bits(coarse, bits + 1)
        self._cache[key] = pkt
        return pkt

    def refine_packet(self, bits_from: int, bits_to: int) -> bytes:
        """升级包：只带 bits_from → bits_to 之间新增的位。"""
        assert bits_to > bits_from
        key = ("ref", bits_from, bits_to)
        if key in self._cache:
            return self._cache[key]
        s_from, s_to = 16 - bits_from, 16 - bits_to
        width = s_from - s_to
        u_from = (self.q >> s_from) + (1 << (bits_from - 1))
        u_to = (self.q >> s_to) + (1 << (bits_to - 1))
        resid = u_to - (u_from << width)                            # ∈ [0, 2^width)
        head = _header("ref", self.name, self.shape, bits_to, bits_from, self.scale)
        pkt = MAGIC + struct.pack("<I", len(head)) + head + _payload_bits(resid, width)
        self._cache[key] = pkt
        return pkt

    def bytes_at(self, bits: int, prev_bits: int = 0) -> int:
        return len(self.base_packet(bits)) if prev_bits == 0 else len(self.refine_packet(prev_bits, bits))


# --------------------------------------------------------------------- 解码
def parse(pkt: bytes):
    assert pkt[:4] == MAGIC, "bad magic"
    (hlen,) = struct.unpack("<I", pkt[4:8])
    head = json.loads(pkt[8:8 + hlen].decode("utf-8"))
    return head, pkt[8 + hlen:]


def decode_payload(payload: bytes, width: int, n: int, wide: bool) -> np.ndarray:
    raw = zlib.decompress(payload)
    dt = np.uint16 if wide else np.uint8
    arr = np.frombuffer(raw, dtype=dt, count=n).astype(np.int64)
    assert arr.size == n
    return arr


def decode_base(pkt: bytes):
    """基础包 → (coarse int64 张量, bits, scale, name, shape)。"""
    head, payload = parse(pkt)
    assert head["k"] == "base"
    bits = head["b"]
    n = int(np.prod(head["s"]))
    wide = bits + 1 > 8
    coarse_u = decode_payload(payload, bits + 1, n, wide=wide).astype(np.int64)
    s = 16 - bits
    q = ((coarse_u - (1 << (bits - 1))) << s)
    return q, bits, head["sc"], head["n"], tuple(head["s"])


def apply_refine(q: np.ndarray, bits_from: int, pkt: bytes):
    """把升级包叠到一个 bits_from 的整数张量上。"""
    head, payload = parse(pkt)
    assert head["k"] == "ref" and head["p"] == bits_from
    bits_to = head["b"]
    s_from, s_to = 16 - bits_from, 16 - bits_to
    width = s_from - s_to
    n = int(np.prod(head["s"]))
    resid = decode_payload(payload, width, n, wide=width > 8)
    u_from = (q >> s_from) + (1 << (bits_from - 1))
    u_new = (u_from << width) | resid
    q_new = (u_new - (1 << (bits_to - 1))) << s_to
    return q_new, bits_to


def decode_stream(packets, bits_target: int):
    """从基础包 + 若干升级包重建 target 位宽的整数张量（严格按码流解码）。"""
    q, bits, scale, name, shape = decode_base(packets[0])
    for pkt in packets[1:]:
        q, bits = apply_refine(q, bits, pkt)
    assert bits == bits_target, f"{bits} != {bits_target}"
    return q.reshape(shape), bits_target, scale, name, shape


# --------------------------------------------------------------------- 自检
def selftest() -> int:
    rng = np.random.default_rng(0)
    cases = [("rand-norm", rng.normal(size=(977, 4)).astype(np.float32)),
             ("rand-heavy", (rng.standard_t(3, size=(5000,)).astype(np.float32) * 3.0)),
             ("const", np.full((513,), 2.5, dtype=np.float32))]
    bad = 0
    for name, x in cases:
        pl = Plane(name, x)
        pkts = [pl.base_packet(4)]
        prev = 4
        for b in LEVELS[1:]:
            pkts.append(pl.refine_packet(prev, b))
            prev = b
        q_stream, bits, scale, _, shape = decode_stream(pkts, 16)
        assert bits == 16
        if not np.array_equal(q_stream.ravel(), pl.q.ravel()):
            bad += 1
            print(f"  ✗ {name}: 码流重建 != q16")
        # 中间位宽也要逐位一致
        for b in LEVELS:
            idx = LEVELS.index(b)
            q_b, _, _, _, _ = decode_stream(pkts[:idx + 1], b)
            if not np.array_equal(q_b.ravel(), truncate(pl.q, b).ravel()):
                bad += 1
                print(f"  ✗ {name} @ {b}bit: 增量重建 != 截断结果")
        total = sum(len(p) for p in pkts)
        print(f"  ✓ {name}: {x.size} elems, 16-bit 码流 {total/1024:.1f} KiB, "
              f"base={len(pkts[0])/1024:.1f} KiB")
    if bad == 0:
        print("selftest: 全部通过（bit-exact，含中间位宽）")
        return 0
    print(f"selftest: {bad} 项失败")
    return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
