#!/usr/bin/env python3
"""freeze.py — 交付件的“内容级”冻结台账。

为什么不用 zip 的 md5 当冻结凭据：zip 把每个文件的 mtime/顺序/权限写进包，
`package_deliverable.sh` 一重跑就整包刷新，md5 必变，与内容是否改动无关。
（2026-11-19 实测：PDF 仍是 43bd3379，重打包后两个 zip 的 md5 换代，编译页数/词数不变。）

所以台账分两层：
  1) pdf：sha256 + md5 + 页数 + 词数   —— 对外凭据，投稿系统上传的就是它。
  2) zip：md5 + 每个包内文件的 sha256 —— 内容级，打包元数据变了也不影响。

用法：
  python3 scripts/freeze.py write   [--repo R]   # 写冻结台账
  python3 scripts/freeze.py check   [--repo R]   # 与台账比对，退出码 0=冻结未破
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import shlex
import zipfile
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

PDF = "deliverable/TCSVT_SI_manuscript.pdf"
ZIPS = ["deliverable/latex_source.zip", "deliverable/supplementary.zip"]
LEDGER = "relay/FREEZE.json"


def root() -> str:
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_pages(path: str):
    try:
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
        pages = next((l.split()[1] for l in out.splitlines() if l.startswith("Pages:")), "?")
        # 与 build_check.sh / package_deliverable.sh 同口径：pdftotext（默认模式）→ python split()
        txt = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True).stdout
        return pages, len(txt.split())
    except Exception:
        return "?", 0


def snapshot(repo: str) -> dict:
    snap = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "pdf": {}, "zips": {}}
    p = os.path.join(repo, PDF)
    if os.path.isfile(p):
        pages, words = pdf_pages(p)
        snap["pdf"] = {
            "path": PDF,
            "sha256": sha256(p),
            "md5": md5(p),
            "pages": pages,
            "words": words,
            "bytes": os.path.getsize(p),
        }
    for z in ZIPS:
        zp = os.path.join(repo, z)
        if not os.path.isfile(zp):
            continue
        files = {}
        with zipfile.ZipFile(zp) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                files[info.filename] = hashlib.sha256(zf.read(info.filename)).hexdigest()
        snap["zips"][os.path.basename(z)] = {
            "sha256": sha256(zp),
            "md5": md5(zp),
            "bytes": os.path.getsize(zp),
            "n_files": len(files),
            "files": files,
        }
    return snap


def cmd_write(repo: str) -> int:
    snap = snapshot(repo)
    out = os.path.join(repo, LEDGER)
    old = None
    if os.path.isfile(out):
        old = json.load(open(out))
        bak = out + ".bak-" + time.strftime("%m%d-%H%M")
        os.replace(out, bak)
        print(f"  旧台账 → {os.path.basename(bak)}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(snap, open(out, "w"), indent=1, ensure_ascii=False, sort_keys=True)
    print(f"✓ 台账已写 {LEDGER}")
    if snap["pdf"]:
        print(f"  PDF {snap['pdf']['pages']} 页 / {snap['pdf']['words']} 词 · md5 {snap['pdf']['md5'][:8]}… · sha256 {snap['pdf']['sha256'][:12]}…")
    for name, z in snap["zips"].items():
        print(f"  {name}: {z['n_files']} 个文件 · md5 {z['md5'][:8]}…")
    if old:
        for name in snap["zips"]:
            o = (old.get("zips") or {}).get(name)
            if not o:
                continue
            added = set(snap["zips"][name]["files"]) - set(o["files"])
            removed = set(o["files"]) - set(snap["zips"][name]["files"])
            changed = [f for f in set(snap["zips"][name]["files"]) & set(o["files"])
                       if snap["zips"][name]["files"][f] != o["files"][f]]
            print(f"  与上一版内容比：+{len(added)} −{len(removed)} 改{len(changed)}"
                  + (f"（改：{', '.join(sorted(changed)[:6])}）" if changed else ""))
    return 0


def cmd_check(repo: str) -> int:
    out = os.path.join(repo, LEDGER)
    if not os.path.isfile(out):
        print(f"✗ 缺台账 {LEDGER}，先跑 freeze.py write")
        return 1
    ref = json.load(open(out))
    cur = snapshot(repo)
    bad = []

    rp, cp = ref.get("pdf") or {}, cur.get("pdf") or {}
    if not cp:
        bad.append("缺交付 PDF")
    elif rp.get("sha256") != cp.get("sha256"):
        bad.append(f"PDF sha256 变 {rp.get('sha256','?')[:12]} → {cp.get('sha256','?')[:12]}")
    elif (rp.get("pages"), rp.get("words")) != (cp.get("pages"), cp.get("words")):
        bad.append(f"PDF 页/词 变 {rp.get('pages')}/{rp.get('words')} → {cp.get('pages')}/{cp.get('words')}")
    else:
        print(f"  PDF ✓ {cp['pages']} 页 / {cp['words']} 词 · md5 {cp['md5'][:8]}…")

    for name in ZIPS:
        b = os.path.basename(name)
        rz, cz = (ref.get("zips") or {}).get(b), (cur.get("zips") or {}).get(b)
        if not cz:
            bad.append(f"缺 {b}")
            continue
        if not rz:
            bad.append(f"台账里没有 {b}")
            continue
        added = sorted(set(cz["files"]) - set(rz["files"]))
        removed = sorted(set(rz["files"]) - set(cz["files"]))
        changed = sorted(f for f in set(cz["files"]) & set(rz["files"]) if cz["files"][f] != rz["files"][f])
        if added or removed or changed:
            bad.append(f"{b} 内容变：+{len(added)} −{len(removed)} 改{len(changed)}"
                       + (f"（改：{', '.join(changed[:6])}）" if changed else "")
                       + (f"（增：{', '.join(added[:6])}）" if added else "")
                       + (f"（删：{', '.join(removed[:6])}）" if removed else ""))
        else:
            same = "md5 同" if cz["md5"] == rz["md5"] else "md5 异（打包元数据）"
            print(f"  {b} ✓ {cz['n_files']} 个文件逐字一致（{same}）")

    if bad:
        print("✗ 冻结已破：")
        for x in bad:
            print("   -", x)
        return 1
    print("✓ 冻结未破（内容级）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check"])
    ap.add_argument("--repo", default=root())
    a = ap.parse_args()
    return cmd_write(a.repo) if a.cmd == "write" else cmd_check(a.repo)


if __name__ == "__main__":
    sys.exit(main())
