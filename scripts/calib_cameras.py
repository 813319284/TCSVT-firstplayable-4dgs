#!/usr/bin/env python3
"""calib_cameras.py — 写出每个场景"标定帧/评测帧各来自哪些相机"的清单。

随包 JSON 只有 calib_ids，没有逐帧 camera 字段，
因此"标定来自单一视角、评测包含未参与标定的相机"这一句无法从交付包核验。
本脚本把这件事做成可核对的清单（只读现有 JSON，不需重跑任何实验）：

  work/final/calib_cameras_<scene>.json
    calib: [{frame_id, camera, source}]
    eval:  [{frame_id, camera, source}]
    counts: {n_calib, n_eval, n_eval_cameras_unseen, cameras_calib, cameras_eval}

相机标签口径：
  · 评测帧：直接用 p0_<scene>.json 里 results[*].per_frame 记录的 camera（权威）。
  · 标定帧：与评测帧不相交，JSON 里没有直接记录，用帧 id 的分段推得（id//200+1）。
    对单目场景（DyCheck）评测帧的 camera 一律为 0，标定帧随之记为 0。

用法： python3 scripts/calib_cameras.py [--repo R]
"""
import argparse
import json
import os
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]


def root() -> str:
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def find(repo: str, rel: str) -> str:
    cands = [f"{repo}/{rel}"]
    if rel.startswith("work/final/"):
        cands.append(f"{repo}/results/raw/{rel[len('work/final/'):]}")
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=root())
    a = ap.parse_args()
    repo = a.repo
    out_any = False
    for sc in SCENES:
        p = find(repo, f"work/final/p0_{sc}.json")
        if not os.path.exists(p):
            print(f"✗ 缺 {p}")
            continue
        d = json.load(open(p))
        per_frame = d["results"][0].get("per_frame") or []
        eval_cam = {x["frame_id"]: x["camera"] for x in per_frame}
        monocular = len(set(eval_cam.values())) == 1 and 0 in eval_cam.values()
        calib = [{"frame_id": i, "camera": 0 if monocular else i // 200 + 1,
                  "source": "id-range inference"} for i in d["calib_ids"]]
        ev = [{"frame_id": i, "camera": eval_cam.get(i, -1), "source": "per_frame record"}
              for i in d["eval_ids"]]
        cams_calib = sorted({x["camera"] for x in calib})
        cams_eval = sorted({x["camera"] for x in ev})
        unseen = [x for x in ev if x["camera"] not in cams_calib]
        rec = {
            "scene": sc,
            "calib": calib,
            "eval": ev,
            "counts": {
                "n_calib": len(calib),
                "n_eval": len(ev),
                "n_eval_cameras_unseen": len(unseen),
                "cameras_calib": cams_calib,
                "cameras_eval": cams_eval,
                "monocular": monocular,
            },
        }
        out = find(repo, f"work/final/calib_cameras_{sc}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rec, open(out, "w"), indent=1, ensure_ascii=False)
        out_any = True
        print(f"{sc:9s} calib cams={cams_calib} eval cams={cams_eval} "
              f"未参与标定的评测帧={len(unseen)}/{len(ev)}")
    return 0 if out_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
