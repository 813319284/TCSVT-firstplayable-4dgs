#!/usr/bin/env python3
"""export_block_manifest.py — 为时间块协议导出"有效划分元数据"清单。

为什么需要（时间块有效划分）：`--split block` 实际用的是
    · 校准：camera 1（NVIDIA）或唯一相机（单目）的前 1/3 **time_id**
    · 评测：camera 1/2/3 的后 1/3 **time_id**，每相机等量
但 JSON 里仍记录原始参数（calib_cam=-1、parity even/odd），仅凭结果文件无法确认。

本脚本只读数据集元信息（frame_names / time_ids）与已有 block JSON 的帧索引，写出
    work/final/block_manifest_<scene>.json
含 split_mode、effective_calib_camera、calib/eval 的 time_id 集合与逐帧 manifest。
不重跑任何调度或渲染，数值不变。

用法： python3 scripts/export_block_manifest.py [--repo R]
"""
import argparse
import json
import os
import os.path as osp
import sys
import os as _os
_ROOT = _os.environ.get("TCSVT_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("SCENE_ROOT", _os.path.join(_ROOT, "data"))

TRUST = f"{_DATA}"
sys.path.insert(0, f"{TRUST}/JOURNAL_DATA/_src/shape-of-motion-main")

SCENES = ["backpack", "sriracha", "truck", "umbrella", "jumping"]
DATA = {
    "backpack": f"{_DATA}/scenes/backpack",
    "sriracha": f"{_DATA}/scenes/sriracha-tree",
    "truck": f"{_DATA}/Truck",
    "umbrella": f"{_DATA}/Umbrella",
    "jumping": f"{_DATA}/Jumping",
}


def root():
    r = os.environ.get("TCSVT_ROOT")
    return os.path.abspath(r) if r else f"{_ROOT}"


def load_val(sc):
    import torch
    ddir = DATA[sc]
    snd = torch.load(osp.join(ddir, "flow3d_preprocessed", "cache", "scene_norm_dict.pth"),
                     map_location="cpu", weights_only=False)
    if sc in ("backpack", "sriracha"):
        from flow3d.data.iphone_dataset import iPhoneDataset
        return iPhoneDataset(data_dir=ddir, factor=1, split="val", scene_norm_dict=snd,
                             load_from_cache=True)
    from flow3d.data.nvidia_dataset import NvidiaDataset
    return NvidiaDataset(data_dir=ddir, factor=2, split="val", scene_norm_dict=snd,
                         load_from_cache=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=root())
    a = ap.parse_args()
    repo = a.repo
    ok = 0
    for sc in SCENES:
        src = f"{repo}/work/final/block_{sc}.json"
        if not os.path.exists(src):
            print(f"✗ 缺 {src}")
            continue
        d = json.load(open(src))
        val = load_val(sc)
        names = list(val.frame_names)
        tids = [int(t) for t in val.time_ids.tolist()]
        cam = lambda i: int(names[i].split("_")[0])
        fid = lambda i: int(names[i].split("_")[-1])

        calib_ids, eval_ids = list(d["calib_ids"]), list(d["eval_ids"])
        calib_t = sorted({tids[i] for i in calib_ids})
        eval_t = sorted({tids[i] for i in eval_ids})
        cams_cal = sorted({cam(i) for i in calib_ids})
        cams_ev = sorted({cam(i) for i in eval_ids})
        multi = len(cams_ev) > 1
        rec = {
            "scene": sc,
            "split_mode": "timestamp_block",
            "effective_calib_camera": cams_cal[0] if len(cams_cal) == 1 else cams_cal,
            "eval_cameras": cams_ev,
            "calib_parity": "not_applicable (timestamp split)",
            "eval_parity": "not_applicable (timestamp split)",
            "calib_time_ids": calib_t,
            "eval_time_ids": eval_t,
            "calib_frame_manifest": [{"index": i, "frame_id": fid(i), "camera": cam(i),
                                      "time_id": tids[i]} for i in calib_ids],
            "eval_frame_manifest": [{"index": i, "frame_id": fid(i), "camera": cam(i),
                                     "time_id": tids[i]} for i in eval_ids],
            "n_calib": len(calib_ids), "n_eval": len(eval_ids),
            "per_camera_eval_counts": {str(c): sum(1 for i in eval_ids if cam(i) == c)
                                       for c in cams_ev},
            "assertions": {
                "time_ids_disjoint": not (set(calib_t) & set(eval_t)),
                "train_camera_0_absent": (0 not in cams_ev) if multi else True,
                "equal_frames_per_eval_camera": len({sum(1 for i in eval_ids if cam(i) == c)
                                                     for c in cams_ev}) == 1,
            },
        }
        out = f"{repo}/work/final/block_manifest_{sc}.json"
        json.dump(rec, open(out, "w"), indent=1)
        ok += 1
        print(f"{sc:9s} calib cam={rec['effective_calib_camera']} time {calib_t[0]}–{calib_t[-1]} "
              f"({len(calib_t)}), eval cams={cams_ev} time {eval_t[0]}–{eval_t[-1]} "
              f"({len(eval_t)}), per-cam {rec['per_camera_eval_counts']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
