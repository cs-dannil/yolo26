"""制作 TT100K 格式训练/验证数据集。

用法:
  # 1) 推荐：把官方 TT100K 转成 YOLO 格式（需先下载官方数据，见 README）
  python scripts/make_dataset.py --mode convert --tt100k /path/to/TT100K

  # 2) 环境受限时：生成与 TT100K 类别一致的合成数据，跑通全流程
  python scripts/make_dataset.py --mode synth --train 400 --val 100

输出目录结构（YOLO 标准）:
  data/tt100k/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    tt100k.yaml
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classes import CLASSES, NAME_TO_ID  # noqa: E402
from src.sign_factory import make_image       # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "tt100k")


def _ensure_dirs():
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATA_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "labels", split), exist_ok=True)


def write_yaml():
    path = os.path.join(DATA_DIR, "tt100k.yaml")
    lines = [
        "# TT100K 交通标志数据集 (YOLO 格式)",
        f"path: {DATA_DIR}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    for i, n in enumerate(CLASSES):
        lines.append(f"  {i}: {n}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] 写入数据集配置 {path}")
    return path


def gen_synth(n_train, n_val, small_ratio):
    _ensure_dirs()
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            img, labels = make_image(small_ratio=small_ratio, seed=1000 * (split == "val") + i)
            stem = f"{split}_{i:05d}"
            img.save(os.path.join(DATA_DIR, "images", split, stem + ".jpg"), quality=88)
            lp = os.path.join(DATA_DIR, "labels", split, stem + ".txt")
            with open(lp, "w") as f:
                for cid, xc, yc, bw, bh in labels:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"[ok] 生成 {split} 集 {n} 张")
    write_yaml()


def convert_tt100k(src):
    """把官方 TT100K (annotations.json + 图片) 转 YOLO。仅转换 45 类子集。"""
    _ensure_dirs()
    ann_path = os.path.join(src, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"[err] 未找到 {ann_path}，请确认 TT100K 路径")
        return
    with open(ann_path, "r") as f:
        data = json.load(f)
    imgs = data["imgs"]
    cnt = {"train": 0, "val": 0}
    for _id, info in imgs.items():
        rel = info["path"]                 # e.g. train/12345.jpg
        split = "train" if rel.startswith("train") else "val"
        objs = info.get("objects", [])
        keep = [o for o in objs if o["category"] in NAME_TO_ID]
        if not keep:
            continue
        src_img = os.path.join(src, rel)
        if not os.path.exists(src_img):
            continue
        from PIL import Image
        im = Image.open(src_img)
        W, H = im.size
        stem = os.path.splitext(os.path.basename(rel))[0]
        im.convert("RGB").save(os.path.join(DATA_DIR, "images", split, stem + ".jpg"), quality=90)
        with open(os.path.join(DATA_DIR, "labels", split, stem + ".txt"), "w") as f:
            for o in keep:
                b = o["bbox"]
                xc = (b["xmin"] + b["xmax"]) / 2 / W
                yc = (b["ymin"] + b["ymax"]) / 2 / H
                bw = (b["xmax"] - b["xmin"]) / W
                bh = (b["ymax"] - b["ymin"]) / H
                f.write(f"{NAME_TO_ID[o['category']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        cnt[split] += 1
    print(f"[ok] 转换完成 train={cnt['train']} val={cnt['val']}")
    write_yaml()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "convert"], default="synth")
    ap.add_argument("--tt100k", default="", help="官方 TT100K 根目录")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--small-ratio", type=float, default=0.35)
    args = ap.parse_args()
    if args.mode == "convert":
        convert_tt100k(args.tt100k)
    else:
        gen_synth(args.train, args.val, args.small_ratio)


if __name__ == "__main__":
    main()
