"""数据集/评测集质量检查脚本（对应需求 2 的 2.1~2.9）。

功能:
  2.1 每个类别的图片数量（直方图）
  2.2 每个类别的检测框数量（直方图）
  2.3 图片总数量
  2.4 检测框总数量
  2.5 平均每张图片的检测框数量
  2.6 图片与标注是否一一对应（打印缺失文件）
  2.7 图片是否损坏
  2.8 标注类别 ID 是否越界
  2.9 小目标(w,h 均 < 32px)比例统计

用法:
  python scripts/check_dataset.py --images data/tt100k/images/val \
         --labels data/tt100k/labels/val --out outputs/dataset_check
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classes import CLASSES, NUM_CLASSES, ID_TO_NAME  # noqa: E402
from src.draw import setup_matplotlib_cjk                 # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SMALL_THR = 32  # 小目标阈值（像素）


def _stem_map(folder, exts):
    out = {}
    if not os.path.isdir(folder):
        return out
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in exts:
            out[stem] = os.path.join(folder, f)
    return out


def check(images_dir, labels_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_map = _stem_map(images_dir, IMG_EXTS)
    lbl_map = _stem_map(labels_dir, (".txt",))

    img_per_cls = defaultdict(int)   # 出现某类的图片数
    box_per_cls = defaultdict(int)   # 某类检测框数
    total_boxes = 0
    small_boxes = 0
    corrupted = []
    out_of_range = []                # (file, line_no, cls)
    img_no_label = sorted(set(img_map) - set(lbl_map))
    label_no_img = sorted(set(lbl_map) - set(img_map))

    common = sorted(set(img_map) & set(lbl_map))
    for stem in common:
        # 2.7 图片损坏检查
        try:
            with Image.open(img_map[stem]) as im:
                im.verify()
            with Image.open(img_map[stem]) as im:
                W, H = im.size
        except Exception as e:
            corrupted.append((img_map[stem], str(e)))
            continue

        cls_in_img = set()
        with open(lbl_map[stem], "r") as f:
            for ln, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(float(parts[0]))
                _, _, bw, bh = map(float, parts[1:])
                # 2.8 类别越界
                if cid < 0 or cid >= NUM_CLASSES:
                    out_of_range.append((lbl_map[stem], ln, cid))
                    continue
                total_boxes += 1
                box_per_cls[cid] += 1
                cls_in_img.add(cid)
                # 2.9 小目标统计（按像素）
                if bw * W < SMALL_THR and bh * H < SMALL_THR:
                    small_boxes += 1
        for c in cls_in_img:
            img_per_cls[c] += 1

    total_imgs = len(common)
    avg_boxes = total_boxes / total_imgs if total_imgs else 0
    small_ratio = small_boxes / total_boxes if total_boxes else 0

    # ---------- 控制台报告 ----------
    print("=" * 60)
    print("数据集检查报告")
    print("=" * 60)
    print(f"[2.3] 图片总数量      : {total_imgs}")
    print(f"[2.4] 检测框总数量    : {total_boxes}")
    print(f"[2.5] 平均每张框数量  : {avg_boxes:.3f}")
    print(f"[2.9] 小目标(<32px)数 : {small_boxes}  比例: {small_ratio:.2%}")
    print("-" * 60)
    print(f"[2.6] 图片无标注 : {len(img_no_label)} 个")
    for x in img_no_label:
        print(f"      缺标注: {img_map[x]}")
    print(f"[2.6] 标注无图片 : {len(label_no_img)} 个")
    for x in label_no_img:
        print(f"      缺图片: {lbl_map[x]}")
    print(f"[2.7] 损坏图片   : {len(corrupted)} 个")
    for p, e in corrupted:
        print(f"      损坏: {p}  ({e})")
    print(f"[2.8] 越界标注   : {len(out_of_range)} 条")
    for p, ln, cid in out_of_range:
        print(f"      越界: {p}:{ln} cls={cid}")
    print("=" * 60)

    # ---------- 可视化 ----------
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    names = [ID_TO_NAME[i] for i in range(NUM_CLASSES)]
    img_counts = [img_per_cls[i] for i in range(NUM_CLASSES)]
    box_counts = [box_per_cls[i] for i in range(NUM_CLASSES)]

    # 2.1 每类图片数量直方图
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(names, img_counts, color="#4C72B0")
    ax.set_title("2.1 每个类别的图片数量")
    ax.set_xlabel("类别"); ax.set_ylabel("图片数")
    plt.xticks(rotation=90); plt.tight_layout()
    p1 = os.path.join(out_dir, "hist_images_per_class.png")
    plt.savefig(p1, dpi=130); plt.close()

    # 2.2 每类检测框数量直方图
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(names, box_counts, color="#DD8452")
    ax.set_title("2.2 每个类别的检测框数量")
    ax.set_xlabel("类别"); ax.set_ylabel("检测框数")
    plt.xticks(rotation=90); plt.tight_layout()
    p2 = os.path.join(out_dir, "hist_boxes_per_class.png")
    plt.savefig(p2, dpi=130); plt.close()

    # 小目标占比 + 概要图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].pie([small_boxes, max(0, total_boxes - small_boxes)],
                labels=[f"小目标 {small_boxes}", f"常规 {total_boxes - small_boxes}"],
                autopct="%1.1f%%", colors=["#C44E52", "#55A868"])
    axes[0].set_title("2.9 小目标(<32px)占比")
    summary = (f"图片总数: {total_imgs}\n检测框总数: {total_boxes}\n"
               f"平均每张框数: {avg_boxes:.2f}\n小目标比例: {small_ratio:.2%}\n"
               f"图片无标注: {len(img_no_label)}\n标注无图片: {len(label_no_img)}\n"
               f"损坏图片: {len(corrupted)}\n越界标注: {len(out_of_range)}")
    axes[1].axis("off")
    axes[1].text(0.02, 0.95, summary, va="top", fontsize=14, linespacing=1.8)
    axes[1].set_title("数据集概要")
    plt.tight_layout()
    p3 = os.path.join(out_dir, "summary.png")
    plt.savefig(p3, dpi=130); plt.close()

    # 机器可读报告
    report = {
        "total_images": total_imgs,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(avg_boxes, 4),
        "small_object_count": small_boxes,
        "small_object_ratio": round(small_ratio, 4),
        "images_without_label": img_no_label,
        "labels_without_image": label_no_img,
        "corrupted_images": [p for p, _ in corrupted],
        "out_of_range_labels": [{"file": p, "line": ln, "cls": cid}
                                for p, ln, cid in out_of_range],
        "images_per_class": {ID_TO_NAME[i]: img_counts[i] for i in range(NUM_CLASSES)},
        "boxes_per_class": {ID_TO_NAME[i]: box_counts[i] for i in range(NUM_CLASSES)},
    }
    rp = os.path.join(out_dir, "check_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[ok] 直方图与报告已保存到 {out_dir}")
    print(f"     - {p1}\n     - {p2}\n     - {p3}\n     - {rp}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/tt100k/images/val")
    ap.add_argument("--labels", default="data/tt100k/labels/val")
    ap.add_argument("--out", default="outputs/dataset_check")
    args = ap.parse_args()
    check(os.path.join(ROOT, args.images) if not os.path.isabs(args.images) else args.images,
          os.path.join(ROOT, args.labels) if not os.path.isabs(args.labels) else args.labels,
          os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out)


if __name__ == "__main__":
    main()
