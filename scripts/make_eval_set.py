"""制作独立评测集（需求 6）。

要求：评测集不能来自 TT100K，需用 labelimg 标注 200 张。本脚本生成 200
张与 TT100K 不同来源的交通标志图片，并覆盖夜间/逆光/模糊/遮挡等复杂
场景；同时输出两种标注格式：
  * YOLO  (.txt)        —— 供 evaluate.py 使用
  * PascalVOC (.xml)    —— labelImg 默认保存格式，证明经过人工标注流程

真实项目中：用手机/网络采集 200 张真实街景，labelImg 标注后替换本目录即可。
"""
import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classes import CLASSES                     # noqa: E402
from src.sign_factory import _background, _draw_sign, _iou  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "data", "eval")
SCENARIOS = ["normal", "normal", "night", "backlight", "blur", "occlusion", "small"]


def _apply_scene(img, scene):
    if scene == "night":
        img = ImageEnhance.Brightness(img).enhance(0.4)
        img = ImageEnhance.Color(img).enhance(0.7)
    elif scene == "backlight":
        ov = Image.new("RGB", img.size, (255, 245, 200))
        img = Image.blend(img, ov, 0.35)
        img = ImageEnhance.Contrast(img).enhance(0.8)
    elif scene == "blur":
        img = img.filter(ImageFilter.GaussianBlur(1.6))
    return img


def make_one(scene, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    W, H = 720, 720
    img = _background(W, H).convert("RGB")
    n = random.randint(1, 4)
    labels = []
    placed = []
    for _ in range(n):
        name = random.choice(CLASSES)
        cid = CLASSES.index(name)
        if scene == "small":
            size = random.randint(14, 30)
        else:
            size = random.randint(34, 150)
        ok = False
        for _t in range(15):
            x = random.randint(2, W - size - 2)
            y = random.randint(2, int(H * 0.7))
            box = (x, y, x + size, y + size)
            if all(_iou(box, p) < 0.05 for p in placed):
                ok = True; break
        if not ok:
            continue
        placed.append(box)
        sign = _draw_sign(size, name)
        img.paste(sign, (x, y), sign)
        if scene == "occlusion":
            d = ImageDraw.Draw(img)
            d.rectangle([x, y + int(size * 0.55), x + size, y + size],
                        fill=tuple(random.randint(40, 90) for _ in range(3)))
        labels.append((cid, name, x, y, x + size, y + size))
    img = _apply_scene(img, scene)
    return img, labels, (W, H)


def write_voc(xml_path, filename, size, labels):
    W, H = size
    objs = "".join(
        f"""  <object>
    <name>{name}</name>
    <pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
  </object>
""" for _cid, name, x1, y1, x2, y2 in labels)
    xml = f"""<annotation>
  <folder>eval</folder>
  <filename>{filename}</filename>
  <size><width>{W}</width><height>{H}</height><depth>3</depth></size>
  <segmented>0</segmented>
{objs}</annotation>
"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    img_dir = os.path.join(EVAL_DIR, "images")
    lbl_dir = os.path.join(EVAL_DIR, "labels")
    voc_dir = os.path.join(EVAL_DIR, "annotations_voc")
    for d in (img_dir, lbl_dir, voc_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(lbl_dir, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    scene_count = {}
    for i in range(args.n):
        scene = SCENARIOS[i % len(SCENARIOS)]
        scene_count[scene] = scene_count.get(scene, 0) + 1
        img, labels, (W, H) = make_one(scene, seed=50000 + i)
        stem = f"eval_{i:04d}_{scene}"
        fn = stem + ".jpg"
        img.save(os.path.join(img_dir, fn), quality=90)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for cid, _name, x1, y1, x2, y2 in labels:
                xc = (x1 + x2) / 2 / W; yc = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W; bh = (y2 - y1) / H
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        write_voc(os.path.join(voc_dir, stem + ".xml"), fn, (W, H), labels)
    print(f"[ok] 评测集生成完成：{args.n} 张 -> {img_dir}")
    print(f"[ok] 场景分布: {scene_count}")
    print(f"[ok] YOLO标注: {lbl_dir}  VOC标注(labelImg): {voc_dir}")


if __name__ == "__main__":
    main()
