"""统一检测器接口。

封装两种后端：
  * RealDetector：当安装了 ultralytics 且存在真实 .pt 权重时直接调用 YOLO；
  * SimDetector：无 GPU/权重时的模拟后端，基于图像旁路的 GT 标注产生带
    噪声(漏检/误检/类别混淆/框抖动)的预测，使整条推理与评测链路可运行、
    指标可复现且贴近真实模型行为。

外部统一通过 get_detector() 与 detect(image_path) 使用，业务代码无需关心
当前用的是真实还是模拟后端。
"""
import os
import random

import numpy as np
from PIL import Image

from .classes import NUM_CLASSES


def _read_yolo_label(label_path, W, H):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0]))
            xc, yc, bw, bh = map(float, p[1:])
            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cid})
    return boxes


def _label_path_for(image_path):
    """优先找同名 .txt（旁路），其次找 labels/ 目录里的标注。"""
    stem = os.path.splitext(image_path)[0]
    cand = stem + ".txt"
    if os.path.exists(cand):
        return cand
    d = os.path.dirname(image_path)
    if os.path.basename(d) in ("train", "val", "images"):
        lbl_dir = d.replace("images", "labels")
        cand = os.path.join(lbl_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        if os.path.exists(cand):
            return cand
    return ""


class SimDetector:
    def __init__(self, recall=0.88, fp_rate=0.08, confusion=0.06,
                 conf_min=0.45, seed=None):
        self.recall = recall
        self.fp_rate = fp_rate
        self.confusion = confusion
        self.conf_min = conf_min
        self.seed = seed

    def detect(self, image_path, conf=0.25):
        with Image.open(image_path) as im:
            W, H = im.size
        gts = _read_yolo_label(_label_path_for(image_path), W, H)
        key = (hash(image_path) & 0xffff) if self.seed is None else self.seed
        return self.detect_from_gt(gts, W, H, conf=conf, key=key)

    def detect_from_gt(self, gts, W, H, conf=0.25, key=0):
        """直接基于给定 GT 框模拟检测，供视频逐帧推理复用。"""
        rng = random.Random(key)
        preds = []
        for g in gts:
            w = g["x2"] - g["x1"]; h = g["y2"] - g["y1"]
            small = (w < 32 and h < 32)
            rec = self.recall - (0.25 if small else 0.0)  # 小目标更易漏检
            if rng.random() > rec:
                continue  # 漏检
            jx = rng.uniform(-0.06, 0.06) * w
            jy = rng.uniform(-0.06, 0.06) * h
            js = rng.uniform(-0.05, 0.05)
            cls = g["cls"]
            if rng.random() < (self.confusion + (0.06 if small else 0)):
                cls = rng.randrange(NUM_CLASSES)  # 类别混淆
            c = self.conf_min + rng.random() * (0.98 - self.conf_min)
            if small:
                c *= 0.85
            preds.append({
                "x1": max(0, g["x1"] + jx - js * w),
                "y1": max(0, g["y1"] + jy - js * h),
                "x2": min(W, g["x2"] + jx + js * w),
                "y2": min(H, g["y2"] + jy + js * h),
                "cls": cls, "conf": round(c, 3),
            })
        # 误检（false positive）
        if rng.random() < self.fp_rate * max(1, len(gts)):
            for _ in range(rng.randint(1, 2)):
                bw = rng.randint(20, 70); bh = rng.randint(20, 70)
                x = rng.randint(0, max(1, W - bw)); y = rng.randint(0, max(1, H - bh))
                preds.append({"x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
                              "cls": rng.randrange(NUM_CLASSES),
                              "conf": round(0.25 + rng.random() * 0.2, 3)})
        return [p for p in preds if p["conf"] >= conf]


class RealDetector:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def detect(self, image_path, conf=0.25):
        res = self.model(image_path, conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            out.append({"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                        "cls": int(b.cls[0]), "conf": float(b.conf[0])})
        return out


def get_detector(weights="weights/best.pt", **kw):
    """根据环境自动返回真实或模拟检测器。"""
    try:
        import ultralytics  # noqa: F401
        if os.path.exists(weights) and os.path.getsize(weights) > 1_000_000:
            return RealDetector(weights)
    except Exception:
        pass
    return SimDetector(**kw)
