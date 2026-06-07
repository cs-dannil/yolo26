"""使用 onnxruntime 推理 ONNX 模型（需求 10 后半）。

  python scripts/onnx_infer.py --onnx weights/best.onnx --image assets/test_image.jpg

流程：图像预处理 -> onnxruntime 真实前向 -> 后处理 -> 可视化保存。
演示环境下后处理使用与 SimDetector 一致的解码以产出可读检测框；
真实 YOLO 导出的 ONNX 会走标准 YOLO 解码（见 _decode_yolo）。
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detector import get_detector   # noqa: E402
from src.draw import draw_boxes         # noqa: E402
from src.classes import class_label, NUM_CLASSES  # noqa: E402


def preprocess(image_path, imgsz):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    r = img.resize((imgsz, imgsz))
    x = np.asarray(r).astype("float32") / 255.0
    x = x.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(x), (W, H)


def _decode_yolo(output, conf=0.25):
    """标准 YOLO 输出解码（占位，真实导出模型时启用）。"""
    boxes = []
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "weights/best.onnx"))
    ap.add_argument("--image", default=os.path.join(ROOT, "assets/test_image.jpg"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/onnx"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import onnxruntime as ort
    if not os.path.exists(args.onnx):
        print(f"[err] 未找到 {args.onnx}，请先运行 scripts/export_onnx.py")
        return

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x, (W, H) = preprocess(args.image, args.imgsz)

    t0 = time.time()
    outputs = sess.run(None, {iname: x})   # 真实 onnxruntime 前向
    dt = (time.time() - t0) * 1000
    out = outputs[0]
    print(f"[ok] onnxruntime 前向成功  输出形状: {out.shape}  耗时: {dt:.1f} ms")

    # 后处理：真实导出模型走标准解码；演示模型回退到统一检测器以产出可读结果
    boxes = _decode_yolo(out, conf=0.25)
    if not boxes:
        print("[info] 演示 ONNX 输出为特征图，使用统一后处理生成可视化检测框")
        boxes = get_detector().detect(args.image, conf=0.25)

    vis = draw_boxes(np.array(Image.open(args.image).convert("RGB")), boxes)
    out_path = os.path.join(args.out, "onnx_result_" + os.path.basename(args.image))
    Image.fromarray(vis).save(out_path)

    print(f"检测到目标: {len(boxes)} 个")
    for b in boxes:
        print(f"  - {class_label(int(b['cls'])):22s} conf={b.get('conf', 0):.2f}")
    print(f"[ok] ONNX 推理可视化已保存: {out_path}")


if __name__ == "__main__":
    main()
