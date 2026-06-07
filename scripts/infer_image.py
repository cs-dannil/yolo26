"""单张图片推理并可视化（需求 4）。

  python scripts/infer_image.py --image assets/test_image.jpg --out outputs/infer

若未提供 --image，则自动生成一张包含交通标志的测试图片再进行推理。
推理结果（预测框+类别+置信度）会绘制并保存。
"""
import argparse
import os
import sys
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detector import get_detector   # noqa: E402
from src.draw import draw_boxes         # noqa: E402
from src.sign_factory import make_image  # noqa: E402
from src.classes import class_label     # noqa: E402


def ensure_test_image():
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    path = os.path.join(ROOT, "assets", "test_image.jpg")
    if not os.path.exists(path):
        img, labels = make_image(min_signs=5, max_signs=6, small_ratio=0.12, seed=21)
        img.save(path, quality=92)
        with open(os.path.splitext(path)[0] + ".txt", "w") as f:
            for cid, xc, yc, bw, bh in labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/infer"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    image = args.image or ensure_test_image()
    det = get_detector(args.weights)

    t0 = time.time()
    boxes = det.detect(image, conf=args.conf)
    dt = (time.time() - t0) * 1000

    img = Image.open(image).convert("RGB")
    import numpy as np
    vis = draw_boxes(np.array(img), boxes, with_label=True)
    out_path = os.path.join(args.out, "result_" + os.path.basename(image))
    Image.fromarray(vis).save(out_path)

    print("=" * 50)
    print(f"推理图片: {image}")
    print(f"检测到目标: {len(boxes)} 个   耗时: {dt:.1f} ms")
    for b in boxes:
        print(f"  - {class_label(int(b['cls'])):22s} conf={b['conf']:.2f} "
              f"box=({int(b['x1'])},{int(b['y1'])},{int(b['x2'])},{int(b['y2'])})")
    print(f"[ok] 可视化结果已保存: {out_path}")


if __name__ == "__main__":
    main()
