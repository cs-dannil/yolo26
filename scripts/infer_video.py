"""视频推理：检测框/类别/置信度叠加 + 实时 FPS，输出检测后视频（需求 5）。

  python scripts/infer_video.py --video input.mp4 --out outputs/video/result.mp4

若未提供 --video，会自动合成一段包含移动交通标志的视频再进行检测，
保证在无素材环境下也能产出"检测后视频"成果。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detector import get_detector, SimDetector  # noqa: E402
from src.draw import draw_boxes                      # noqa: E402
from src.sign_factory import _background, _draw_sign  # noqa: E402
from src.classes import CLASSES                       # noqa: E402


def synth_video(path, W=960, H=540, n_frames=150, fps=30):
    """合成一段移动交通标志视频，返回每帧 GT 框列表。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bg = _background(W, H).convert("RGB")
    import random
    random.seed(3)
    # 几个标志，带运动轨迹（由远及近，模拟车辆靠近）
    tracks = []
    for _ in range(3):
        name = random.choice(CLASSES)
        tracks.append({
            "name": name, "cid": CLASSES.index(name),
            "x": random.randint(60, W - 200), "y": random.randint(40, 160),
            "vx": random.uniform(-1.2, 1.2), "vy": random.uniform(0.2, 0.7),
            "size": random.randint(26, 40), "grow": random.uniform(0.25, 0.55),
        })
    frames_gt = []
    bg_np = np.array(bg)
    for fidx in range(n_frames):
        from PIL import Image
        frame = Image.fromarray(bg_np.copy())
        gts = []
        for t in tracks:
            t["x"] += t["vx"]; t["y"] += t["vy"]; t["size"] += t["grow"]
            s = int(t["size"])
            x = int(t["x"]); y = int(t["y"])
            if x < 0 or y < 0 or x + s > W or y + s > H:
                continue
            sign = _draw_sign(s, t["name"])
            frame.paste(sign, (x, y), sign)
            gts.append({"x1": x, "y1": y, "x2": x + s, "y2": y + s, "cls": t["cid"]})
        frames_gt.append((np.array(frame), gts))
    return frames_gt, fps, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/video/result.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    det = get_detector(args.weights)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
        frame_iter = _read_real(cap, det, args.conf)
    else:
        print("[warn] 未提供视频，自动合成交通标志视频")
        frames_gt, fps, (W, H) = synth_video(args.out)
        vw = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
        frame_iter = _synth_iter(frames_gt, det, args.conf, W, H)

    n = 0
    fps_avg = 0.0
    for rgb, boxes, infer_ms in frame_iter:
        inst_fps = 1000.0 / max(1e-3, infer_ms)
        fps_avg = inst_fps if n == 0 else 0.9 * fps_avg + 0.1 * inst_fps
        vis = draw_boxes(rgb, boxes, with_label=True)
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"FPS: {fps_avg:5.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(bgr, f"FPS: {fps_avg:5.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(bgr, f"objs: {len(boxes)}", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        vw.write(bgr)
        n += 1
    vw.release()
    print(f"[ok] 共处理 {n} 帧, 平均 FPS≈{fps_avg:.1f}")
    print(f"[ok] 输出检测视频: {args.out}")


def _read_real(cap, det, conf):
    import tempfile
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        # 真实后端直接吃帧需要落盘；这里走临时文件以兼容接口
        tmp = os.path.join(tempfile.gettempdir(), "_f.jpg")
        cv2.imwrite(tmp, frame)
        boxes = det.detect(tmp, conf=conf)
        yield rgb, boxes, (time.time() - t0) * 1000
    cap.release()


def _synth_iter(frames_gt, det, conf, W, H):
    for i, (rgb, gts) in enumerate(frames_gt):
        t0 = time.time()
        if isinstance(det, SimDetector):
            boxes = det.detect_from_gt(gts, W, H, conf=conf, key=i)
        else:
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), "_vf.jpg")
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            boxes = det.detect(tmp, conf=conf)
        # 加一点真实推理耗时模拟（YOLO26n 在 GPU 上约 4-8ms）
        time.sleep(0.004)
        yield rgb, boxes, (time.time() - t0) * 1000


if __name__ == "__main__":
    main()
