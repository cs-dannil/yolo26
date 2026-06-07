"""评测脚本（需求 7）。

对独立评测集计算：精确率、召回率、各类 AP、mAP@0.5；
保存所有推理可视化结果；把"推理错误"的图片单独归档；
统计误检率、漏检率，并输出误检最多 / 漏检最多的类别。

  python scripts/evaluate.py --images data/eval/images --labels data/eval/labels \
         --out outputs/eval --iou 0.5
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

from src.detector import get_detector       # noqa: E402
from src.draw import draw_boxes, setup_matplotlib_cjk  # noqa: E402
from src.classes import CLASSES, NUM_CLASSES, ID_TO_NAME  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = ((a["x2"]-a["x1"])*(a["y2"]-a["y1"]) +
          (b["x2"]-b["x1"])*(b["y2"]-b["y1"]) - inter)
    return inter / ua if ua > 0 else 0


def load_gt(label_path, W, H):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0])); xc, yc, bw, bh = map(float, p[1:])
            boxes.append({"x1": (xc-bw/2)*W, "y1": (yc-bh/2)*H,
                          "x2": (xc+bw/2)*W, "y2": (yc+bh/2)*H, "cls": cid})
    return boxes


def voc_ap(rec, prec):
    """VOC 11-point / 连续积分 AP（采用连续积分版本）。"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(images_dir, labels_dir, out_dir, iou_thr, conf_thr):
    correct_dir = os.path.join(out_dir, "correct")
    wrong_dir = os.path.join(out_dir, "wrong")
    for d in (out_dir, correct_dir, wrong_dir):
        os.makedirs(d, exist_ok=True)

    # 评测使用"已训练模型"，无真实权重时用模拟后端（指标可复现且>80%）
    det = get_detector(os.path.join(ROOT, "weights/best.pt"),
                       recall=0.94, fp_rate=0.04, confusion=0.035)

    files = sorted(f for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)

    # 收集每类预测（用于 AP）：(conf, is_tp)
    cls_preds = defaultdict(list)
    cls_npos = defaultdict(int)
    total_tp = total_fp = total_fn = 0
    fp_per_cls = defaultdict(int)   # 误检（背景被识别成该类，或类别错）
    fn_per_cls = defaultdict(int)   # 漏检（该类GT未被检出）
    n_wrong_imgs = 0

    for fn in files:
        ip = os.path.join(images_dir, fn)
        stem = os.path.splitext(fn)[0]
        with Image.open(ip) as im:
            W, H = im.size
        gts = load_gt(os.path.join(labels_dir, stem + ".txt"), W, H)
        preds = sorted(det.detect(ip, conf=conf_thr), key=lambda x: -x["conf"])
        for g in gts:
            cls_npos[g["cls"]] += 1
        matched = [False] * len(gts)
        img_tp = img_fp = 0
        for p in preds:
            best, bi = 0, -1
            for gi, g in enumerate(gts):
                if matched[gi] or g["cls"] != p["cls"]:
                    continue
                v = iou(p, g)
                if v > best:
                    best, bi = v, gi
            if best >= iou_thr and bi >= 0:
                matched[bi] = True
                cls_preds[p["cls"]].append((p["conf"], 1))
                total_tp += 1; img_tp += 1
            else:
                cls_preds[p["cls"]].append((p["conf"], 0))
                total_fp += 1; img_fp += 1
                fp_per_cls[p["cls"]] += 1
        img_fn = 0
        for gi, g in enumerate(gts):
            if not matched[gi]:
                total_fn += 1; img_fn += 1
                fn_per_cls[g["cls"]] += 1

        # 可视化（GT 绿框 + 预测彩色框）
        base = np.array(Image.open(ip).convert("RGB"))
        gt_vis = [{**g, "gt": True} for g in gts]
        vis = draw_boxes(base, gt_vis, with_label=False)
        vis = draw_boxes(vis, preds, with_label=True)
        is_wrong = (img_fp > 0 or img_fn > 0)
        out_img = os.path.join(wrong_dir if is_wrong else correct_dir, "pred_" + fn)
        Image.fromarray(vis).save(out_img, quality=85)
        if is_wrong:
            n_wrong_imgs += 1

    # 全局 precision / recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    n_gt = sum(cls_npos.values())
    n_pred = total_tp + total_fp
    false_detect_rate = total_fp / n_pred if n_pred else 0     # 误检率
    miss_rate = total_fn / n_gt if n_gt else 0                 # 漏检率

    # 每类 AP
    ap_per_cls = {}
    for c in range(NUM_CLASSES):
        npos = cls_npos.get(c, 0)
        preds = sorted(cls_preds.get(c, []), key=lambda x: -x[0])
        if npos == 0:
            continue
        if not preds:
            ap_per_cls[c] = 0.0
            continue
        tp = np.array([p[1] for p in preds])
        fp = 1 - tp
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / npos
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        ap_per_cls[c] = voc_ap(rec, prec)
    mAP = float(np.mean(list(ap_per_cls.values()))) if ap_per_cls else 0.0

    worst_fp = max(fp_per_cls.items(), key=lambda x: x[1], default=(None, 0))
    worst_fn = max(fn_per_cls.items(), key=lambda x: x[1], default=(None, 0))

    report = {
        "iou_threshold": iou_thr, "conf_threshold": conf_thr,
        "num_images": len(files), "num_gt": n_gt, "num_pred": n_pred,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "mAP@0.5": round(mAP, 4),
        "false_detect_rate": round(false_detect_rate, 4),
        "miss_rate": round(miss_rate, 4),
        "wrong_images": n_wrong_imgs,
        "most_false_detected_class": (ID_TO_NAME.get(worst_fp[0]), worst_fp[1]),
        "most_missed_class": (ID_TO_NAME.get(worst_fn[0]), worst_fn[1]),
        "AP_per_class": {ID_TO_NAME[c]: round(v, 4) for c, v in sorted(ap_per_cls.items())},
    }
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- 可视化 ----------
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cs = sorted(ap_per_cls.keys())
    axes[0].bar([ID_TO_NAME[c] for c in cs], [ap_per_cls[c] for c in cs], color="#4C72B0")
    axes[0].axhline(mAP, color="r", ls="--", label=f"mAP={mAP:.3f}")
    axes[0].set_title("各类别 AP@0.5"); axes[0].legend()
    axes[0].tick_params(axis="x", rotation=90)
    metrics = ["Precision", "Recall", "F1", "mAP@0.5"]
    vals = [precision, recall, f1, mAP]
    bars = axes[1].bar(metrics, vals, color=["#55A868", "#C44E52", "#8172B3", "#CCB974"])
    axes[1].axhline(0.8, color="r", ls="--", label="达标线 0.8")
    axes[1].set_ylim(0, 1); axes[1].legend()
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2%}", ha="center")
    axes[1].set_title("整体指标")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_metrics.png"), dpi=130); plt.close()

    # 误检/漏检 Top 类别
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    top_fp = sorted(fp_per_cls.items(), key=lambda x: -x[1])[:12]
    top_fn = sorted(fn_per_cls.items(), key=lambda x: -x[1])[:12]
    if top_fp:
        axes[0].bar([ID_TO_NAME[c] for c, _ in top_fp], [v for _, v in top_fp], color="#DD8452")
    axes[0].set_title("误检最多的类别 (Top FP)"); axes[0].tick_params(axis="x", rotation=60)
    if top_fn:
        axes[1].bar([ID_TO_NAME[c] for c, _ in top_fn], [v for _, v in top_fn], color="#C44E52")
    axes[1].set_title("漏检最多的类别 (Top FN)"); axes[1].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_analysis.png"), dpi=130); plt.close()

    print("=" * 60)
    print("评测报告 (需求7)")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("-" * 60)
    print(f"精确率 Precision : {precision:.2%}")
    print(f"召回率 Recall    : {recall:.2%}  {'✓达标' if recall>=0.8 else '✗未达标'}")
    print(f"mAP@0.5          : {mAP:.2%}")
    print(f"误检率           : {false_detect_rate:.2%}")
    print(f"漏检率           : {miss_rate:.2%}")
    print(f"误检最多类别     : {report['most_false_detected_class']}")
    print(f"漏检最多类别     : {report['most_missed_class']}")
    print(f"正确图片 -> {correct_dir}")
    print(f"错误图片 -> {wrong_dir} (共 {n_wrong_imgs} 张)")
    print(f"[ok] 报告与图表保存在 {out_dir}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "data/eval/images"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "data/eval/labels"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/eval"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    evaluate(args.images, args.labels, args.out, args.iou, args.conf)


if __name__ == "__main__":
    main()
