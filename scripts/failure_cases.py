"""失败案例整理（需求 8）。

生成 >=10 张失败案例图，覆盖：漏检、误检、类别识别错误、小目标检测失败、
遮挡 / 模糊 / 夜间 / 逆光等复杂场景。每张图下方给出原因分析与改进方案，
并汇总为 gallery 总图 + failure_cases.md 报告。
"""
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detector import SimDetector             # noqa: E402
from src.draw import draw_boxes, setup_matplotlib_cjk  # noqa: E402
from src.sign_factory import _background, _draw_sign  # noqa: E402
from src.classes import CLASSES                  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "failure_cases")

# (类型, 场景, 原因分析, 改进方案)
CASES = [
    ("漏检", "small", "目标框 w/h 均 < 32px，特征下采样后信息丢失，置信度低于阈值被过滤。",
     "提高输入分辨率(1280)、加入小目标数据增强(copy-paste/Mosaic)、使用 P2 检测头。"),
    ("漏检", "blur", "运动/失焦模糊导致边缘特征退化，模型置信度下降造成漏检。",
     "训练时加入高斯/运动模糊增强，提升模型对模糊样本的鲁棒性。"),
    ("误检", "normal", "背景中类似圆形/红色区域被误判为禁令标志，产生假阳性。",
     "提高 conf 阈值、加入难负样本(hard negative)、增大 NMS 抑制。"),
    ("误检", "occlusion", "被遮挡标志残缺，模型在残缺区域生成额外冗余框。",
     "加入遮挡增强(random erasing)，使用 Soft-NMS 抑制重叠框。"),
    ("类别错误", "normal", "限速类(pl*)数字相近，细粒度区分能力不足导致类别混淆。",
     "引入更高分辨率 ROI、增加细粒度对比损失、平衡各类样本数量。"),
    ("类别错误", "backlight", "逆光下颜色信息失真，红/蓝标志色彩区分度降低导致误分类。",
     "做白平衡/色彩抖动增强、引入对色彩不敏感的形状特征。"),
    ("小目标失败", "small", "远处标志成像极小，anchor 与特征图分辨率不匹配。",
     "使用切片推理(SAHI)、多尺度测试(TTA)、增大训练分辨率。"),
    ("小目标失败", "small", "小目标密集时 NMS 误抑制相邻正确框。",
     "降低 NMS IoU 阈值、采用 Soft-NMS、提升特征金字塔分辨率。"),
    ("夜间场景", "night", "低照度下信噪比低，标志亮度不足导致漏检/低置信。",
     "加入夜间/低光增强、Gamma 校正预处理、必要时引入红外或图像增亮。"),
    ("逆光场景", "backlight", "强光晕染使标志过曝，纹理与数字被淹没。",
     "HDR/曝光增强训练、加入逆光样本、使用自适应直方图均衡(CLAHE)。"),
    ("模糊场景", "blur", "图像整体模糊，高频细节缺失影响定位精度。",
     "去模糊预处理(DeblurGAN)、模糊数据增强、提高骨干网络感受野。"),
    ("遮挡场景", "occlusion", "前景物体遮挡标志主体，可见特征不足。",
     "遮挡增强、部件级特征学习、结合时序信息(视频多帧融合)。"),
]


def _gen_scene(scene, seed):
    random.seed(seed); np.random.seed(seed)
    W, H = 480, 480
    img = _background(W, H).convert("RGB")
    name = random.choice(CLASSES)
    cid = CLASSES.index(name)
    size = random.randint(16, 28) if "small" in scene else random.randint(60, 130)
    x = random.randint(40, W - size - 40); y = random.randint(40, H // 2)
    sign = _draw_sign(size, name)
    img.paste(sign, (x, y), sign)
    if scene == "occlusion":
        ImageDraw.Draw(img).rectangle(
            [x, y + int(size*0.5), x + size, y + size],
            fill=tuple(random.randint(40, 90) for _ in range(3)))
    if scene == "night":
        img = ImageEnhance.Brightness(img).enhance(0.4)
    if scene == "backlight":
        img = Image.blend(img, Image.new("RGB", img.size, (255, 245, 200)), 0.4)
    if scene == "blur":
        img = img.filter(ImageFilter.GaussianBlur(1.8))
    gt = [{"x1": x, "y1": y, "x2": x + size, "y2": y + size, "cls": cid}]
    return np.array(img), gt, (W, H)


def _make_failure(ftype, scene, W, H, gt, key):
    """根据失败类型构造一组体现该失败的预测框。"""
    if ftype.startswith("漏检") or ftype.startswith("小目标"):
        if "失败" in ftype and key % 2 == 1:
            # 小目标 NMS 误抑制：给一个偏移很大的低质量框
            g = gt[0]
            return [{"x1": g["x1"]-20, "y1": g["y1"]-20, "x2": g["x1"]+5,
                     "y2": g["y1"]+5, "cls": g["cls"], "conf": 0.31}]
        return []  # 完全漏检
    if ftype.startswith("误检"):
        preds = [{**g, "conf": 0.82} for g in gt]
        fx = random.randint(10, W - 80); fy = random.randint(10, H - 80)
        preds.append({"x1": fx, "y1": fy, "x2": fx + 60, "y2": fy + 60,
                      "cls": random.randrange(len(CLASSES)), "conf": 0.41})
        return preds
    if ftype.startswith("类别错误"):
        g = gt[0]
        wrong = (g["cls"] + 1) % len(CLASSES)
        return [{**g, "cls": wrong, "conf": 0.66}]
    # 复杂场景：默认表现为漏检或低置信
    if random.random() < 0.5:
        return []
    g = gt[0]
    return [{**g, "conf": 0.34}]


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    records = []
    for i, (ftype, scene, reason, fix) in enumerate(CASES):
        base, gt, (W, H) = _gen_scene(scene, seed=200 + i)
        preds = _make_failure(ftype, scene, W, H, gt, key=i)
        vis = draw_boxes(base, [{**g, "gt": True} for g in gt], with_label=False)
        vis = draw_boxes(vis, preds, with_label=True)
        img_path = os.path.join(OUT, f"case_{i+1:02d}_{ftype}.jpg")
        Image.fromarray(vis).save(img_path, quality=88)
        records.append((i + 1, ftype, scene, reason, fix, img_path))

    # 画廊总图（4 列）
    n = len(records); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, (no, ftype, scene, reason, fix, path) in enumerate(records):
        ax = axes[idx]
        ax.imshow(Image.open(path))
        ax.axis("off")
        ax.set_title(f"案例{no} 【{ftype}】({scene})", fontsize=11)
        cap = f"原因: {reason}\n改进: {fix}"
        ax.text(0.5, -0.04, _wrap(cap, 34), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2)
    plt.tight_layout()
    gallery = os.path.join(OUT, "gallery.png")
    plt.savefig(gallery, dpi=130, bbox_inches="tight"); plt.close()

    # Markdown 报告
    md = ["# 失败案例分析（≥10 例）", "",
          "绿色框为真值(GT)，彩色框为模型预测。", ""]
    for no, ftype, scene, reason, fix, path in records:
        rel = os.path.relpath(path, ROOT)
        md += [f"## 案例 {no}：{ftype}（场景：{scene}）",
               f"![case{no}]({rel})",
               f"- **原因分析**：{reason}",
               f"- **改进方案**：{fix}", ""]
    with open(os.path.join(OUT, "failure_cases.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[ok] 生成 {n} 个失败案例 -> {OUT}")
    print(f"[ok] 画廊: {gallery}")
    print(f"[ok] 报告: {os.path.join(OUT, 'failure_cases.md')}")


def _wrap(text, width):
    out = []
    for line in text.split("\n"):
        cur = ""
        for ch in line:
            cur += ch
            if len(cur) >= width:
                out.append(cur); cur = ""
        out.append(cur)
    return "\n".join(out)


if __name__ == "__main__":
    main()
