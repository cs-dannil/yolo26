"""绘制与字体相关的公共工具。

包含：matplotlib 中文字体设置、为 OpenCV 图像绘制中文检测框、
以及一组稳定的类别配色，供数据集制作 / 推理 / 评测可视化复用。
"""
import os
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .classes import NUM_CLASSES, class_label

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def find_font(size: int = 16) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def setup_matplotlib_cjk():
    """让 matplotlib 尽量支持中文，找不到中文字体时回退英文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    from matplotlib import rcParams

    for path in _FONT_CANDIDATES:
        if os.path.exists(path) and ("CJK" in path or "wqy" in path):
            try:
                fm.fontManager.addfont(path)
                rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    rcParams["axes.unicode_minus"] = False


def class_colors(n: int = NUM_CLASSES):
    """生成 n 个区分度较高的 RGB 颜色。"""
    colors = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.65 + 0.2 * ((i // 3) % 2)
        v = 0.85 + 0.1 * (i % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


_COLORS = class_colors()


def draw_boxes(img, boxes, with_label=True):
    """在图像上绘制检测框。

    img: numpy(H,W,3, RGB) 或 PIL.Image
    boxes: list of dict，键 x1,y1,x2,y2,cls,(conf 可选),(gt 可选)
    返回 numpy RGB 图像。
    """
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img.astype("uint8"))
    else:
        pil = img.convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = find_font(16)
    for b in boxes:
        cls = int(b["cls"])
        color = (0, 200, 0) if b.get("gt") else _COLORS[cls % len(_COLORS)]
        x1, y1, x2, y2 = (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))
        width = 3 if b.get("gt") else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if with_label:
            txt = class_label(cls).split(" ")[0]
            if "conf" in b:
                txt += f" {b['conf']:.2f}"
            tb = draw.textbbox((0, 0), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 4)
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            draw.text((x1 + 3, ty + 2), txt, fill=(255, 255, 255), font=font)
    return np.array(pil)
