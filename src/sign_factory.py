"""合成交通标志图像生成器。

在没有 GPU、无法在线下载数 GB 真实 TT100K 的环境下，本模块用于生成
与 TT100K 类别体系一致、外观接近真实交通标志的合成样本（红圈禁令、
蓝圈指示、黄三角警告等），用来跑通整条数据→检查→训练→评测链路。
真实使用时把 data/tt100k 换成官方数据即可，类别定义完全一致。
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from .draw import find_font
from .classes import CLASSES, CLASS_ZH

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SIGN_DIR = os.path.join(_HERE, "data", "real_signs")
REAL_BG_DIR = os.path.join(_HERE, "data", "real_backgrounds")

_REAL_SIGN_CACHE = {}
_REAL_BG_LIST = None


def has_real_assets():
    return os.path.isdir(REAL_SIGN_DIR) and len(os.listdir(REAL_SIGN_DIR)) > 0


def _load_real_sign(name):
    """加载真实标志面 PNG（RGBA），不存在返回 None。带缓存。"""
    if name in _REAL_SIGN_CACHE:
        return _REAL_SIGN_CACHE[name]
    p = os.path.join(REAL_SIGN_DIR, name + ".png")
    im = None
    if os.path.exists(p):
        try:
            im = Image.open(p).convert("RGBA")
        except Exception:
            im = None
    _REAL_SIGN_CACHE[name] = im
    return im


def _real_sign_render(size, name):
    """返回真实标志缩放到 size 的 RGBA 图，附带轻微增强；无真实素材时返回 None。"""
    im = _load_real_sign(name)
    if im is None:
        return None
    s = im.resize((size, size), Image.LANCZOS)
    if random.random() < 0.4:  # 轻微旋转，贴近真实拍摄角度
        s = s.rotate(random.uniform(-8, 8), expand=False, resample=Image.BICUBIC)
    if random.random() < 0.5:  # 亮度抖动
        s = ImageEnhance.Brightness(s).enhance(random.uniform(0.8, 1.15))
    return s


def render_sign(size, name):
    """统一标志渲染：优先真实标志面，缺失时回退矢量绘制。"""
    return _real_sign_render(size, name) or _draw_sign(size, name)


def _real_background(w, h):
    """随机选一张真实照片作背景并裁剪到 w×h；无真实素材时返回 None。"""
    global _REAL_BG_LIST
    if _REAL_BG_LIST is None:
        _REAL_BG_LIST = []
        if os.path.isdir(REAL_BG_DIR):
            _REAL_BG_LIST = [os.path.join(REAL_BG_DIR, f)
                             for f in os.listdir(REAL_BG_DIR)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not _REAL_BG_LIST:
        return None
    try:
        im = Image.open(random.choice(_REAL_BG_LIST)).convert("RGB")
    except Exception:
        return None
    # 等比缩放后中心裁剪
    scale = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1), Image.LANCZOS)
    left = random.randint(0, max(0, im.width - w))
    top = random.randint(0, max(0, im.height - h))
    return im.crop((left, top, left + w, top + h))


def _sign_group(name: str) -> str:
    if name.startswith("w"):
        return "warn"        # 黄底三角 警告
    if name.startswith(("i", "il")):
        return "indi"        # 蓝底圆 指示
    return "prohibit"        # 红圈 禁令/限速/限高/限重


def _draw_sign(size: int, name: str) -> Image.Image:
    """绘制单个标志图（RGBA），size 为边长像素。"""
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    group = _sign_group(name)
    pad = max(1, s // 12)
    font = find_font(max(8, int(s * 0.34)))

    # 标志上的文字：限速/限高/限重显示数字，其余显示代号尾部
    text = ""
    digits = "".join(ch for ch in name if ch.isdigit() or ch == ".")
    if name.startswith(("pl", "il", "ph", "pm", "pr")) and digits:
        text = digits

    if group == "warn":
        d.polygon([(s // 2, pad), (pad, s - pad), (s - pad, s - pad)],
                  fill=(255, 210, 0, 255), outline=(0, 0, 0, 255))
        d.line([(s // 2, pad + s // 8), (s // 2, s - pad - s // 8)],
               fill=(0, 0, 0, 255), width=max(2, s // 18))
    elif group == "indi":
        d.ellipse([pad, pad, s - pad, s - pad],
                  fill=(0, 70, 200, 255), outline=(255, 255, 255, 255),
                  width=max(2, s // 22))
        if text:
            _center_text(d, s, text, font, (255, 255, 255, 255))
    else:  # prohibit
        d.ellipse([pad, pad, s - pad, s - pad],
                  fill=(255, 255, 255, 255), outline=(210, 20, 20, 255),
                  width=max(3, s // 9))
        if name in ("pn", "pne", "p10", "p3", "p6", "p12", "p26"):
            d.line([(pad + s // 6, s - pad - s // 6),
                    (s - pad - s // 6, pad + s // 6)],
                   fill=(210, 20, 20, 255), width=max(3, s // 12))
        if text:
            _center_text(d, s, text, font, (20, 20, 20, 255))
    return img


def _center_text(draw, s, text, font, color):
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((s - tw) / 2 - tb[0], (s - th) / 2 - tb[1]), text,
              fill=color, font=font)


def _background(w: int, h: int) -> Image.Image:
    """生成接近街景的渐变背景 + 道路/建筑色块。"""
    top = np.array([random.randint(120, 200), random.randint(150, 210),
                    random.randint(190, 235)], dtype=np.float32)
    bot = np.array([random.randint(70, 130)] * 3, dtype=np.float32)
    grad = (np.linspace(0, 1, h)[:, None, None] * (bot - top) + top)
    arr = np.repeat(grad, w, axis=1).astype("uint8")
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 7)):
        x0 = random.randint(0, w)
        bw = random.randint(w // 12, w // 4)
        bh = random.randint(h // 6, h // 2)
        c = tuple(random.randint(90, 170) for _ in range(3))
        d.rectangle([x0, h - bh, x0 + bw, h], fill=c)
    d.rectangle([0, int(h * 0.78), w, h], fill=(70, 70, 75))  # 路面
    return img.filter(ImageFilter.GaussianBlur(0.6))


def make_image(w=640, h=640, min_signs=1, max_signs=4,
               classes=None, small_ratio=0.35, seed=None):
    """生成一张图及其 YOLO 标注。

    返回 (PIL.Image RGB, labels)；labels 为 [(cls_id, xc, yc, bw, bh), ...]
    （归一化坐标）。small_ratio 控制小目标(<32px)出现概率。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
    classes = classes or CLASSES
    img = _real_background(w, h) or _background(w, h)
    n = random.randint(min_signs, max_signs)
    labels = []
    placed = []
    for _ in range(n):
        name = random.choice(classes)
        cid = CLASSES.index(name)
        if random.random() < small_ratio:
            size = random.randint(14, 30)          # 小目标
        else:
            size = random.randint(40, min(150, w // 4))
        # 避免重叠太多
        ok = False
        for _try in range(15):
            x = random.randint(2, w - size - 2)
            y = random.randint(2, int(h * 0.7))
            box = (x, y, x + size, y + size)
            if all(_iou(box, p) < 0.05 for p in placed):
                ok = True
                break
        if not ok:
            continue
        placed.append(box)
        sign = _real_sign_render(size, name) or _draw_sign(size, name)
        if size < 32 and random.random() < 0.5:
            sign = sign.filter(ImageFilter.GaussianBlur(0.6))
        img.paste(sign, (x, y), sign)
        xc = (x + size / 2) / w
        yc = (y + size / 2) / h
        labels.append((cid, xc, yc, size / w, size / h))
    # 轻微整体噪声/模糊，贴近真实采集
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img.convert("RGB"), labels


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0
