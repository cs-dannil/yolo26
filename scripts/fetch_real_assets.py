"""下载真实交通标志图片与真实街景背景，用于"结合真实图片"的数据模拟。

标志面：从维基共享资源（Wikimedia Commons）下载中国 GB 5768 标准交通标志
矢量图（自动渲染为带透明通道的 PNG）。命名遵循 GB 国标编号：
  禁 = 禁令(红圈)，示 = 指示(蓝)，警 = 警告(黄三角)，
  禁39-X=限速, 禁35-X=限高, 禁38-X=限重, 禁40-X=解除限速, 示14-X=最低限速。
背景：下载真实街景/真实照片，用于把真实标志合成到真实场景中。

  python scripts/fetch_real_assets.py            # 下载标志面+背景
  python scripts/fetch_real_assets.py --bg 16    # 指定背景数量

下载失败的类别会在后续合成时自动回退为矢量绘制，不影响流程。
"""
import argparse
import os
import sys
import time
import urllib.parse

import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classes import CLASSES  # noqa: E402

SIGN_DIR = os.path.join(ROOT, "data", "real_signs")
BG_DIR = os.path.join(ROOT, "data", "real_backgrounds")
UA = {"User-Agent": "Mozilla/5.0 (EduProject TT100K real-asset fetcher)"}
FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"

# 每个类别对应的维基共享资源候选文件（按优先级尝试），文件名遵循 GB 编号。
CANDIDATES = {
    # 禁令类（红圈）
    "pne": ["CN road sign 禁 2.svg", "China road sign 禁 2.svg"],
    "p10": ["CN road sign 禁 3.svg", "China road sign 禁 3.svg"],
    "p26": ["CN road sign 禁 4.svg"],
    "p3":  ["CN road sign 禁 6.svg"],
    "p12": ["CN road sign 禁 11.svg"],
    "p6":  ["CN road sign 禁 13.svg"],
    "p23": ["CN road sign 禁 21.svg"],
    "p19": ["CN road sign 禁 22.svg"],
    "p5":  ["CN road sign 禁 27.svg", "China road sign 禁 27.svg"],
    "p11": ["China road sign 禁 30.svg", "CN road sign 禁 30.svg"],
    "p27": ["China road sign 禁 31.svg", "CN road sign 禁 31.svg"],
    "pn":  ["China road sign 禁 33.svg", "China road sign 禁 32.svg",
            "CN road sign 禁 33.svg"],
    "po":  ["CN road sign 禁 1.svg"],
    "pg":  ["CN road sign 禁 41.svg", "CN road sign 禁 42.svg"],
    "pr40": ["CN road sign 禁 40-40.svg", "CN road sign 禁 40-60.svg"],
    # 限速 / 限高 / 限重 / 最低限速（数字类）
    "pl5":   ["CN road sign 禁 39-5.svg"],
    "pl20":  ["CN road sign 禁 39-20.svg"],
    "pl30":  ["CN road sign 禁 39-30.svg"],
    "pl40":  ["CN road sign 禁 39-40.svg"],
    "pl50":  ["CN road sign 禁 39-50.svg"],
    "pl60":  ["CN road sign 禁 39-60.svg"],
    "pl70":  ["CN road sign 禁 39-70.svg"],
    "pl80":  ["CN road sign 禁 39-80.svg"],
    "pl100": ["CN road sign 禁 39-100.svg"],
    "pl120": ["CN road sign 禁 39-120.svg"],
    "ph4":   ["CN road sign 禁 35-4.svg"],
    "ph4.5": ["CN road sign 禁 35-4.5.svg"],
    "ph5":   ["CN road sign 禁 35-5.svg"],
    "pm20":  ["CN road sign 禁 38-20.svg"],
    "pm30":  ["CN road sign 禁 38-30.svg"],
    "pm55":  ["CN road sign 禁 38-55.svg"],
    "il60":  ["CN road sign 示 14-60.svg"],
    "il80":  ["CN road sign 示 14-80.svg"],
    "il100": ["CN road sign 示 14-100.svg"],
    # 指示类（蓝）
    "i2":  ["China road sign 示 8.svg", "CN road sign 示 8.svg"],
    "i4":  ["China road sign 示 7.svg", "CN road sign 示 7.svg"],
    "i5":  ["China road sign 示 4.svg", "CN road sign 示 4.svg", "CN road sign 示 5.svg"],
    "io":  ["China road sign 示 2.svg", "CN road sign 示 2.svg", "CN road sign 示 3.svg"],
    "ip":  ["China road sign 示 9.svg", "CN road sign 示 9.svg", "CN road sign 注 1.svg"],
    # 警告类（黄三角）
    "w13": ["CN road sign 警 1-1.svg", "CN road sign 警 1.svg"],
    "w32": ["CN road sign 警 34-1.svg", "CN road sign 警 35.svg"],
    "w55": ["CN road sign 警 11-1.svg"],
    "w57": ["CN road sign 警 10-1.svg"],
    "w59": ["CN road sign 警 7-1.svg", "CN road sign 警 8-1.svg"],
    "wo":  ["CN road sign 警 24.svg", "CN road sign 警 28.svg"],
}

# 真实街景背景（维基共享资源，CC 许可）。下载失败的用 picsum 真实照片补足。
BG_COMMONS = [
    "千祥街禁止机动车驶入.jpg",
]


def _download(url, timeout=30):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 800 and \
               r.headers.get("content-type", "").startswith("image"):
                return r.content
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_signs():
    os.makedirs(SIGN_DIR, exist_ok=True)
    ok, fail = [], []
    for cls in CLASSES:
        out = os.path.join(SIGN_DIR, cls + ".png")
        if os.path.exists(out):
            ok.append(cls)
            continue
        got = False
        for fname in CANDIDATES.get(cls, []):
            url = FILEPATH + urllib.parse.quote(fname) + "?width=256"
            data = _download(url)
            if data:
                tmp = out + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                try:
                    im = Image.open(tmp).convert("RGBA")
                    im.save(out)
                    os.remove(tmp)
                    got = True
                    break
                except Exception:
                    if os.path.exists(tmp):
                        os.remove(tmp)
        (ok if got else fail).append(cls)
        print(f"  {'[ok ]' if got else '[miss]'} {cls}")
    print(f"\n标志面：成功 {len(ok)}/{len(CLASSES)}，回退绘制 {len(fail)}：{fail}")
    return ok, fail


# 无现成国标文件的数字变体：由同族真实模板改写数字派生（外观仍为真实标志样式）
DERIVE = {
    "il80":  ("il60", "80"), "il100": ("il60", "100"),
    "pl5":   ("pl50", "5"),  "pl120": ("pl100", "120"),
    "ph4":   ("pl50", "4m"), "ph4.5": ("pl50", "4.5m"), "ph5": ("pl50", "5m"),
    "pm20":  ("pl50", "20t"), "pm30": ("pl50", "30t"), "pm55": ("pl50", "55t"),
}


def derive_signs():
    """对没有现成国标矢量图的限高/限重/部分限速，从真实同族模板改写数字派生。"""
    from PIL import ImageDraw
    from src.draw import find_font
    made = []
    for cls, (base_cls, text) in DERIVE.items():
        out = os.path.join(SIGN_DIR, cls + ".png")
        base = os.path.join(SIGN_DIR, base_cls + ".png")
        if os.path.exists(out) or not os.path.exists(base):
            continue
        im = Image.open(base).convert("RGBA")
        W, H = im.size
        d = ImageDraw.Draw(im)
        # 在标志中心覆盖白色圆，清除原数字后写入新文本
        cx, cy, r = W / 2, H / 2, min(W, H) * 0.30
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
        font = find_font(int(min(W, H) * (0.30 if len(text) <= 2 else 0.22)))
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.text((cx - tw / 2 - tb[0], cy - th / 2 - tb[1]), text,
               fill=(20, 20, 20, 255), font=font)
        im.save(out)
        made.append(cls)
    if made:
        print(f"派生数字变体（基于真实模板）：{made}")
    return made


def fetch_backgrounds(n):
    os.makedirs(BG_DIR, exist_ok=True)
    count = 0
    for fname in BG_COMMONS:
        url = FILEPATH + urllib.parse.quote(fname) + "?width=960"
        data = _download(url)
        if data:
            p = os.path.join(BG_DIR, f"bg_{count:02d}.jpg")
            try:
                Image.open(_BytesIO(data)).convert("RGB").save(p, quality=85)
                count += 1
            except Exception:
                pass
    # 用 picsum 真实照片补足背景数量
    while count < n:
        data = _download(f"https://picsum.photos/seed/road{count}/960/720")
        if not data:
            break
        p = os.path.join(BG_DIR, f"bg_{count:02d}.jpg")
        try:
            Image.open(_BytesIO(data)).convert("RGB").save(p, quality=85)
            count += 1
            print(f"  [bg ] {p}")
        except Exception:
            break
    print(f"\n背景：共获取 {count} 张真实背景 -> {BG_DIR}")
    return count


def _BytesIO(b):
    import io
    return io.BytesIO(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", type=int, default=14, help="背景图数量")
    ap.add_argument("--signs-only", action="store_true")
    args = ap.parse_args()
    print("下载真实交通标志面 (Wikimedia Commons, GB 5768) ...")
    fetch_signs()
    derive_signs()
    if not args.signs_only:
        print("\n下载真实街景背景 ...")
        fetch_backgrounds(args.bg)
    print("\n[done] 真实素材已就绪。运行 make_dataset.py / make_eval_set.py 即用真实标志合成。")


if __name__ == "__main__":
    main()
