"""YOLO26 交通标志识别 - 图形界面（需求 9，PySide6）。

功能标签页：
  - 图片推理：选择/生成图片，运行检测并可视化
  - 视频推理：选择视频运行检测（带 FPS）
  - 数据集检查：展示 check_dataset 直方图与报告
  - 模型评测：展示 evaluate 指标与误检/漏检分析
  - ONNX 推理：onnxruntime 推理结果
  - 失败案例：展示失败案例画廊

  运行: python gui/app.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QTabWidget, QFileDialog, QTextEdit, QScrollArea, QFrame,
    QListWidget, QSplitter,
)
from PySide6.QtGui import QPixmap, QImage, QFont  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.detector import get_detector  # noqa: E402
from src.draw import draw_boxes        # noqa: E402
from src.classes import class_label    # noqa: E402

STYLE = """
QMainWindow { background: #1e1f2b; }
QWidget { color: #e6e6ee; font-size: 14px; }
QTabWidget::pane { border: 1px solid #33344a; border-radius: 8px; }
QTabBar::tab { background: #2a2b3d; padding: 10px 22px; margin: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #4c6ef5; color: white; }
QPushButton { background: #4c6ef5; color: white; border: none; padding: 9px 18px;
    border-radius: 8px; font-weight: bold; }
QPushButton:hover { background: #5c7cfa; }
QPushButton:pressed { background: #3b5bdb; }
QTextEdit, QListWidget { background: #252636; border: 1px solid #33344a; border-radius: 8px; }
QLabel#title { font-size: 22px; font-weight: bold; color: #4c6ef5; }
QLabel#img { background: #15151f; border: 1px dashed #33344a; border-radius: 8px; }
"""


def np_to_pixmap(arr, max_w=820):
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    pm = QPixmap.fromImage(img)
    if w > max_w:
        pm = pm.scaledToWidth(max_w, Qt.SmoothTransformation)
    return pm


class ImageTab(QWidget):
    def __init__(self):
        super().__init__()
        self.det = get_detector()
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_demo = QPushButton("生成示例图")
        self.btn_run = QPushButton("运行检测")
        for b in (self.btn_open, self.btn_demo, self.btn_run):
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        split = QSplitter(Qt.Horizontal)
        self.img_label = QLabel("请选择或生成一张交通标志图片")
        self.img_label.setObjectName("img")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(640, 460)
        self.info = QTextEdit(); self.info.setReadOnly(True); self.info.setMaximumWidth(330)
        split.addWidget(self.img_label); split.addWidget(self.info)
        lay.addWidget(split)
        self.path = None
        self.btn_open.clicked.connect(self.open)
        self.btn_demo.clicked.connect(self.demo)
        self.btn_run.clicked.connect(self.run)

    def open(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择图片", ROOT,
                                           "Images (*.jpg *.jpeg *.png *.bmp)")
        if p:
            self.path = p
            self.img_label.setPixmap(np_to_pixmap(np.array(Image.open(p).convert("RGB"))))
            self.info.setText(f"已加载: {os.path.basename(p)}")

    def demo(self):
        from src.sign_factory import make_image
        img, labels = make_image(min_signs=4, max_signs=6, small_ratio=0.2)
        p = os.path.join(ROOT, "outputs", "infer", "_gui_demo.jpg")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        img.save(p)
        with open(os.path.splitext(p)[0] + ".txt", "w") as f:
            for cid, xc, yc, bw, bh in labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        self.path = p
        self.img_label.setPixmap(np_to_pixmap(np.array(img)))
        self.info.setText("已生成示例图片，点击『运行检测』")

    def run(self):
        if not self.path:
            self.info.setText("请先打开或生成图片")
            return
        boxes = self.det.detect(self.path, conf=0.25)
        vis = draw_boxes(np.array(Image.open(self.path).convert("RGB")), boxes)
        self.img_label.setPixmap(np_to_pixmap(vis))
        txt = [f"检测到 {len(boxes)} 个目标:", ""]
        for b in boxes:
            txt.append(f"• {class_label(int(b['cls']))}\n  conf={b['conf']:.2f}")
        self.info.setText("\n".join(txt))


class GalleryTab(QWidget):
    """通用图片浏览页：用于数据集检查/评测/失败案例的成果展示。"""
    def __init__(self, folder, patterns=(".png", ".jpg")):
        super().__init__()
        self.folder = folder
        lay = QHBoxLayout(self)
        self.listw = QListWidget(); self.listw.setMaximumWidth(280)
        self.view = QLabel("选择左侧文件查看"); self.view.setObjectName("img")
        self.view.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea(); scroll.setWidget(self.view); scroll.setWidgetResizable(True)
        lay.addWidget(self.listw); lay.addWidget(scroll)
        self.files = []
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if os.path.splitext(f)[1].lower() in patterns:
                    self.files.append(os.path.join(folder, f))
                    self.listw.addItem(f)
        self.listw.currentRowChanged.connect(self.show)
        if self.files:
            self.listw.setCurrentRow(0)

    def show(self, i):
        if 0 <= i < len(self.files):
            self.view.setPixmap(np_to_pixmap(
                np.array(Image.open(self.files[i]).convert("RGB")), max_w=1000))


class TextTab(QWidget):
    def __init__(self, json_path, title):
        super().__init__()
        lay = QVBoxLayout(self)
        t = QLabel(title); t.setObjectName("title")
        lay.addWidget(t)
        box = QTextEdit(); box.setReadOnly(True)
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                box.setText(f.read())
        else:
            box.setText(f"未找到 {json_path}\n请先运行对应脚本生成结果。")
        lay.addWidget(box)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26 交通标志识别系统")
        self.resize(1180, 760)
        central = QWidget(); self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        header = QLabel("🚦 YOLO26 交通标志识别系统")
        header.setObjectName("title")
        header.setAlignment(Qt.AlignCenter)
        lay.addWidget(header)
        tabs = QTabWidget()
        tabs.addTab(ImageTab(), "图片推理")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/dataset_check")), "数据集检查")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/train")), "训练结果")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/eval")), "评测图表")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/eval/wrong")), "错误案例")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/failure_cases")), "失败案例")
        tabs.addTab(GalleryTab(os.path.join(ROOT, "outputs/onnx")), "ONNX推理")
        tabs.addTab(TextTab(os.path.join(ROOT, "outputs/eval/eval_report.json"), "评测报告(JSON)"), "评测报告")
        lay.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Noto Sans CJK SC", 10))
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
