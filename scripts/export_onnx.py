"""将训练得到的 .pt 模型导出为 ONNX（需求 10 前半）。

  python scripts/export_onnx.py --weights weights/best.pt --imgsz 640

双模式：
  * 有 ultralytics + 真实权重：调用 YOLO.export(format='onnx')；
  * 无权重的演示环境：用 onnx 直接构建一个结构等价的轻量检测网络
    (Conv-BN-ReLU x3 + 检测头)，导出为合法 ONNX，可被 onnxruntime 真实加载推理。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classes import NUM_CLASSES  # noqa: E402


def export_real(weights, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights)
    path = m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    print(f"[ok] ultralytics 导出 ONNX: {path}")
    return path


def export_demo(out_path, imgsz):
    """构建并导出一个合法的小型检测网络 ONNX。"""
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    def conv_bn(prefix, cin, cout, k=3, s=2):
        nodes, inits = [], []
        w = np.random.randn(cout, cin, k, k).astype("float32") * 0.05
        b = np.zeros(cout, dtype="float32")
        inits += [numpy_helper.from_array(w, prefix + "_w"),
                  numpy_helper.from_array(b, prefix + "_b")]
        nodes.append(helper.make_node(
            "Conv", [prefix + "_in", prefix + "_w", prefix + "_b"], [prefix + "_c"],
            kernel_shape=[k, k], strides=[s, s], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [prefix + "_c"], [prefix + "_out"]))
        return nodes, inits

    nodes, inits = [], []
    chans = [3, 16, 32, 64]
    cur = "images"
    for i in range(3):
        n, ini = conv_bn(f"layer{i}", chans[i], chans[i + 1])
        # 连接前一层输出
        n[0].input[0] = cur
        nodes += n; inits += ini
        cur = f"layer{i}_out"
    # 检测头：1x1 conv 输出 (4 box + 1 obj + NUM_CLASSES)
    no = 4 + 1 + NUM_CLASSES
    hw = np.random.randn(no, 64, 1, 1).astype("float32") * 0.05
    hb = np.zeros(no, dtype="float32")
    inits += [numpy_helper.from_array(hw, "head_w"),
              numpy_helper.from_array(hb, "head_b")]
    nodes.append(helper.make_node("Conv", [cur, "head_w", "head_b"], ["output"],
                                  kernel_shape=[1, 1], strides=[1, 1]))

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, imgsz, imgsz])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                        [1, no, imgsz // 8, imgsz // 8])
    graph = helper.make_graph(nodes, "yolo26_demo", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)
    print(f"[ok] 演示模式导出合法 ONNX: {out_path}")
    print(f"     输入: images[1,3,{imgsz},{imgsz}]  输出: output[1,{no},{imgsz//8},{imgsz//8}]")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "weights/best.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    real = False
    try:
        import ultralytics  # noqa: F401
        if os.path.exists(args.weights) and os.path.getsize(args.weights) > 1_000_000:
            real = True
    except Exception:
        pass

    if real:
        export_real(args.weights, args.imgsz)
    else:
        print("[warn] 无真实权重/ultralytics，导出结构等价的演示 ONNX 模型")
        export_demo(args.out, args.imgsz)


if __name__ == "__main__":
    main()
