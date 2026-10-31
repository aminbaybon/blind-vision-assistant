"""Check the Android MiDaS input against the ONNX graph and the official recipe.

Official tf/run_onnx.py reads RGB in 0..1, stretches to 256x256, and does not
call NormalizeImage. This file's graph already subtracts ImageNet mean and
divides by std. Feeding normalized values again is the bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "android" / "app" / "src" / "main" / "assets" / "midas_small.onnx"
sys.path.insert(0, str(ROOT))


def graph_normalizes_imagenet(path: Path) -> None:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    constants = []
    for node in model.graph.node[:4]:
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.t.dims:
                constants.append(numpy_helper.to_array(attribute.t).reshape(-1).tolist())
    assert model.graph.input[0].name == "0"
    dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 256, 256]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    assert np.allclose(constants[0], mean, atol=1e-6)
    assert np.allclose(constants[1], std, atol=1e-6)
    assert model.graph.node[0].op_type == "Constant"
    assert model.graph.node[1].op_type == "Sub"
    assert model.graph.node[3].op_type == "Div"


def android_input(rgb_0_1: np.ndarray) -> np.ndarray:
    """NCHW float RGB in 0..1. No second ImageNet normalization."""
    if rgb_0_1.shape[0] != 256 or rgb_0_1.shape[1] != 256:
        raise AssertionError("caller stretches to 256x256 first, as Bitmap does")
    chw = np.transpose(rgb_0_1, (2, 0, 1)).astype(np.float32)
    return chw.reshape(1, 3, 256, 256)


def double_normalized(rgb_0_1: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    changed = (rgb_0_1 - mean) / std
    return np.transpose(changed, (2, 0, 1)).astype(np.float32).reshape(1, 3, 256, 256)


def stretch(image: np.ndarray) -> np.ndarray:
    """Bilinear stretch, the Android Bitmap.createScaledBitmap default."""
    import cv2

    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    graph_normalizes_imagenet(MODEL)
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    source = np.zeros((180, 240, 3), dtype=np.float32)
    source[:, :, 0] = np.linspace(0.05, 0.95, 240, dtype=np.float32)
    source[:, :, 1] = 0.4
    source[:, :, 2] = 0.2
    rgb = stretch(source)
    assert rgb.shape == (256, 256, 3)
    once = android_input(rgb)
    twice = double_normalized(rgb)
    assert once.shape == (1, 3, 256, 256)
    assert float(once[0, 0].max()) <= 1.0
    red_pixel = once[0, 0, 10, 10]
    assert abs(float(red_pixel) - float(rgb[10, 10, 0])) < 1e-6
    depth_once = session.run(None, {name: once})[0]
    depth_twice = session.run(None, {name: twice})[0]
    assert depth_once.shape[-2:] == (256, 256)
    delta = float(np.max(np.abs(depth_once.astype(np.float32) - depth_twice.astype(np.float32))))
    if delta < 1e-3:
        raise SystemExit("double normalization did not change the depth map")
    print(f"ok midas input; double-norm max abs delta {delta:.4f}")


if __name__ == "__main__":
    main()
