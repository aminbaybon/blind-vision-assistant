"""Desktop CPU timings. Not a phone measurement and not a 3 GB RAM result."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"


def once(path: Path, shape: tuple[int, ...]) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    data = np.random.default_rng(0).random(shape, dtype=np.float32)
    session.run(None, {name: data})
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        session.run(None, {name: data})
        samples.append(time.perf_counter() - started)
    return {
        "file": path.name,
        "input": list(shape),
        "seconds": [round(item, 4) for item in samples],
        "median_seconds": round(sorted(samples)[1], 4),
        "provider": "CPUExecutionProvider",
        "machine": "desktop Python, not an Android phone",
    }


def yolo_quant_check() -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = ASSETS / "yolo26n.onnx"
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "yolo26n.int8.onnx"
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
        import onnxruntime as ort

        raw = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        quant = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        name = raw.get_inputs()[0].name
        data = np.random.default_rng(1).random((1, 3, 416, 416), dtype=np.float32)
        base = raw.run(None, {name: data})[0].astype(np.float32)
        changed = quant.run(None, {name: data})[0].astype(np.float32)
        diff = np.max(np.abs(base - changed))
        return {
            "model": "yolo26n.onnx",
            "method": "dynamic INT8 weights, temporary file, not shipped",
            "quantized_bytes": target.stat().st_size,
            "max_abs_output_delta_on_one_random_tensor": round(float(diff), 6),
            "shipped": False,
            "reason": "One random tensor is not an accuracy test on labeled images.",
        }


def main() -> None:
    report = {
        "yolo26n_fp32": once(ASSETS / "yolo26n.onnx", (1, 3, 416, 416)),
        "midas_small_fp32": once(ASSETS / "midas_small.onnx", (1, 3, 256, 256)),
    }
    try:
        report["yolo_int8_probe"] = yolo_quant_check()
    except Exception as error:  # noqa: BLE001
        report["yolo_int8_probe"] = {"shipped": False, "error": str(error)}
    out = ROOT / "tests" / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "desktop_cpu_bench.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
