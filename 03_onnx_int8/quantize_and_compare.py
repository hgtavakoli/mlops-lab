"""Step 3: load the 'champion' model from the registry, export to ONNX, quantize to INT8,
and compare latency and per-class mAP.

Run from the 03_onnx_int8 folder (after step 2):
    python quantize_and_compare.py
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import onnxruntime as ort
from mlflow import MlflowClient
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO, settings

settings.update({"mlflow": False})

p = argparse.ArgumentParser()
p.add_argument("--tracking", default="sqlite:///../02_mlflow/mlflow.db")
p.add_argument("--model-name", default="yolo-lab")
p.add_argument("--alias", default="champion")
p.add_argument("--data", default="coco128.yaml")
p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--calib-images", type=int, default=64)
p.add_argument("--runs", type=int, default=100, help="timed runs per model for latency")
args = p.parse_args()

mlflow.set_tracking_uri(args.tracking)
client = MlflowClient()

# 1) Get the model the registry calls 'champion' -- not "whatever best.pt is on disk".
mv = client.get_model_version_by_alias(args.model_name, args.alias)
weights = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path="work")
print(f"[1] {args.model_name} v{mv.version} (alias '{args.alias}') from run {mv.run_id}")

# 2) Export to ONNX (FP32).
fp32 = Path(YOLO(weights).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False))
fp32_pre = fp32.with_name("model_fp32_pre.onnx")
quant_pre_process(str(fp32), str(fp32_pre))  # shape inference + cleanup recommended before quantizing
print(f"[2] exported {fp32.name}")


# 3) Calibration data: real images, preprocessed exactly like inference (letterbox, RGB, /255).
def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def to_tensor(path, size):
    img = letterbox(cv2.imread(path), size)[:, :, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


ds_images = sorted(glob.glob(str(Path(settings["datasets_dir"]) / "coco128" / "images" / "train2017" / "*.jpg")))
calib_paths = ds_images[: args.calib_images]
input_name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.it = iter([{input_name: to_tensor(pth, args.imgsz)} for pth in paths])

    def get_next(self):
        return next(self.it, None)


# 4) Two INT8 variants:
#    A) everything quantized
#    B) the detection head (last block) kept in FP32 -- a common fix when INT8 hurts accuracy
import onnx

graph_nodes = [n.name for n in onnx.load(str(fp32_pre)).graph.node]
head_prefix = sorted({n.split("/")[1] for n in graph_nodes if n.startswith("/model.")}, key=lambda s: int(s.split(".")[1]))[-1]
head_nodes = [n for n in graph_nodes if n.startswith(f"/{head_prefix}/")]

variants = {}
for tag, exclude in [("int8_all", []), ("int8_head_fp32", head_nodes)]:
    out = fp32.with_name(f"model_{tag}.onnx")
    quantize_static(str(fp32_pre), str(out), Reader(calib_paths),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    nodes_to_exclude=exclude)
    variants[tag] = out
print(f"[3] quantized with {len(calib_paths)} calibration images; head kept in FP32 in variant B: /{head_prefix}/ ({len(head_nodes)} nodes)")


# 5) Latency: same input, warm-up first, then p50 / p99 over N runs. Batch size 1, like real-time perception.
def latency_ms(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = to_tensor(ds_images[0], args.imgsz)
    for _ in range(10):
        sess.run(None, {input_name: x})
    ts = []
    for _ in range(args.runs):
        t = time.perf_counter()
        sess.run(None, {input_name: x})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


# 6) Accuracy: the same validation for every model, per class.
def evaluate(path, name):
    res = YOLO(str(path), task="detect").val(data=args.data, imgsz=args.imgsz, batch=1, device="cpu",
                                              plots=False, verbose=False, project="runs", name=f"val_{name}", exist_ok=True)
    per_class = {res.names[int(c)]: float(res.box.maps[int(c)]) for c in res.box.ap_class_index}
    return float(res.box.map), float(res.box.map50), per_class


models = {"onnx_fp32": fp32, **variants}
results = {}
for name, path in models.items():
    p50, p99 = latency_ms(path)
    m, m50, per_class = evaluate(path, name)
    size_mb = Path(path).stat().st_size / 1e6
    results[name] = {"mAP50_95": m, "mAP50": m50, "p50_ms": p50, "p99_ms": p99, "size_MB": size_mb, "per_class": per_class}

# 7) Report.
print("\n=== Overall (batch 1, CPU, imgsz %d) ===" % args.imgsz)
print(f"{'model':<16}{'mAP50-95':>10}{'mAP50':>8}{'p50 ms':>9}{'p99 ms':>9}{'MB':>7}")
for name, r in results.items():
    print(f"{name:<16}{r['mAP50_95']:>10.3f}{r['mAP50']:>8.3f}{r['p50_ms']:>9.1f}{r['p99_ms']:>9.1f}{r['size_MB']:>7.1f}")

base = results["onnx_fp32"]["per_class"]
for tag in variants:
    drops = sorted(((base[c] - results[tag]["per_class"].get(c, 0.0), c) for c in base), reverse=True)
    print(f"\n=== {tag}: classes that lost the most mAP50-95 vs FP32 ===")
    for d, c in drops[:8]:
        print(f"  {c:<16} {base[c]:.3f} -> {results[tag]['per_class'].get(c, 0.0):.3f}  (-{d:.3f})")

# 8) Log the comparison to MLflow, linked to the same registered model version.
mlflow.set_experiment("yolo-lab")
with mlflow.start_run(run_name=f"onnx-int8-v{mv.version}"):
    mlflow.set_tags({"model_version": mv.version, "source_run": mv.run_id})
    for name, r in results.items():
        mlflow.log_metrics({f"{name}_{k}": v for k, v in r.items() if k != "per_class"})
    Path("work/results.json").write_text(json.dumps(results, indent=2))
    mlflow.log_artifact("work/results.json")
print("\nlogged to MLflow experiment 'yolo-lab'")
