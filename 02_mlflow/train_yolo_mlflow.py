"""Step 2: train YOLO briefly, track the run in MLflow, register the model with an alias.

Run from the 02_mlflow folder:
    python train_yolo_mlflow.py --epochs 2
Then open the UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import platform
import re
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO, settings

# Ultralytics has its own MLflow hook. Switch it off so you see exactly what we log ourselves.
settings.update({"mlflow": False})

MODEL_NAME = "yolo-lab"

p = argparse.ArgumentParser()
p.add_argument("--data", default="coco128.yaml", help="dataset config; coco128 downloads automatically (~7 MB)")
p.add_argument("--weights", default="yolo11n.pt", help="starting weights")
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--alias", default="champion")
args = p.parse_args()

# A database backend is needed for the Model Registry (a plain folder can't hold it).
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("yolo-lab")


def clean(name: str) -> str:
    """MLflow metric names allow letters, digits, _ - . / and spaces; keep it simple."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


with mlflow.start_run(run_name=f"yolo11n-e{args.epochs}-{args.imgsz}px") as run:
    # 1) Everything needed to reproduce the run: parameters, data, environment.
    mlflow.log_params(vars(args))
    import torch, ultralytics
    mlflow.set_tags({
        "data": args.data,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    })

    # 2) Train.
    t0 = time.time()
    model = YOLO(args.weights)
    train_res = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, device="cpu", workers=0,
        project="runs", name="train", exist_ok=True, plots=False, verbose=False,
    )
    mlflow.log_metric("train_seconds", round(time.time() - t0, 1))

    # 3) Evaluate the best checkpoint; log overall and per-class mAP.
    best = Path(train_res.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best))
    val = best_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device="cpu",
                         plots=False, verbose=False, project="runs", name="val", exist_ok=True)
    mlflow.log_metrics({"mAP50_95": float(val.box.map), "mAP50": float(val.box.map50)})
    names = val.names
    for cls_idx in val.box.ap_class_index:
        mlflow.log_metric(f"AP_{clean(names[int(cls_idx)])}", float(val.box.maps[int(cls_idx)]))

    # 4) Store the weights as an artifact of this run.
    mlflow.log_artifact(str(best), artifact_path="weights")
    source = f"{run.info.artifact_uri}/weights/best.pt"

    # 5) Register a new version and point the alias at it.
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME, description="Lab model: YOLO11n fine-tuned on coco128")
    except mlflow.exceptions.MlflowException:
        pass  # already exists from an earlier run
    mv = client.create_model_version(MODEL_NAME, source=source, run_id=run.info.run_id,
                                     tags={"mAP50_95": f"{val.box.map:.4f}"})
    client.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)

    print("\n=== Summary ===")
    print(f"run id:       {run.info.run_id}")
    print(f"mAP50-95:     {val.box.map:.4f}   mAP50: {val.box.map50:.4f}")
    print(f"registered:   {MODEL_NAME} version {mv.version}, alias '{args.alias}'")
    print("open the UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
