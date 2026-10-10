"""Smoke test: the training script from step 1 runs end to end and learns something."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_training_runs_and_writes_summary(tmp_path):
    env = dict(os.environ, EPOCHS="2", OUT_DIR=str(tmp_path), SEED="42")
    subprocess.run([sys.executable, str(ROOT / "01_k8s_job" / "train.py")], env=env, check=True, timeout=300)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (tmp_path / "model.pt").exists()
    assert summary["epochs"] == 2
    assert summary["val_acc"] > 0.8   # far above chance (0.33) for 3 classes
