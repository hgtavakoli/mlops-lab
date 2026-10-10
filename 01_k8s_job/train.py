"""Step 1: a tiny PyTorch training that runs as a Kubernetes Job.

It trains a small classifier on synthetic data, so it needs no download.
Settings come from environment variables, the way a Job passes config.
"""
import json
import os
import time

import torch
from torch import nn

EPOCHS = int(os.environ.get("EPOCHS", "5"))
LR = float(os.environ.get("LR", "0.01"))
SEED = int(os.environ.get("SEED", "42"))
OUT_DIR = os.environ.get("OUT_DIR", "/output")
torch.manual_seed(SEED)

# Synthetic data: 2,000 points, 20 features, 3 classes.
n, d, k = 2000, 20, 3
true_w = torch.randn(d, k)
x = torch.randn(n, d)
y = (x @ true_w).argmax(dim=1)
x_train, y_train = x[:1600], y[:1600]
x_val, y_val = x[1600:], y[1600:]

model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, k))
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print(f"[job] start: epochs={EPOCHS} lr={LR} seed={SEED} torch={torch.__version__}", flush=True)
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    for i in range(0, len(x_train), 64):
        xb, yb = x_train[i:i + 64], y_train[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
    print(f"[job] epoch {epoch}/{EPOCHS} loss={loss.item():.4f} val_acc={acc:.3f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
summary = {"epochs": EPOCHS, "lr": LR, "seed": SEED, "val_acc": round(acc, 4), "seconds": round(time.time() - t0, 1)}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f)
print(f"[job] done: {json.dumps(summary)}", flush=True)
