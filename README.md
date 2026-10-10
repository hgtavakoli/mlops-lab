# MLOps lab for the MicroVision interview

Three small exercises that cover the gaps in the posting: a Kubernetes Job, MLflow tracking with a model registry, and INT8 quantization with a per-class check. Total time is about 2–3 hours on a Windows laptop, CPU only.

Steps 2 and 3 were tested end to end before you got them (results in `03_onnx_int8/reference_results_cloud.txt`). Step 1 couldn't be tested in advance, so do it first: that's where setup problems show up.

After each step there are **check questions**. Answer them in the chat in your own words and I'll correct you. That's the interview practice.

---

## 0. Setup (20 min)

1. Start **Docker Desktop** and wait until it says "Engine running".
2. In **PowerShell**, install kind and kubectl:

   ```powershell
   winget install Kubernetes.kind
   winget install Kubernetes.kubectl
   ```

   Close and reopen PowerShell, then check: `kind version` and `kubectl version --client`.
3. Python environment (Python 3.10–3.12), from this `mlops_lab` folder:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

   If PowerShell refuses to run `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

---

## 1. Training as a Kubernetes Job (45 min)

Files: `01_k8s_job/train.py` (tiny PyTorch training, synthetic data), `Dockerfile`, `job.yaml`.

```powershell
kind create cluster --name lab                    # a one-node Kubernetes cluster inside Docker
kubectl get nodes                                 # wait for STATUS Ready

docker build -t lab-train:0.1 01_k8s_job          # build the training image (~1 GB, first time is slow)
kind load docker-image lab-train:0.1 --name lab   # copy the image into the cluster; there is no registry

kubectl apply -f 01_k8s_job/job.yaml              # submit the Job
kubectl get jobs,pods                             # watch it go Pending -> Running -> Completed
kubectl logs job/train-v1 -f                      # stream the training log
kubectl describe job train-v1                     # events, retries, timing
```

**Break it on purpose (this is where you learn):**

- **Change a hyperparameter.** Set `EPOCHS` to `"20"` in `job.yaml` and run `kubectl apply` again. It fails, because a Job's pod template can't be changed. Run `kubectl delete job train-v1`, then apply again. Note that real pipelines give every run a new Job name.
- **Run out of memory.** Set `limits: memory: 50Mi` and re-run. Then `kubectl get pods` and `kubectl describe pod <pod-name>`. Look for `OOMKilled` and count the retries (`backoffLimit: 2`).
- **Pending forever.** Set `requests: cpu: "64"`. The pod stays `Pending`, and `describe` says why ("Insufficient cpu"). This is how GPU jobs wait when every GPU is busy.

Clean up when done: `kind delete cluster --name lab`.

**Check questions**

1. Why is training a **Job** and not a **Deployment**? Why `restartPolicy: Never`?
2. What is the difference between `requests` and `limits`? What happens at each one?
3. Why did we need `kind load docker-image` and `imagePullPolicy: IfNotPresent`? What replaces this on a real cluster?
4. How would you ask for a GPU, and what has to be installed on the cluster for that line to work?
5. The model was written to `emptyDir`. What happens to it when the pod is deleted, and where should it go instead?

---

## 2. MLflow tracking and the model registry (45 min)

```powershell
cd 02_mlflow
python train_yolo_mlflow.py --epochs 2                         # ~2-5 min on CPU; downloads YOLO11n weights and coco128 (~7 MB)
mlflow ui --backend-store-uri sqlite:///mlflow.db              # open http://127.0.0.1:5000
```

In the UI, find:

- the **run**: parameters, tags (versions of torch and ultralytics), metrics, and per-class `AP_*`
- the **artifact** `weights/best.pt`
- **Models → yolo-lab**: version 1 with the alias `champion`

Then make a second model and promote it the way a team would:

```powershell
python train_yolo_mlflow.py --epochs 3 --alias challenger
```

Compare the two runs side by side in the UI (tick both, then **Compare**). If the challenger is better, move `champion` to it in the UI (model version page → aliases). Step 3 always uses whatever `champion` points to.

**Check questions**

1. What did we log so that someone else could reproduce this run? What is still missing?
2. What's the difference between a run, an artifact, a registered model, a version and an alias?
3. Why do downstream steps load "champion" instead of a file path?
4. The tracking store here is a SQLite file. What would a team use instead?

---

## 3. ONNX, INT8, and checking what you lost (30 min)

```powershell
cd ..\03_onnx_int8
python quantize_and_compare.py
```

The script:

1. loads the `champion` model from the registry
2. exports it to ONNX in FP32
3. quantizes it to INT8 twice: once fully, and once with the detection head kept in FP32
4. measures p50 and p99 latency at batch size 1
5. validates every model and prints the classes that lost the most mAP
6. logs the comparison back to MLflow

Compare your numbers with `reference_results_cloud.txt`, which is the same script on a cloud CPU. They will differ, and the difference is part of the lesson.

**Check questions**

1. Why do you need **calibration images** for static INT8 quantization? What happens if they don't look like real data?
2. Fully quantized, whole classes dropped to 0 mAP, and keeping the last block in FP32 fixed most of it. Why? (Hint: look at what the head's output tensor contains.)
3. Overall mAP fell only about 0.01 with the head in FP32. Was that safe to ship? Look at the per-class list.
4. Was INT8 faster on your machine? Why might it be slower on a laptop CPU but much faster with TensorRT on an NVIDIA chip?
5. How would this script become a **release gate** in CI?

---

## 4. Five lines (10 min)

Fill in `04_what_surprised_me.md`. That's your answer to "What have you done about the gaps?"

Be honest about scale: "I ran a hands-on lab: a training Job on a local kind cluster, MLflow tracking with a registry and aliases, and INT8 quantization with a per-class check." Don't call it production experience.

---

## 5. CI/CD (2–3 hours)

See `05_ci_cd/README.md`: put the lab on GitHub, and add a pipeline (lint, tests, a per-class model release gate, then build and push the Docker image) with exercises that make it fail on purpose.
