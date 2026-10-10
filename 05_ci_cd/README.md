# Step 5: CI/CD for the lab (about 2–3 hours)

You'll put the lab on GitHub and add a pipeline that runs on every change:

```
push / pull request ──► lint ──► tests ──► release gate ──► (main only) build image ──► push to registry
                        └──────────── CI: is this change safe? ───────────┘   └──── CD: ship it ────┘
```

Everything here is already written and tested: 8 tests pass and the gate works on your step-3 results. Your job is to run it, break it on purpose, and be able to explain it.

---

## 0. The ideas in 2 minutes

| Word | Meaning |
|---|---|
| **CI** (continuous integration) | Every change is checked automatically (lint, tests) before it joins the main code |
| **CD** (continuous delivery) | Every change that passes is packaged and published automatically (here: a Docker image) |
| **Workflow / pipeline** | The file `.github/workflows/ci.yml`: what to run, and when |
| **Trigger** | What starts it: a push to `main`, a pull request, or a button (`workflow_dispatch`) |
| **Job** | One block of work on a fresh machine (`lint`, `test`, `release-gate`, `build-and-push-image`) |
| **Step** | One command inside a job |
| **Runner** | The machine that runs a job (GitHub gives free Linux VMs) |
| **`needs:`** | Order between jobs: tests only run if lint passed, and so on |
| **Registry** | Where images are stored: GitHub Container Registry (`ghcr.io`) |
| **Branch protection** | A rule that `main` only accepts changes through a pull request whose checks are green |

The same ideas in **GitLab CI** (MicroVision's tool):

| GitHub Actions | GitLab CI |
|---|---|
| `.github/workflows/ci.yml` | `.gitlab-ci.yml` |
| job + `needs:` | job + `stage:` (and `needs:`) |
| runner (`runs-on:`) | runner (`tags:`) |
| pull request | merge request |
| `ghcr.io` | GitLab Container Registry |
| `secrets.GITHUB_TOKEN` | `CI_JOB_TOKEN` |

**What's new in the lab folder:**

| File | What it is |
|---|---|
| `lablib/gate.py` | Release gate: fails if the candidate model loses > 0.02 overall mAP or > 0.10 on **any one class** |
| `lablib/preprocess.py` | The image preprocessing from step 3, as a testable function |
| `tests/` | 5 gate tests, 2 preprocessing tests, 1 smoke test that runs the step-1 training |
| `gate/release.json` | Which model is the release candidate, and the thresholds |
| `gate/results_reference.json` | The step-3 results the gate reads |
| `.github/workflows/ci.yml` | The pipeline |
| `pyproject.toml`, `requirements-dev.txt`, `.gitignore` | Lint and test settings, test dependencies, files Git must ignore |

---

## 1. Run the checks on your laptop first (15 min)

In the `(mlops-lab)` window, from the `mlops_lab` folder:

```powershell
pip install -r requirements-dev.txt
ruff check lablib tests 01_k8s_job
pytest -v
python -m lablib.gate --config gate/release.json
```

Expected: `All checks passed!`, `8 passed` (the smoke test takes about a minute), then `PASS`.

Optional: to use **your own** step-3 numbers, run `copy 03_onnx_int8\work\results.json gate\results_reference.json`.

---

## 2. Put the lab on GitHub (20 min)

1. Install Git if needed: `winget install Git.Git`, then reopen PowerShell. Set your name once:

   ```powershell
   git config --global user.name "Hooman Tavakoli Ghinani"
   git config --global user.email "hgtavakoli@gmail.com"
   ```

2. On github.com, create a new **public** repository named `mlops-lab`. Leave "Add a README" **unticked**.
3. In the `mlops_lab` folder:

   ```powershell
   git init -b main
   git add .
   git status
   ```

   **Check the list before committing.** There must be no `.pt`, `.onnx`, `mlflow.db`, `datasets` or `runs`; `.gitignore` excludes them. Big model files don't belong in Git.

   ```powershell
   git commit -m "MLOps lab: K8s Job, MLflow, INT8 quantization, CI pipeline"
   git remote add origin https://github.com/<your-user>/mlops-lab.git
   git push -u origin main
   ```

   A browser window asks you to log in to GitHub the first time.

4. On GitHub, open the **Actions** tab. The `ci` workflow starts by itself. Click it and watch the four jobs go green: lint → test → release-gate → build-and-push-image.
5. On your GitHub profile, open **Packages**. You'll see `lab-train`: the Docker image that CD built and pushed, tagged with the commit SHA and `latest`.

---

## 3. Protect `main` (5 min)

Repository → **Settings → Branches → Add branch protection rule** (or **Rules → Rulesets**):

- Branch name pattern: `main`
- Tick **Require a pull request before merging**. Set required approvals to **0**, because you can't approve your own pull request.
- Tick **Require status checks to pass**, and add `lint`, `test` and `release-gate`.

From now on, every change goes through a pull request that must be green.

---

## 4. Exercise A: fix a real bug through a pull request (20 min)

Step 1 printed `Failed to initialize NumPy` because the image didn't install NumPy. Fix it the way a team would:

```powershell
git checkout -b fix/numpy-in-image
```

In `01_k8s_job/Dockerfile`, change the install line to

```dockerfile
RUN pip install --no-cache-dir torch numpy --index-url https://download.pytorch.org/whl/cpu
```

```powershell
git commit -am "Install numpy in the training image"
git push -u origin fix/numpy-in-image
```

On GitHub, click **Compare & pull request**. Write two lines on what changed and why. Watch the checks run on the pull request, then **Merge**. After the merge, `main` runs again, including **build-and-push-image**: a new image is published with the fix.

---

## 5. Exercise B: a test catches a bug (15 min)

```powershell
git checkout main
git pull
git checkout -b break/padding
```

In `lablib/preprocess.py`, change the padding value `114` to `0`. Commit, push and open a pull request. The **test** job turns red, `test_letterbox_keeps_aspect_and_pads_grey` fails. If you set up branch protection in section 3, the **Merge** button is blocked; without it, GitHub only shows a warning and still lets you merge. Read the error in the job log. Then close the pull request without merging.

This is what CI is for: wrong preprocessing silently changes a model's inputs, and the test catches it before it reaches `main`.

---

## 6. Exercise C: the release gate stops a bad model (15 min)

```powershell
git checkout main
git checkout -b release/int8-head-fp32
```

In `gate/release.json`, change `"candidate": "onnx_fp32"` to `"candidate": "int8_head_fp32"`. Commit, push and open a pull request. **lint** and **test** are green, but **release-gate** fails:

```
overall mAP50-95: 0.385 -> 0.374 (drop +0.012, allowed 0.02)   <- the average looks fine
classes that dropped more than 0.1:
  sandwich  0.895 -> 0.712  (-0.183)
  couch     0.370 -> 0.222  (-0.148)
  ...
FAIL
```

That's your step-3 finding, now enforced automatically. Close the pull request. **Don't** "fix" it by raising `max_class_drop`: weakening a gate to get a model through is exactly what a gate exists to prevent.

---

## 7. Optional: deploy the CI-built image to kind (20 min)

1. Make the package public: GitHub → Packages → `lab-train` → Package settings → Change visibility → Public.
2. Point the Job at the registry instead of a local image. In `01_k8s_job/job.yaml`, set `image: ghcr.io/<your-user>/mlops-lab/lab-train:latest` (all lower case) and `imagePullPolicy: Always`.
3. Run `kind create cluster --name lab`, `kubectl apply -f 01_k8s_job/job.yaml` and `kubectl logs job/train-v1 -f`.

The cluster now pulls the image that CI built and tested. There's no `kind load` and no manual build: that's the full CI/CD loop.

---

## Check questions (answer them in the chat)

1. What's the difference between CI and CD in this pipeline? Which job is CD?
2. Why does `build-and-push-image` run only on `main` and not on pull requests?
3. Why do tests run on a fresh machine instead of trusting that they pass on your laptop?
4. In exercise C, lint and tests were green but the pull request was blocked. What did the gate know that the tests didn't?
5. Why are `.pt`, `.onnx` and `mlflow.db` in `.gitignore`? Where do models live instead?
6. How would you translate `ci.yml` into a `.gitlab-ci.yml`? Name the stages.

---

## After you've done it: CV and interview

Add only what you actually ran. If you completed steps 2–6, this bullet goes under the MLOps project:

> Added CI/CD with GitHub Actions: linting (ruff), unit and smoke tests (pytest), an automated per-class model release gate, and Docker images built and pushed to GitHub Container Registry on every merge to a protected main branch.

Skills row: add `GitHub Actions (CI/CD), pytest` to the MLOps tooling row, and the repo link to the header or the project title.

In an interview:

> "In my lab, every pull request runs lint, unit tests, a smoke training run and a per-class release gate; merges to main build and publish the training image. The gate blocked an INT8 model whose overall mAP fell only 0.01 because one class fell 0.18."

Be honest about the scale: it's a personal project with GitHub Actions, not a team pipeline. Code review here means a pull-request workflow with required checks, not reviews by other people.
