# What surprised me (five lines, my own words)

1. Kubernetes:
2. MLflow / registry:
3. INT8 accuracy:
4. INT8 speed:
5. What I'd do next at MicroVision:

---

For comparison, the same lab on a cloud CPU found:

- Fully quantized YOLO11n fell from 0.385 to 0.149 mAP50-95, and whole classes such as tv, laptop and sandwich went to 0.
- Keeping the detection head in FP32 brought it back to 0.374. Overall that looks fine, but single classes still lost up to 0.18 (sandwich 0.895 → 0.712). An overall-mAP gate would have let that through.
- INT8 was **slower** than FP32 on that CPU (p50 about 29–34 ms vs 13 ms), while the file shrank from 10.6 to 3.2–4.4 MB. Whether INT8 is faster depends on the hardware and runtime (TensorRT on NVIDIA, TI's own toolchain on TI), so it has to be measured on the target.
- coco128 has few examples per class, so per-class numbers are noisy. A real gate needs enough examples per slice.
