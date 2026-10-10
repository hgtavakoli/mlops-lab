"""Model release gate: compare a candidate model's metrics with the baseline, class by class.

Used by CI: the pipeline fails (exit code 1) if the candidate model lost too much accuracy,
overall or on any single class. Run it yourself from the repo root:

    python -m lablib.gate --config gate/release.json
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateResult:
    passed: bool
    overall_drop: float
    failures: list = field(default_factory=list)  # (class name, baseline AP, candidate AP, drop)


def check(baseline: dict, candidate: dict, max_overall_drop: float, max_class_drop: float) -> GateResult:
    """baseline / candidate: {"mAP50_95": float, "per_class": {class_name: AP}}.

    A class missing from the candidate counts as AP 0: the model stopped detecting it.
    """
    overall_drop = baseline["mAP50_95"] - candidate["mAP50_95"]
    failures = []
    for name, base_ap in sorted(baseline["per_class"].items()):
        cand_ap = candidate["per_class"].get(name, 0.0)
        drop = base_ap - cand_ap
        if drop > max_class_drop:
            failures.append((name, base_ap, cand_ap, drop))
    failures.sort(key=lambda f: f[3], reverse=True)
    passed = overall_drop <= max_overall_drop and not failures
    return GateResult(passed=passed, overall_drop=overall_drop, failures=failures)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="gate/release.json")
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    results = json.loads(Path(cfg["results_file"]).read_text())
    base = results[cfg["baseline"]]
    cand = results[cfg["candidate"]]
    res = check(base, cand, cfg["max_overall_drop"], cfg["max_class_drop"])

    print(f"Release gate: candidate '{cfg['candidate']}' vs baseline '{cfg['baseline']}'")
    print(f"  overall mAP50-95: {base['mAP50_95']:.3f} -> {cand['mAP50_95']:.3f} "
          f"(drop {res.overall_drop:+.3f}, allowed {cfg['max_overall_drop']})")
    print(f"  latency p50: {base['p50_ms']:.1f} ms -> {cand['p50_ms']:.1f} ms (reported, not gated)")
    if res.failures:
        print(f"  classes that dropped more than {cfg['max_class_drop']}:")
        for name, b, c, d in res.failures:
            print(f"    {name:<16} {b:.3f} -> {c:.3f}  (-{d:.3f})")
    print("PASS" if res.passed else "FAIL")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
