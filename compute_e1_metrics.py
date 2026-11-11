"""
compute_e1_metrics.py — Experiment 1 metrics against a manually compiled reference set.

Compare pipeline verdicts (data/e1_results/*_audit.json) with the manual
reference annotations (data/ground_truth_e1.json) and calculate:

  - exact three-class agreement (YES / PARTIAL / NOT_STATED);
  - ordinal agreement within one step (NOT_STATED=0 < PARTIAL=1 < YES=2);
  - criterion-level precision, recall and F1, macro-averaged across 24 criteria
    (binary formulation: positive = YES|PARTIAL; negative = NOT_STATED);
  - GDPR-category precision, recall and F1, micro-averaged within each category;
  - false-positive and false-negative counts; and
  - the complete report in data/e1_metrics.json.

The seguridad_social policy is excluded because its URL redirected to a home
portal and the captured text contained no assessable information. NOT_APPLICABLE
cases in the reference set are excluded from all calculations.

Usage:
  python compute_e1_metrics.py
  python compute_e1_metrics.py --results data/e1_results --gt data/ground_truth_e1.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# The repository historically stored results and reference annotations under
# data1/. New pipeline runs write them under data/. Prefer data/ when available.
_DATA = Path("data") if Path("data/ground_truth_e1.json").exists() else Path("data1")

RESULTS_DIR = _DATA / "e1_results"
GT_PATH     = _DATA / "ground_truth_e1.json"
METRICS_OUT = _DATA / "e1_metrics.json"

EXCLUDED_POLICIES = {"seguridad_social"}   # URL not suitable for assessment.

SCALE = {"NOT_STATED": 0, "PARTIAL": 1, "YES": 2}   # ordinal scale

# The 24 criteria in the localised checklist files, in their prescribed order.
ALL_CRITERIA = [f"C{n:02d}" for n in range(1, 25)]

CRIT_NAMES = {
    "C01": "Controller identity",
    "C02": "Controller contact details",
    "C03": "Data protection officer",
    "C04": "Purposes of processing",
    "C05": "Lawful basis of processing",
    "C06": "Legitimate-interest justification",
    "C07": "Categories of personal data",
    "C08": "Source of personal data",
    "C09": "Recipients and third parties",
    "C10": "International transfers",
    "C11": "Retention periods",
    "C12": "Right of access",
    "C13": "Right to rectification",
    "C14": "Right to erasure",
    "C15": "Right to data portability",
    "C16": "Right to object",
    "C17": "Withdrawal of consent",
    "C18": "Right to lodge a complaint",
    "C19": "Automated decision-making and profiling",
    "C20": "Security measures",
    "C21": "Clear and accessible language",
    "C22": "Notification of policy changes",
    "C23": "Special categories of personal data",
    "C24": "Processing of children's data",
}

# GDPR category mapping (eight thematic groups) used in the category-level table.
CRIT_TO_CAT = {
    "C01": "Identity and contact details", "C02": "Identity and contact details", "C03": "Identity and contact details",
    "C04": "Lawful bases (Article 6)", "C05": "Lawful bases (Article 6)", "C06": "Lawful bases (Article 6)",
    "C07": "Personal data processed", "C08": "Personal data processed",
    "C09": "Recipients and transfers", "C10": "Recipients and transfers",
    "C12": "Data subject rights", "C13": "Data subject rights", "C14": "Data subject rights",
    "C15": "Data subject rights", "C16": "Data subject rights", "C17": "Data subject rights",
    "C11": "Additional information", "C18": "Additional information", "C19": "Additional information",
    "C20": "Additional information", "C21": "Additional information", "C22": "Additional information",
    "C23": "Special-category data (Article 9)",
    "C24": "Children (Article 8)",
}
CAT_ORDER = [
    "Identity and contact details", "Lawful bases (Article 6)", "Personal data processed",
    "Recipients and transfers", "Data subject rights",
    "Additional information", "Special-category data (Article 9)", "Children (Article 8)",
]


def prf(tp: int, fp: int, fn: int):
    """Return precision, recall and F1; assign 1.0 when no positives or errors exist."""
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def load_gt(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["policy"]: p["ground_truth"] for p in data["policies"]}


def load_pipeline(results_dir: Path):
    results = {}
    for p in sorted(results_dir.glob("*_audit.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        name = p.stem.replace("_audit", "")
        results[name] = {r["criterion_id"]: r["verdict"] for r in data.get("results", [])}
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 metrics")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--gt",      type=Path, default=GT_PATH)
    parser.add_argument("--out",     type=Path, default=METRICS_OUT)
    args = parser.parse_args()

    assert args.results.exists(), f"Results directory not found: {args.results}"
    assert args.gt.exists(),      f"Reference annotation file not found: {args.gt}"

    gt_all   = load_gt(args.gt)
    pipe_all = load_pipeline(args.results)

    counts   = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0} for c in ALL_CRITERIA}
    per_policy = []
    skipped  = []

    tot_exact = tot_within1 = tot_comps = 0
    tot_fp = tot_fn = 0
    n_yes_partial_swap = 0        # confusiones YES <-> PARTIAL (dist. ordinal = 1, ambos positivos)
    n_partial_not_stated = 0       # GT PARTIAL, pipeline NOT_STATED (FN de distancia 1)

    for policy in sorted(gt_all):
        if policy in EXCLUDED_POLICIES:
            skipped.append(f"{policy} — excluded (URL not suitable for assessment)")
            continue
        if policy not in pipe_all:
            skipped.append(f"{policy} — no pipeline results in {args.results}")
            continue

        gt_v   = gt_all[policy]
        pipe_v = pipe_all[policy]
        row = {"policy": policy, "exact": 0, "within1": 0, "n": 0, "discrepancies": []}

        for crit in ALL_CRITERIA:
            g = gt_v.get(crit)
            if g is None or g == "NOT_APPLICABLE":
                continue
            p = pipe_v.get(crit, "NOT_STATED")
            if p == "NOT_APPLICABLE":
                p = "NOT_STATED"

            gs, ps = SCALE[g], SCALE[p]
            diff = ps - gs
            exact   = (diff == 0)
            within1 = (abs(diff) <= 1)

            row["n"] += 1
            tot_comps += 1
            if exact:
                row["exact"] += 1
                tot_exact += 1
            if within1:
                row["within1"] += 1
                tot_within1 += 1
            if not exact:
                row["discrepancies"].append({"criterion": crit, "gt": g, "pipeline": p, "diff": diff})
                if {g, p} == {"YES", "PARTIAL"}:
                    n_yes_partial_swap += 1
                elif g == "PARTIAL" and p == "NOT_STATED":
                    n_partial_not_stated += 1

            gt_pos   = 1 if g in ("YES", "PARTIAL") else 0
            pipe_pos = 1 if p in ("YES", "PARTIAL") else 0
            tp = int(pipe_pos == 1 and gt_pos == 1)
            fp = int(pipe_pos == 1 and gt_pos == 0)
            fn = int(pipe_pos == 0 and gt_pos == 1)
            tn = int(pipe_pos == 0 and gt_pos == 0)

            c = counts[crit]
            c["tp"] += tp; c["fp"] += fp; c["fn"] += fn; c["tn"] += tn; c["n"] += 1
            tot_fp += fp; tot_fn += fn

        per_policy.append(row)

    # ── Criterion-level metrics ────────────────────────────────────────────────
    crit_metrics = {}
    for crit in ALL_CRITERIA:
        c = counts[crit]
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        crit_metrics[crit] = {"name": CRIT_NAMES[crit], "n": c["n"],
                              "precision": p, "recall": r, "f1": f1, **c}

    n_crit   = len(crit_metrics)
    macro_p  = sum(m["precision"] for m in crit_metrics.values()) / n_crit
    macro_r  = sum(m["recall"]    for m in crit_metrics.values()) / n_crit
    macro_f1 = sum(m["f1"]        for m in crit_metrics.values()) / n_crit

    # ── Category-level metrics (micro-average) ───────────────────────────────
    cat_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for crit, c in counts.items():
        cat = CRIT_TO_CAT[crit]
        for k in ("tp", "fp", "fn", "tn"):
            cat_counts[cat][k] += c[k]
    cat_metrics = {}
    for cat in CAT_ORDER:
        c = cat_counts[cat]
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        cat_metrics[cat] = {"precision": p, "recall": r, "f1": f1, **c}

    # ── Console output ──────────────────────────────────────────────────
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    print("=" * 78)
    print("EXPERIMENT 1 — Evaluation against the manual reference set")
    print("=" * 78)
    print(f"Policies evaluated : {len(per_policy)}")
    print(f"Comparisons        : {tot_comps}  (NOT_APPLICABLE reference cases excluded)")
    for s in skipped:
        print(f"  [SKIP] {s}")
    print()
    print(f"Exact agreement (three classes): {tot_exact}/{tot_comps} ({pct(tot_exact, tot_comps)})")
    print(f"Agreement within one ordinal step: {tot_within1}/{tot_comps} ({pct(tot_within1, tot_comps)})")
    print(f"False positives (FP)            : {tot_fp}")
    print(f"False negatives (FN)            : {tot_fn}")
    print(f"  YES<->PARTIAL disagreements        : {n_yes_partial_swap}")
    print(f"  Reference=PARTIAL, pipeline=NOT_STATED: {n_partial_not_stated}")
    print()
    print(f"Macro-averaged precision: {macro_p:.3f}")
    print(f"Macro-averaged recall   : {macro_r:.3f}")
    print(f"Macro-averaged F1       : {macro_f1:.3f}")
    print()

    header = f"{'ID':<5}{'Criterion':<38}{'n':>4}{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>4}{'FN':>4}"
    print(header)
    print("-" * len(header))
    for crit in ALL_CRITERIA:
        m = crit_metrics[crit]
        print(f"{crit:<5}{m['name'][:37]:<38}{m['n']:>4}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['tp']:>5}{m['fp']:>4}{m['fn']:>4}")
    print("-" * len(header))
    print(f"{'':5}{'Macro-average (24 criteria)':<38}{'':>4}{macro_p:>8.3f}{macro_r:>8.3f}{macro_f1:>8.3f}")
    print()

    print(f"{'GDPR category':<34}{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 58)
    for cat in CAT_ORDER:
        m = cat_metrics[cat]
        print(f"{cat:<34}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")

    # ── Serialise JSON ────────────────────────────────────────────────────────
    output = {
        "n_policies": len(per_policy),
        "n_comparisons": tot_comps,
        "excluded_policies": sorted(EXCLUDED_POLICIES),
        "agreement": {
            "exact": tot_exact,
            "exact_pct": round(100 * tot_exact / tot_comps, 1) if tot_comps else None,
            "within1": tot_within1,
            "within1_pct": round(100 * tot_within1 / tot_comps, 1) if tot_comps else None,
        },
        "false_positives": tot_fp,
        "false_negatives": tot_fn,
        "yes_partial_swaps": n_yes_partial_swap,
        "partial_to_not_stated": n_partial_not_stated,
        "overall": {
            "precision_macro": round(macro_p, 4),
            "recall_macro": round(macro_r, 4),
            "f1_macro": round(macro_f1, 4),
        },
        "criteria_metrics": crit_metrics,
        "category_metrics": cat_metrics,
        "per_policy": per_policy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to -> {args.out}")


if __name__ == "__main__":
    main()
