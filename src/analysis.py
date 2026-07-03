"""Answers PharmaCorp's four questions over all 50 extracted patients, with churn
handled as a first-class caveat rather than swept under the rug, and writes the
supporting charts.

Population rules (why each question uses a different denominator):
  - Q1 uses ALL 50 records — the % on a biologic, with 'unknown' broken out rather
    than folded into 'off'. `derive.q1_range` gives the stated/inferred band.
  - Q2 (barriers) is only defined for patients NOT confirmed on a biologic, i.e.
    `on_biologic in (False, None)` — includes the genuinely-unknown/churned-mid-
    decision cases (a barrier can be known even when the final outcome isn't, e.g.
    P049), excludes anyone confirmed on one (barriers wouldn't apply to them).
  - Q3 (treatments before a biologic) uses ALL 50 — `treatment_sequence` already
    returns the empty list for anyone who never tried anything first, which IS an
    answer (some patients go straight to a biologic).
  - Q4 (referral steps) only counts patients with a STATED step count — i.e. the
    referral chain actually resolved (a biologic was prescribed or discussed). Churned
    patients where that point was never reached are excluded from the average and
    reported separately as "not assessable", not silently coded as 0 or skipped.

Run:  python src/analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from derive import q1_range, referral_steps, treatment_sequence  # noqa: E402
from schema import PatientRecord, Status  # noqa: E402

CACHE_DIR = Path("data/cache")
OUT_DIR = Path("data/outputs")
CHART_DIR = OUT_DIR / "charts"
RECORDS_JSON = OUT_DIR / "records.json"  # committed evidence — see load_records()


def load_records(cache_dir: Path = CACHE_DIR) -> list[PatientRecord]:
    """data/cache/ (raw extraction cache) is gitignored and regenerable — it won't
    exist in a fresh checkout without running run_extraction.py first, which needs
    an API key. Fall back to data/outputs/records.json (the committed evidence
    from the last real run) so `python src/analysis.py` reproduces the same
    numbers/charts out of the box, with no key required. PatientRecord.model_validate
    ignores the extra 'derived' key records.json carries alongside the raw fields."""
    records = [PatientRecord.model_validate_json(f.read_text()) for f in sorted(cache_dir.glob("P*.json"))]
    if records:
        return records
    if RECORDS_JSON.exists():
        print(f"(no data/cache/ found — reproducing from committed {RECORDS_JSON})")
        return [PatientRecord.model_validate(r) for r in json.loads(RECORDS_JSON.read_text())]
    return records


def q1_on_biologic(records: list[PatientRecord]) -> dict:
    on = sum(1 for r in records if r.on_biologic is True)
    off = sum(1 for r in records if r.on_biologic is False)
    unknown = sum(1 for r in records if r.on_biologic is None)
    lo, hi, band = q1_range(records)
    return {"n": len(records), "on": on, "off": off, "unknown": unknown,
            "pct_on_point_estimate": on / len(records) if records else 0,
            "band_low": lo, "band_high": hi, "band_detail": band}


def q2_barriers(records: list[PatientRecord]) -> dict:
    not_confirmed_on = [r for r in records if r.on_biologic is not True]
    counts = Counter(b.type.value for r in not_confirmed_on for b in r.biologic.barriers)
    no_barrier_identified = sum(1 for r in not_confirmed_on if not r.biologic.barriers)
    churned_unknown = sum(
        1 for r in not_confirmed_on
        if not r.biologic.barriers and r.biologic.barriers_status is Status.UNKNOWN_CHURNED
    )
    return {
        "denominator_n": len(not_confirmed_on),
        "barrier_counts": dict(counts),
        "no_barrier_identified": no_barrier_identified,
        "of_which_churned_before_known": churned_unknown,
    }


def q3_treatments_before_biologic(records: list[PatientRecord]) -> dict:
    per_patient_classes = [set(treatment_sequence(r)) for r in records]
    counts = Counter(c for classes in per_patient_classes for c in classes)
    none_tried_first = sum(1 for classes in per_patient_classes if not classes)
    return {"n": len(records), "class_counts": dict(counts), "none_tried_first": none_tried_first}


def q4_referral_steps(records: list[PatientRecord]) -> dict:
    resolved = []
    for r in records:
        sig = referral_steps(r)
        if sig.status is Status.STATED and sig.value is not None:
            resolved.append((r, sig.value))
    steps = [v for _, v in resolved]
    entry_types = Counter()
    for r, _ in resolved:
        on_path = sorted((e for e in r.encounters if e.on_referral_path),
                          key=lambda e: e.sequence_index or 0)
        if on_path:
            entry_types[on_path[0].encounter_type.value] += 1
    not_assessable = len(records) - len(resolved)
    return {
        "n_resolved": len(resolved),
        "not_assessable_n": not_assessable,
        "step_counts": dict(Counter(steps)),
        "mean_steps": sum(steps) / len(steps) if steps else None,
        "entry_point_mix": dict(entry_types),
    }


def churn_summary(records: list[PatientRecord]) -> dict:
    churned = sum(1 for r in records if r.churn)
    return {"n": len(records), "churned": churned, "churn_rate": churned / len(records) if records else 0}


def _bar_chart(labels, values, title, ylabel, path, color="#4C72B0"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_charts(q1, q2, q3, q4, eval_result, out_dir: Path = CHART_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _bar_chart(["On", "Off", "Unknown"], [q1["on"], q1["off"], q1["unknown"]],
               "Q1 — biologic status across 50 patients", "patients",
               out_dir / "q1_biologic_status.png")

    if q2["barrier_counts"]:
        items = sorted(q2["barrier_counts"].items(), key=lambda x: -x[1])
        _bar_chart([k for k, _ in items], [v for _, v in items],
                   "Q2 — reasons not (yet) on a biologic", "patients (of not-confirmed-on)",
                   out_dir / "q2_barrier_frequency.png", color="#DD8452")

    if q3["class_counts"]:
        items = sorted(q3["class_counts"].items(), key=lambda x: -x[1])
        _bar_chart([k for k, _ in items], [v for _, v in items],
                   "Q3 — treatment classes tried/discussed before a biologic", "patients",
                   out_dir / "q3_treatment_class_frequency.png", color="#55A868")

    if q4["step_counts"]:
        items = sorted(q4["step_counts"].items())
        _bar_chart([str(k) for k, _ in items], [v for _, v in items],
                   "Q4 — referral steps to the biologic decision point", "patients",
                   out_dir / "q4_step_histogram.png", color="#C44E52")
    if q4["entry_point_mix"]:
        items = sorted(q4["entry_point_mix"].items(), key=lambda x: -x[1])
        _bar_chart([k for k, _ in items], [v for _, v in items],
                   "Q4 — entry point into the referral chain", "patients",
                   out_dir / "q4_entry_point_mix.png", color="#8172B2")

    if eval_result:
        scores = eval_result.get("per_field_scores", {})
        labels, vals = [], []
        for field, s in sorted(scores.items()):
            if not s:
                continue
            labels.append(field)
            vals.append(sum(s) / len(s) * 100)
        if labels:
            _bar_chart(labels, vals, "Eval — golden-set agreement by field", "% agreement",
                       out_dir / "eval_agreement.png", color="#64B5CD")


def main() -> None:
    records = load_records()
    if not records:
        print("No cached records found in data/cache — run extraction first.")
        return

    q1, q2, q3, q4, churn = (
        q1_on_biologic(records), q2_barriers(records),
        q3_treatments_before_biologic(records), q4_referral_steps(records),
        churn_summary(records),
    )

    print(f"n = {len(records)} patients  ·  churned/truncated = {churn['churned']} "
          f"({churn['churn_rate']:.0%})\n")

    print("Q1 — % on a biologic")
    print(f"  point estimate: {q1['pct_on_point_estimate']:.0%}  "
          f"(on={q1['on']}, off={q1['off']}, unknown={q1['unknown']})")
    print(f"  band (stated-only .. everything that could still be on): "
          f"{q1['band_low']:.0%}-{q1['band_high']:.0%}")
    cwe = q1["band_detail"].get("confidence_weighted_estimate")
    if cwe is not None:
        print(f"  confidence-weighted estimate (secondary — trusts the model's own "
              f"self-reported confidence on inferred cases): {cwe:.0%}\n")
    else:
        print()

    print(f"Q2 — reasons not on a biologic (of {q2['denominator_n']} not confirmed on one)")
    for k, v in sorted(q2["barrier_counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:20} {v}")
    print(f"  no reason identified: {q2['no_barrier_identified']} "
          f"(of which churned before the reason surfaced: {q2['of_which_churned_before_known']})\n")

    print(f"Q3 — treatment classes tried/discussed before a biologic (of {q3['n']})")
    for k, v in sorted(q3["class_counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:20} {v}")
    print(f"  none tried first (straight to biologic or never got there): {q3['none_tried_first']}\n")

    print(f"Q4 — referral steps (resolved for {q4['n_resolved']}/{len(records)}; "
          f"{q4['not_assessable_n']} not assessable — biologic never reached before churn/end)")
    for k, v in sorted(q4["step_counts"].items()):
        print(f"  {k} steps: {v} patients")
    if q4["mean_steps"] is not None:
        print(f"  mean: {q4['mean_steps']:.1f} steps")
    print("  entry points:", q4["entry_point_mix"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "churn": churn}
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    try:
        from evaluate import evaluate
        eval_result = evaluate()
    except Exception as e:  # eval is a nice-to-have for the chart, not a hard dependency
        print(f"\n(skipping eval-agreement chart: {e})")
        eval_result = None

    make_charts(q1, q2, q3, q4, eval_result)
    print(f"\nWrote data/outputs/analysis_summary.json and charts to {CHART_DIR}/")


if __name__ == "__main__":
    main()
