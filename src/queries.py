"""EXTRA / BONUS — not required by the brief. Turn the typed records into
queryable tables, and answer ad-hoc questions beyond PharmaCorp's four fixed
ones — a demonstration that the structured output is *usable* for open-ended
analysis, not just the four questions it was built to answer. Safe to skip;
the four required answers live in analysis.py and are documented in the
README's Analysis section.

The schema flattens into three frames:
  - patients  : one row per patient (wide; demographics, disease, biologic, derived)
  - treatments: one row per (patient, drug)  -> 'who tried X', 'what worked'
  - encounters: one row per (patient, visit)  -> pathway queries

Every ad-hoc question is then 2-3 lines of pandas. Uncertainty is respected by
filtering on the `*_basis` (Status) columns, so 'unknown' never silently counts.

Run:  python src/queries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from golden_set import RECORDS  # noqa: E402
from derive import normalize_drug_name, referral_steps  # noqa: E402
from schema import PatientRecord  # noqa: E402

USEFUL = {"effective", "partial"}  # what "found it useful" means


def to_frames(records: list[PatientRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pats, treats, encs = [], [], []
    for r in records:
        pats.append({
            "patient_id": r.patient_id,
            "age": r.profile.age.value,
            "years_since_dx": r.profile.years_since_diagnosis.value,
            "occupation": r.profile.occupation.value,
            "comorbidities": r.profile.comorbidities,
            "location": r.disease.location.value.value if r.disease.location.value else None,
            "severity": r.disease.severity.value.value if r.disease.severity.value else None,
            "severity_basis": r.disease.severity.status.value,
            "symptoms": [s.value for s in r.disease.symptoms],
            "disease_control": r.disease.disease_control.value.value if r.disease.disease_control.value else None,
            "disease_control_basis": r.disease.disease_control.status.value,
            "on_biologic": r.on_biologic,
            "biologic_status": r.biologic.status.value.value if r.biologic.status.value else None,
            "biologic_basis": r.biologic.status.status.value,
            "completeness": r.completeness.value,
            "churn": r.churn,
            "referral_steps": referral_steps(r).value,
        })
        for t in r.treatments:
            treats.append({
                "patient_id": r.patient_id,
                "seq": t.sequence_index,
                "name_raw": t.name_raw,
                "drug": normalize_drug_name(t.name_raw),
                "drug_class": t.drug_class.value if t.drug_class else None,
                "outcome": t.outcome.value if t.outcome else None,
                "is_current": t.is_current,
                "reason_stopped": t.reason_stopped,
            })
        for e in r.encounters:
            encs.append({
                "patient_id": r.patient_id,
                "seq": e.sequence_index,
                "type": e.encounter_type.value,
                "biologic_decision_point": e.biologic_decision_point,
                "on_referral_path": e.on_referral_path,
            })
    return pd.DataFrame(pats), pd.DataFrame(treats), pd.DataFrame(encs)


def q_drug_usefulness(treats: pd.DataFrame, name: str) -> None:
    """Of people who tried <drug>, how many found it useful?"""
    canon = normalize_drug_name(name)
    sub = treats[treats["drug"] == canon]
    known = sub[sub["outcome"].notna()]
    useful = known[known["outcome"].isin(USEFUL)]
    print(f'\nQ) Of people who tried "{name}", how many found it useful?')
    print(f"   tried: {len(sub)}  |  outcome known: {len(known)}  |  useful (effective/partial): {len(useful)}")
    print(sub[["patient_id", "name_raw", "outcome"]].to_string(index=False))


def q_best_treatment_for_symptom(pats: pd.DataFrame, treats: pd.DataFrame, symptom: str) -> None:
    """Among patients with <symptom>, which treatment class had the best outcomes?
    NB: this is a patient-level ASSOCIATION (drug targets the disease, not the symptom),
    and at n=4 it's only illustrative."""
    ids = pats[pats["symptoms"].apply(lambda s: symptom in s)]["patient_id"]
    sub = treats[treats["patient_id"].isin(ids) & treats["outcome"].notna()]
    print(f'\nQ) Among patients with "{symptom}", which class had the best outcomes?')
    if sub.empty:
        print("   (no outcome-known treatments for that symptom in this sample)")
        return
    tbl = sub.assign(useful=sub["outcome"].isin(USEFUL)).groupby("drug_class")["useful"].agg(["mean", "count"])
    tbl = tbl.rename(columns={"mean": "useful_rate", "count": "n"}).sort_values("useful_rate", ascending=False)
    print(tbl.round(2).to_string())


def q_remission_by_tenure(pats: pd.DataFrame, years: float) -> None:
    """For people sick > <years>, how many in remission vs still struggling?"""
    sub = pats[(pats["years_since_dx"].notna()) & (pats["years_since_dx"] > years)]
    sub = sub[sub["disease_control_basis"].isin(["stated", "inferred"])]
    print(f"\nQ) Of patients diagnosed > {years} years ago, disease control breakdown:")
    if sub.empty:
        print("   (none in this 4-record sample)")
        return
    print(sub["disease_control"].value_counts().to_string())
    print(sub[["patient_id", "years_since_dx", "disease_control"]].to_string(index=False))


def main() -> None:
    pats, treats, encs = to_frames(RECORDS)
    print("patients frame columns:\n ", list(pats.columns))
    print("\ntreatments frame (long — one row per drug):")
    print(treats[["patient_id", "seq", "name_raw", "drug_class", "outcome", "is_current"]].to_string(index=False))

    q_drug_usefulness(treats, "mesalamine")
    q_best_treatment_for_symptom(pats, treats, "diarrhea")
    q_remission_by_tenure(pats, 5)


if __name__ == "__main__":
    main()
