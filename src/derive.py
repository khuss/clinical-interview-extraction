"""Deterministic derivations — the trustworthy, LLM-free core.

Everything here is pure Python over an already-extracted PatientRecord, so it's
reproducible and unit-testable. The LLM does fuzzy reading; this does the exact
counting, normalizing, and the headline Q1/Q4 numbers.
"""

from __future__ import annotations

from schema import BiologicStatus, DrugClass, PatientRecord, Signal, Status

# Canonical drug names: brand/generic/synonym -> canonical. Closes the gap where
# per-drug queries ("who tried mesalamine") string-matched raw text. Lets Asacol /
# Pentasa / Lialda all resolve to mesalamine. Matching is substring-based (see
# normalize_drug_name), so the "generic name maps to itself" rows below (e.g.
# "adalimumab": "adalimumab") are NOT dead weight: they strip dosage/formulation
# cruft a model might append to an already-generic name raw string (e.g.
# "Adalimumab 40mg injection" -> "adalimumab"), the same cleanup job the brand-name
# rows do for brand names. Without them, an unmatched raw string falls through to
# the fallback and is returned unchanged (uncleaned) rather than canonicalized.
_DRUG_ALIASES: dict[str, str] = {
    "humira": "adalimumab", "adalimumab": "adalimumab",
    "remicade": "infliximab", "infliximab": "infliximab",
    "stelara": "ustekinumab", "ustekinumab": "ustekinumab",
    "entyvio": "vedolizumab", "vedolizumab": "vedolizumab",
    "skyrizi": "risankizumab", "risankizumab": "risankizumab",
    "cimzia": "certolizumab", "certolizumab": "certolizumab",
    "simponi": "golimumab", "golimumab": "golimumab",
    "tysabri": "natalizumab", "natalizumab": "natalizumab",
    "rinvoq": "upadacitinib", "upadacitinib": "upadacitinib",
    "mesalamine": "mesalamine", "mesalazine": "mesalamine", "asacol": "mesalamine",
    "pentasa": "mesalamine", "lialda": "mesalamine", "5-asa": "mesalamine",
    "sulfasalazine": "sulfasalazine", "azulfidine": "sulfasalazine",
    "azathioprine": "azathioprine", "imuran": "azathioprine",
    "6-mp": "mercaptopurine", "mercaptopurine": "mercaptopurine",
    "methotrexate": "methotrexate",
    "prednisone": "prednisone", "prednisolone": "prednisone",
    "budesonide": "budesonide", "entocort": "budesonide",
}


def normalize_drug_name(name_raw: str) -> str:
    """Map a raw drug mention to a canonical name; unknowns pass through lower-cased."""
    key = name_raw.strip().lower()
    for alias, canon in _DRUG_ALIASES.items():
        if alias in key:
            return canon
    return key


def on_biologic(rec: PatientRecord) -> bool | None:
    """Q1 headline as a single function of biologic.status. None == genuinely unknown."""
    return rec.on_biologic


def referral_steps(rec: PatientRecord) -> Signal[int]:
    """Longest chain to the biologic prescriber/discusser (the terminal node).
    Off-path side-consults don't count. Consecutive on-path visits to the SAME
    encounter_type (e.g. an initial GI visit and a later follow-up with that same
    GI where the biologic gets raised) collapse into ONE step -- continuity of
    care with one provider over time isn't a new referral hop, it's the same hop
    revisited. This is what a naive "count every encounter record" approach gets
    wrong: the LLM sometimes emits a separate Encounter for a later visit with a
    provider already introduced, which would otherwise silently inflate the count
    by one per follow-up (found via golden-set evaluation on P010/P044). No
    terminal node reached -> UNKNOWN_CHURNED if truncated, else NOT_MENTIONED."""
    decision_idx = None
    for enc in rec.encounters:
        if enc.biologic_decision_point:  # set when a biologic was prescribed OR discussed
            decision_idx = enc.sequence_index  # last one wins = end of chain
    if decision_idx is None:
        status = Status.UNKNOWN_CHURNED if rec.churn else Status.NOT_MENTIONED
        return Signal(status=status)

    on_path = sorted(
        (e for e in rec.encounters if e.on_referral_path and (e.sequence_index or 0) <= decision_idx),
        key=lambda e: e.sequence_index or 0,
    )
    count = 0
    prev_type = None
    for e in on_path:
        if e.encounter_type != prev_type:
            count += 1
            prev_type = e.encounter_type
    return Signal(value=count, status=Status.STATED,
                  evidence=f"{count} on-path steps (consecutive same-provider-type visits "
                           "merged) up to the biologic decision point")


def treatment_sequence(rec: PatientRecord) -> list[str]:
    """Ordered drug classes tried BEFORE a biologic was started (Q3)."""
    out: list[str] = []
    for t in sorted(rec.treatments, key=lambda x: x.sequence_index or 0):
        if t.drug_class == DrugClass.BIOLOGIC:
            break
        if t.drug_class:
            out.append(t.drug_class.value)
    return out


def q1_range(records: list[PatientRecord]) -> tuple[float, float, dict[str, int]]:
    """Q1 as a band, driven by the epistemic basis itself:
      lower = STATED on only (quoted, certain);
      upper = STATED on + INFERRED on + UNKNOWN (everything that *could* be on).
    An INFERRED 'not_on' is treated as off — it doesn't widen the band.

    Also returns (inside the detail dict, NOT as a 4th tuple element — keeps the
    call signature stable) a confidence_weighted_estimate: a single point inside
    the band that uses `confidence` numerically for the first time anywhere in
    this codebase. A STATED 'on' counts at full weight (it's a direct quote, no
    reason to discount it); an INFERRED 'on' counts at the model's own
    self-reported confidence (0.5 if it didn't give one) instead of counting
    every inferred case identically. This is explicitly a SECONDARY, softer
    number — it trusts the model's confidence calibration, which we haven't
    independently validated, so the lower/upper band (driven only by the
    discrete stated/inferred/unknown split, no trust in the confidence number
    required) remains the primary, auditable answer."""
    n = len(records) or 1

    def st(r: PatientRecord):
        return r.biologic.status

    stated_on = sum(1 for r in records if st(r).value is BiologicStatus.ON and st(r).status is Status.STATED)
    inferred_on = sum(1 for r in records if st(r).value is BiologicStatus.ON and st(r).status is Status.INFERRED)
    unknown = sum(1 for r in records if r.on_biologic is None)
    off = len(records) - stated_on - inferred_on - unknown

    confidence_weighted = float(stated_on) + sum(
        (st(r).confidence if st(r).confidence is not None else 0.5)
        for r in records if st(r).value is BiologicStatus.ON and st(r).status is Status.INFERRED
    )

    return stated_on / n, (stated_on + inferred_on + unknown) / n, {
        "stated_on": stated_on, "inferred_on": inferred_on, "unknown": unknown, "off": off, "n": len(records),
        "confidence_weighted_estimate": confidence_weighted / n,
    }
