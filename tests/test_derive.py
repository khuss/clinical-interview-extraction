"""Test for the deterministic derive stage — pure Python, no LLM involved, so
this is the cheapest possible test to keep green and among the most important
to trust (this is the code that produces the actual numbers PharmaCorp gets
quoted)."""
from __future__ import annotations

from derive import referral_steps
from schema import Completeness, Encounter, EncounterType, PatientRecord


def _record(encounters, completeness=Completeness.COMPLETE):
    return PatientRecord(patient_id="TEST", encounters=encounters, completeness=completeness)


def test_referral_steps_merges_consecutive_same_provider_visits():
    """P010/P044 case: an initial visit and a later follow-up with the SAME
    encounter_type (e.g. two GI-specialist encounters, the first diagnosing and
    the second where the biologic gets raised) is ONE referral hop, not two —
    continuity of care with one provider over time isn't a new step. This is a
    real bug this test caught during golden-set evaluation: a naive "count every
    encounter record" approach silently inflated the count by one per follow-up."""
    encs = [
        Encounter(encounter_type=EncounterType.GP_PCP, sequence_index=1),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, sequence_index=2),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, biologic_decision_point=True, sequence_index=3),
    ]
    assert referral_steps(_record(encs)).value == 2  # GP + GI, not GP + GI + GI
