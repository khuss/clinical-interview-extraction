"""Schema validation tests — the LLM-adjacent test the challenge asks for.

We don't call a real model here (no network, no flakiness, no cost). Instead we
validate PatientRecord against a JSON string SHAPED like what extract_one actually
receives back from the model (same keys, same nesting, same enum value strings),
which is exactly the boundary that breaks in practice: a model omits a field, sends
an enum value we didn't expect, or nests something slightly wrong. If PatientRecord
can't parse real-shaped output, extract_one's "validate, retry once" loop is the
thing that would catch it at run time — this test is the offline version of that
same check.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schema import BiologicStatus, PatientRecord, Status

# A trimmed but realistic model response: same shape produced by extract_one for a
# patient who is clearly ON a biologic, with the uncertainty fields populated the
# way the SYSTEM_PROMPT instructs (status/evidence/confidence on every signal).
SAMPLE_LLM_RESPONSE = json.dumps({
    "patient_id": "P999",
    "profile": {
        "name": "Test Patient",
        "age": {"value": 40, "status": "stated", "evidence": "I'm 40", "confidence": 0.99},
        "years_since_diagnosis": {"value": 2.0, "status": "stated", "evidence": "two years ago"},
        "occupation": {"value": None, "status": "not_mentioned"},
        "comorbidities": [],
    },
    "disease": {
        "location": {"value": "ileal", "status": "stated", "evidence": "in my ileum"},
        "severity": {"value": "moderate", "status": "inferred", "confidence": 0.5},
        "symptoms": ["abdominal_pain", "diarrhea"],
        "disease_control": {"value": "partial", "status": "stated"},
    },
    "treatments": [
        {"name_raw": "Humira", "drug_class": "biologic", "outcome": "effective",
         "is_current": True, "sequence_index": 1},
    ],
    "encounters": [
        {"encounter_type": "gp_pcp", "sequence_index": 1},
        {"encounter_type": "gi_specialist", "biologic_decision_point": True, "sequence_index": 2},
    ],
    "biologic": {
        "status": {"value": "on", "status": "stated", "evidence": "on Humira", "confidence": 0.95},
        "drug_name": "Humira",
        "barriers": [],
        "barriers_status": "not_mentioned",
    },
    "completeness": "complete",
})


def test_validates_realistic_llm_response():
    """The exact boundary extract_one crosses: raw JSON text -> validated PatientRecord."""
    rec = PatientRecord.model_validate_json(SAMPLE_LLM_RESPONSE)
    assert rec.patient_id == "P999"
    assert rec.biologic.status.value is BiologicStatus.ON
    assert rec.biologic.status.status is Status.STATED
    # The computed properties are what the analysis and evaluate.py actually consume.
    assert rec.on_biologic is True
    assert rec.churn is False
    assert len(rec.treatments) == 1
    assert rec.treatments[0].drug_class.value == "biologic"


def test_invalid_enum_value_is_rejected():
    """A hallucinated/out-of-vocabulary enum value must fail validation loudly —
    this is exactly what triggers extract_one's one-shot self-correction retry,
    so it needs to actually raise, not silently coerce to None."""
    bad = json.dumps({
        "patient_id": "P997",
        "biologic": {"status": {"value": "kind_of_on_i_guess", "status": "stated"}},
    })
    with pytest.raises(ValidationError):
        PatientRecord.model_validate_json(bad)
