"""Test for the response-parsing helper in extract.py — the layer that has to
survive whatever formatting quirk the model adds around its JSON (prose, markdown
fences, both), before validation even gets a chance to run."""
from __future__ import annotations

from extract import _coerce_json


def test_coerce_json_strips_markdown_fences():
    raw = '```json\n{"patient_id": "P001"}\n```'
    assert _coerce_json(raw) == '{"patient_id": "P001"}'
