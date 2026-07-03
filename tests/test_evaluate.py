"""Test for evaluate.py's per-field scoring rules — the logic that turns raw
model/gold disagreements into the Q1-Q4 headline percentages, so it's worth
pinning down independently of any actual extraction run."""
from __future__ import annotations

from evaluate import score_field


def test_barriers_partial_overlap_scores_between_zero_and_one():
    # 1 shared out of 2 union -> Jaccard 0.5, not a hard 0 (exact-set would fail this).
    score = score_field("barriers", ["cost_insurance", "patient_fear"], ["cost_insurance"])
    assert 0.0 < score < 1.0
    assert score == 0.5
