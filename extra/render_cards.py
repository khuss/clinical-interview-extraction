"""Renders a handful of extracted PatientRecords as a static, self-contained HTML
"card" view — no server, no JS framework, just data -> HTML — so a reviewer can see
the schema's uncertainty states (stated/inferred/denied/not_mentioned/unknown_churned)
laid out visually instead of squinting at raw JSON.

Purely a documentation aid: it reads whatever is already in `data/cache/{id}.json`
(the real extraction pipeline's output) and does not call an LLM or change any data.

Run:  python src/render_cards.py [PATIENT_ID ...]
Default patient set (no args): P001 (clean, on-biologic), P047 (churned/truncated),
P048 (known extraction bug, kept in on purpose — see README known limitations).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE_DIR = Path("data/cache")
OUT_PATH = Path("data/outputs/patient_cards.html")

DEFAULT_PATIENTS = ["P001", "P047", "P048"]

STATUS_COLOR = {
    "stated": "#3fb950",
    "inferred": "#3fb950",
    "denied": "#58a6ff",
    "not_mentioned": "#8b949e",
    "unknown_churned": "#d29922",
}

TREATMENT_COLOR = {
    "effective": "#2ea043",
    "partial": "#9e6a03",
    "ineffective": "#8b949e",
    "adverse_reaction": "#b62324",
    "ongoing_unknown": "#9e6a03",
    "not_started": "#8b949e",
    None: "#8b949e",
}

ENCOUNTER_COLOR_DECISION = "#1f6feb"
ENCOUNTER_COLOR_ONPATH = "#30363d"
ENCOUNTER_COLOR_OFFPATH = "#21262d"


def esc(s) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def badge(status: str, extra_label: str | None = None) -> str:
    color = STATUS_COLOR.get(status, "#8b949e")
    label = extra_label or status
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;">{esc(label)}</span>'
    )


def signal_row(label: str, sig: dict, unit: str = "") -> str:
    value = sig.get("value")
    status = sig.get("status", "not_mentioned")
    evidence = sig.get("evidence")
    confidence = sig.get("confidence")
    value_str = "&mdash;" if value is None else f"{esc(value)}{unit}"
    conf_str = f'{confidence:.2f}' if confidence is not None else "not set"
    evidence_html = (
        f'<div style="color:#8b949e;font-size:13px;margin-top:2px;border-left:2px solid #30363d;'
        f'padding-left:8px;">&ldquo;{esc(evidence)}&rdquo;</div>'
        if evidence
        else '<div style="color:#586069;font-size:13px;font-style:italic;margin-top:2px;">no evidence recorded</div>'
    )
    return f"""
    <div style="padding:10px 0;border-bottom:1px solid #21262d;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <div style="color:#8b949e;font-size:13px;min-width:160px;">{esc(label)}</div>
        <div style="flex:1;font-size:15px;font-weight:600;">{value_str} {badge(status)}</div>
        <div style="color:#8b949e;font-size:12px;">{conf_str}</div>
      </div>
      {evidence_html}
    </div>"""


def treatment_chain_html(treatments: list[dict]) -> str:
    boxes, cards = [], []
    ordered = sorted(treatments, key=lambda t: t.get("sequence_index") or 0)
    for i, t in enumerate(ordered):
        color = TREATMENT_COLOR.get(t.get("outcome"), "#8b949e")
        current = " · current" if t.get("is_current") else ""
        boxes.append(f"""
        <div style="background:{color}22;border:1px solid {color};border-radius:8px;
                    padding:10px 14px;min-width:140px;">
          <div style="font-weight:700;">{t.get('sequence_index', i+1)} &middot; {esc(t.get('name_raw'))}</div>
          <div style="color:{color};font-size:12px;">{esc(t.get('drug_class'))} &middot; {esc(t.get('outcome'))}{current}</div>
        </div>""")
        if i < len(ordered) - 1:
            boxes.append('<div style="color:#8b949e;font-size:20px;">&rarr;</div>')
        reason = t.get("reason_stopped")
        detail = f"stopped: {reason}" if reason else ("current" if t.get("is_current") else "outcome recorded, not current")
        evidence = t.get("evidence")
        cards.append(f"""
        <div style="padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;">
          {badge('stated')} <b>{esc(t.get('name_raw'))}</b> &mdash; {esc(detail)}
          {f'<div style="color:#8b949e;margin-top:2px;">&ldquo;{esc(evidence)}&rdquo;</div>' if evidence else ''}
        </div>""")
    row = f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">{"".join(boxes)}</div>'
    return row + "".join(cards)


def referral_chain_html(encounters: list[dict]) -> str:
    ordered = sorted(encounters, key=lambda e: e.get("sequence_index") or 0)
    on_path_count = 0
    prev_type = None
    boxes = []
    decision_idx = next((e.get("sequence_index") for e in ordered if e.get("biologic_decision_point")), None)
    for i, e in enumerate(ordered):
        is_decision = e.get("biologic_decision_point")
        on_path = e.get("on_referral_path", True)
        color = ENCOUNTER_COLOR_DECISION if is_decision else (ENCOUNTER_COLOR_ONPATH if on_path else ENCOUNTER_COLOR_OFFPATH)
        note = "biologic decision point" if is_decision else ("on path" if on_path else "off path (side consult)")
        if on_path and (decision_idx is None or (e.get("sequence_index") or 0) <= decision_idx):
            if e.get("encounter_type") != prev_type:
                on_path_count += 1
                prev_type = e.get("encounter_type")
        boxes.append(f"""
        <div style="background:{color}22;border:1px solid {color};border-radius:8px;
                    padding:10px 14px;min-width:150px;">
          <div style="font-weight:700;">{e.get('sequence_index', i+1)} &middot; {esc(e.get('encounter_type'))}</div>
          <div style="color:{color};font-size:12px;">{esc(e.get('specialty_raw') or '')} &middot; {esc(note)}</div>
        </div>""")
        if i < len(ordered) - 1:
            boxes.append('<div style="color:#8b949e;font-size:20px;">&rarr;</div>')
    row = f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">{"".join(boxes)}</div>'
    footer = (
        f'<div style="color:#8b949e;font-size:13px;margin-top:10px;border-left:2px solid #30363d;padding-left:8px;">'
        f'{on_path_count} on-path step(s) up to the biologic decision point (consecutive same-type visits merged) '
        f'&rarr; referral_steps = {on_path_count}</div>'
        if decision_idx is not None
        else '<div style="color:#8b949e;font-size:13px;margin-top:10px;">no biologic decision point reached in this transcript</div>'
    )
    return row + footer


def card_html(rec: dict) -> str:
    pid = rec["patient_id"]
    name = rec.get("profile", {}).get("name") or "?"
    completeness = rec.get("completeness", "ambiguous")
    churned = completeness == "truncated"
    bio = rec.get("biologic", {})
    bio_status = bio.get("status", {})
    on_biologic = {"on": "Yes", "not_on": "No", "planned": "No (planned)", "discussed_rejected": "No (discussed, rejected)"}.get(
        bio_status.get("value"), "Unknown"
    )
    treatments = rec.get("treatments", [])
    encounters = rec.get("encounters", [])
    n_before_biologic = 0
    for t in sorted(treatments, key=lambda t: t.get("sequence_index") or 0):
        if t.get("drug_class") == "biologic":
            break
        n_before_biologic += 1

    barrier_rows = "".join(
        f"""
        <div style="padding:8px 0;border-bottom:1px solid #21262d;display:flex;justify-content:space-between;">
          <div><b>{esc(b.get('type'))}</b>
            <div style="color:#8b949e;font-size:13px;">&ldquo;{esc(b.get('evidence'))}&rdquo;</div>
          </div>
          <div>{'&#10003; overcome' if b.get('overcome') else 'not overcome'}</div>
        </div>"""
        for b in bio.get("barriers", [])
    ) or '<div style="color:#586069;font-size:13px;font-style:italic;">no barriers recorded</div>'

    return f"""
  <section style="background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:24px;margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;">
      <div>
        <span style="font-size:20px;font-weight:700;">{esc(pid)} &middot; {esc(name)}</span>
        <span style="background:#21262d;border-radius:999px;padding:2px 10px;font-size:12px;margin-left:8px;">
          &#128274; name = PII, excluded from analysis</span>
      </div>
      <div>
        {badge('not_mentioned' if not churned else 'unknown_churned', completeness)}
        <span style="background:#21262d;border-radius:999px;padding:2px 10px;font-size:12px;margin-left:6px;">
          churn: {'yes' if churned else 'no'}</span>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;">
      <div style="background:#161b22;border-radius:8px;padding:12px;">
        <div style="color:#8b949e;font-size:12px;">on biologic &middot; Q1</div>
        <div style="font-size:22px;font-weight:700;">{on_biologic}</div>
        <div style="color:#8b949e;font-size:12px;">status: {esc(bio_status.get('value'))} &middot; conf {bio_status.get('confidence', 'n/a')}</div>
      </div>
      <div style="background:#161b22;border-radius:8px;padding:12px;">
        <div style="color:#8b949e;font-size:12px;">referral steps &middot; Q4</div>
        <div style="font-size:22px;font-weight:700;">{len(encounters)} touchpoint(s)</div>
        <div style="color:#8b949e;font-size:12px;">see referral pathway below</div>
      </div>
      <div style="background:#161b22;border-radius:8px;padding:12px;">
        <div style="color:#8b949e;font-size:12px;">lines before biologic &middot; Q3</div>
        <div style="font-size:22px;font-weight:700;">{n_before_biologic}</div>
        <div style="color:#8b949e;font-size:12px;">{len(treatments)} treatment(s) total</div>
      </div>
    </div>

    <h4 style="margin:20px 0 8px;color:#c9d1d9;">treatment journey (Q3)</h4>
    {treatment_chain_html(treatments) if treatments else '<div style="color:#586069;font-style:italic;">no treatments recorded</div>'}

    <h4 style="margin:20px 0 8px;color:#c9d1d9;">referral pathway (Q4)</h4>
    {referral_chain_html(encounters) if encounters else '<div style="color:#586069;font-style:italic;">no encounters recorded</div>'}

    <h4 style="margin:20px 0 8px;color:#c9d1d9;">biologic assessment (Q1 / Q2)</h4>
    {signal_row('biologic status', bio_status)}
    <div style="margin-top:10px;">{barrier_rows}</div>
    <div style="color:#8b949e;font-size:12px;margin-top:4px;">barriers_status: {esc(bio.get('barriers_status'))}</div>
  </section>"""


def render(patient_ids: list[str]) -> str:
    cards = []
    for pid in patient_ids:
        f = CACHE_DIR / f"{pid}.json"
        if not f.exists():
            cards.append(f'<p style="color:#b62324;">{esc(pid)}: not found in {CACHE_DIR}/</p>')
            continue
        cards.append(card_html(json.loads(f.read_text())))
    body = "\n".join(cards)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Patient record cards</title></head>
<body style="background:#010409;color:#e6edf3;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
             padding:32px;max-width:900px;margin:0 auto;">
  <h2 style="color:#c9d1d9;">Example extracted patient records</h2>
  <p style="color:#8b949e;">
    Generated by <code>src/render_cards.py</code> directly from <code>data/cache/*.json</code> —
    the real pipeline output, not hand-typed. Colors: green = stated/effective/on-path,
    amber = unknown/churned/partial, blue = denied/decision-point, gray = not_mentioned/off-path.
    See the README's Schema design section for what each field means.
  </p>
  {body}
</body></html>"""


def main() -> None:
    patient_ids = sys.argv[1:] or DEFAULT_PATIENTS
    html = render(patient_ids)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH} ({len(patient_ids)} patient(s))")


if __name__ == "__main__":
    main()
