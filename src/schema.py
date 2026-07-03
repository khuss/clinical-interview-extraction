"""Pydantic schema for the the company Crohn's extraction challenge.

Design goals
------------
1. Every fact carries *why it's blank* (4-state Status), not just a value.
2. Every fact carries *proof* (an evidence snippet) so a reviewer can audit.
3. The shape mirrors the 4 PharmaCorp questions and is "merge-friendly"
   (normalized enums + ordered sequences) so records could later be combined
   across re-interviews.
4. The LLM fills the raw signals; a deterministic Python stage fills
   `DerivedMetrics`. The LLM never counts referral steps or decides the
   headline biologic yes/no — that's done in code, testably.

Retrospective vs ongoing updates
--------------------------------
A record is a *snapshot as of the interview*. When a patient's story updates
later, `Status` is the merge key — no field is ever overwritten:
  - STATED / DENIED      -> settled fact; a later update only appends, never re-asks.
  - UNKNOWN_CHURNED      -> the "ongoing" slot: story stopped here, follow up.
  - NOT_MENTIONED        -> complete but silent; fill only if genuinely new.
So "the story changes over time" is handled by resolving UNKNOWN_CHURNED
fields, not by mutating STATED ones.

Derived from hand-labelling P001, P002, P047, P048.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Uncertainty primitives
# ---------------------------------------------------------------------------


class Status(str, Enum):
    """The epistemic basis for a field — *how* we know it, or *why* it's blank.

    Two axes live here: whether there's a value, and what backs it.
      - has a value:  STATED (direct quote) / INFERRED (reasoned) / DENIED (explicit no)
      - no value:     NOT_MENTIONED (never came up) / UNKNOWN_CHURNED (cut off first)

    The hard split is NOT_MENTIONED vs UNKNOWN_CHURNED: both have no value, but
    they differ in *why*, and that difference is what the client pays for. A field
    is only UNKNOWN_CHURNED if the transcript was truncated AND the field was
    plausibly still to come; an explicitly-evidenced field stays STATED even in a
    truncated transcript. INFERRED carries a value we derived rather than quoted
    (e.g. 'not_on' from present-tense current meds) — confidence carries the degree.
    """

    STATED = "stated"  # value given verbatim in the transcript
    INFERRED = "inferred"  # value reasoned from context, not directly quoted
    DENIED = "denied"  # explicitly ruled out ("no", "never tried")
    NOT_MENTIONED = "not_mentioned"  # complete transcript, never came up
    UNKNOWN_CHURNED = "unknown_churned"  # transcript cut off before this surfaced


class Signal(BaseModel, Generic[T]):
    """A single fact = value + why-it's-blank + proof + how-sure.

    Used for every scalar/categorical field so uncertainty is first-class
    and never silently collapses into ``None``.
    """

    value: T | None = None
    status: Status = Status.NOT_MENTIONED
    evidence: str | None = Field(
        default=None, description="Verbatim snippet from the transcript supporting `value`."
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="LLM self-reported confidence in `value`."
    )


# ---------------------------------------------------------------------------
# Controlled vocabularies (enums) — the match-friendly dimensions
# ---------------------------------------------------------------------------


class BiologicStatus(str, Enum):
    """Q1 headline. Kept separate from the 4-state Status:
    Status says *do we know*; this says *what we know*."""

    ON = "on"  # currently taking a biologic
    NOT_ON = "not_on"  # confidently not on one as of the interview
    PLANNED = "planned"  # recommended / next step, not yet started (e.g. P048)
    DISCUSSED_REJECTED = "discussed_rejected"  # raised then declined/ruled out


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    MODERATE_TO_SEVERE = "moderate_to_severe"
    SEVERE = "severe"


class DiseaseLocation(str, Enum):
    ILEAL = "ileal"
    COLONIC = "colonic"
    ILEOCOLONIC = "ileocolonic"
    SMALL_INTESTINE_OTHER = "small_intestine_other"
    UPPER_GI = "upper_gi"
    PERIANAL = "perianal"


class DrugClass(str, Enum):
    AMINOSALICYLATE = "aminosalicylate"  # mesalamine, sulfasalazine
    CORTICOSTEROID = "corticosteroid"  # prednisone, budesonide
    IMMUNOMODULATOR = "immunomodulator"  # azathioprine, 6-MP, methotrexate
    BIOLOGIC = "biologic"  # Humira, Remicade, Stelara, Entyvio, ...
    JAK_INHIBITOR = "jak_inhibitor"  # Rinvoq
    ANTIBIOTIC = "antibiotic"
    OTHER = "other"


class TreatmentOutcome(str, Enum):
    EFFECTIVE = "effective"
    PARTIAL = "partial"  # some improvement, e.g. "maybe 30%"
    INEFFECTIVE = "ineffective"
    ADVERSE_REACTION = "adverse_reaction"  # stopped due to side effects
    ONGOING_UNKNOWN = "ongoing_unknown"  # currently on it, verdict not yet in
    NOT_STARTED = "not_started"  # discussed/planned but not begun


class BarrierType(str, Enum):
    """Q2: why a patient is not (yet) on a biologic."""

    STEP_THERAPY = "step_therapy"  # must fail other meds first
    NOT_SEVERE_ENOUGH = "not_severe_enough"  # doctor judges not yet indicated
    PATIENT_FEAR = "patient_fear"  # injection / side-effect fear
    COST_INSURANCE = "cost_insurance"  # denied coverage, affordability
    ACCESS = "access"  # referral / availability friction
    SURGERY_FIRST = "surgery_first"  # biologic on hold pending surgery (P048)
    NOT_YET_INDICATED = "not_yet_indicated"  # too early in the journey
    COMORBIDITY_RISK = "comorbidity_risk"  # another condition makes it too risky (e.g. COPD + infection risk, P050) — doctor-driven veto, distinct from NOT_SEVERE_ENOUGH (which is about the Crohn's itself, not a co-existing condition)
    CHURNED_BEFORE_KNOWN = "churned_before_known"  # cut off before reason surfaced


class EncounterType(str, Enum):
    """Q4: nodes in the referral pathway, in order."""

    GP_PCP = "gp_pcp"  # family doctor / primary care
    ER = "er"  # emergency presentation
    GI_SPECIALIST = "gi_specialist"  # gastroenterologist (can prescribe biologic)
    SURGEON = "surgeon"
    OTHER_SPECIALIST = "other_specialist"
    UNKNOWN = "unknown"


class DiseaseControl(str, Enum):
    """Replaces a subjective 1-5 score with an auditable coarse band."""

    REMISSION = "remission"
    PARTIAL = "partial"
    POOR = "poor"


class SymptomFlag(str, Enum):
    """Hallmark Crohn's symptoms as normalized flags (not free text) so they're
    usable as match-keys for patient similarity. This is the CLINICAL vocabulary,
    not just what appeared in any one transcript, so the model always has a
    correct slot to map a mention to (and absence stays absence).
    """

    # Luminal / GI
    ABDOMINAL_PAIN = "abdominal_pain"
    DIARRHEA = "diarrhea"
    BLOODY_DIARRHEA = "bloody_diarrhea"  # rectal bleeding / bloody stool
    URGENCY = "urgency"  # urgency / tenesmus
    CONSTIPATION = "constipation"  # often with strictures
    # Systemic / constitutional
    WEIGHT_LOSS = "weight_loss"
    FATIGUE = "fatigue"
    FEVER = "fever"
    NIGHT_SWEATS = "night_sweats"
    APPETITE_LOSS = "appetite_loss"
    NAUSEA_VOMITING = "nausea_vomiting"
    ANEMIA = "anemia"
    MALNUTRITION = "malnutrition"  # B12 / iron / vit-D deficiency, etc.
    # Structural / complications
    BOWEL_OBSTRUCTION = "bowel_obstruction"  # stricture / blockage
    FISTULA = "fistula"
    PERIANAL_DISEASE = "perianal_disease"  # abscess / fissure
    # Extra-intestinal manifestations (EIMs)
    MOUTH_ULCERS = "mouth_ulcers"
    JOINT = "joint"  # arthralgia / arthritis
    SKIN = "skin"  # erythema nodosum, pyoderma, psoriasis assoc.
    EYE = "eye"  # uveitis / episcleritis
    EXTRA_INTESTINAL_OTHER = "extra_intestinal_other"  # e.g. PSC, other EIMs


class Completeness(str, Enum):
    """The single transcript-level signal. `churn` is derived from this (= TRUNCATED),
    so we don't carry the same fact twice. TRUNCATED is the gate for UNKNOWN_CHURNED."""

    COMPLETE = "complete"  # narrative reaches a natural stopping point
    TRUNCATED = "truncated"  # cut off mid-thought (the patient churned)
    AMBIGUOUS = "ambiguous"  # unclear whether complete


# ---------------------------------------------------------------------------
# Structured sub-objects
# ---------------------------------------------------------------------------


class PersonalProfile(BaseModel):
    """Socio-demographics. Name is captured but excluded from analysis (PII);
    drop it before any aggregation/sharing."""

    name: str | None = Field(default=None, description="PII — do not use in analysis.")
    age: Signal[int] = Field(default_factory=Signal)
    years_since_diagnosis: Signal[float] = Field(default_factory=Signal)
    occupation: Signal[str] = Field(default_factory=Signal)
    comorbidities: list[str] = Field(
        default_factory=list,
        description="Other conditions, e.g. diabetes, hypothyroid, depression, psoriasis.",
    )


class DiseaseProfile(BaseModel):
    location: Signal[DiseaseLocation] = Field(default_factory=Signal)
    severity: Signal[Severity] = Field(default_factory=Signal)
    symptoms: list[SymptomFlag] = Field(
        default_factory=list, description="Normalized hallmark-symptom flags (match-keys)."
    )
    disease_control: Signal[DiseaseControl] = Field(default_factory=Signal)


class Treatment(BaseModel):
    """One medication tried or discussed. Ordered list answers Q3."""

    name_raw: str = Field(description="Drug name as said in the transcript.")
    drug_class: DrugClass | None = None  # normalized in the deterministic stage
    outcome: TreatmentOutcome | None = None
    reason_stopped: str | None = None
    is_current: bool | None = Field(
        default=None, description="Patient still on it as of the interview."
    )
    sequence_index: int | None = Field(
        default=None, description="Order in the treatment journey (1 = first)."
    )
    evidence: str | None = None


class Encounter(BaseModel):
    """One clinical touchpoint. Ordered list answers Q4."""

    encounter_type: EncounterType
    specialty_raw: str | None = None
    biologic_decision_point: bool = Field(
        default=False,
        description="True if this is where the biologic was PRESCRIBED or DISCUSSED/"
        "recommended — the terminal node of the referral chain (Q4).",
    )
    on_referral_path: bool = Field(
        default=True,
        description="False for a side-consult / dead-end that did NOT refer onward; "
        "such nodes don't count toward the chain length.",
    )
    sequence_index: int | None = None
    evidence: str | None = None


class Barrier(BaseModel):
    """One reason a patient is not (yet) on a biologic. A patient can have
    several at once (P048: surgery-first AND cost/insurance)."""

    type: BarrierType
    evidence: str | None = None
    overcome: bool | None = Field(
        default=None, description="True if resolved (e.g. P001 insurance won on appeal)."
    )


class BiologicAssessment(BaseModel):
    """Everything PharmaCorp's Q1/Q2 hinge on, in one place."""

    status: Signal[BiologicStatus] = Field(default_factory=Signal)
    drug_name: str | None = Field(default=None, description="Biologic taken/proposed, if named.")
    barriers: list[Barrier] = Field(
        default_factory=list, description="Q2: all reasons not (yet) on a biologic (can be >1)."
    )
    barriers_status: Status = Field(
        default=Status.NOT_MENTIONED,
        description="4-state for the barrier question itself: UNKNOWN_CHURNED when the "
        "reason was cut off (P047), NOT_MENTIONED when N/A (already on a biologic).",
    )


# ---------------------------------------------------------------------------
# Deterministic derivations (filled by Python, NOT the LLM)
# ---------------------------------------------------------------------------


class DerivedMetrics(BaseModel):
    """Computed in the deterministic stage from the raw signals above, so the
    headline numbers are reproducible and testable (no hallucination).

    NB: the Q1 yes/no is NOT stored here — it's a single function of
    `biologic.status` exposed as `PatientRecord.on_biologic`, to avoid keeping
    the same fact in two places. This object only holds *aggregations* that
    combine several raw fields (steps, sequence)."""

    referral_steps_gp_to_prescriber: Signal[int] = Field(
        default_factory=Signal, description="Q4: count of steps; UNKNOWN_CHURNED if truncated."
    )
    treatment_sequence: list[str] = Field(
        default_factory=list, description="Ordered drug classes before biologic, for Q3."
    )


# ---------------------------------------------------------------------------
# Top-level record
# ---------------------------------------------------------------------------


class PatientRecord(BaseModel):
    """One row per patient_id (50 total). The LLM populates everything except
    `derived`, which the deterministic stage fills afterward."""

    patient_id: str
    profile: PersonalProfile = Field(default_factory=PersonalProfile)
    disease: DiseaseProfile = Field(default_factory=DiseaseProfile)
    treatments: list[Treatment] = Field(default_factory=list)
    encounters: list[Encounter] = Field(default_factory=list)
    biologic: BiologicAssessment = Field(default_factory=BiologicAssessment)

    completeness: Completeness = Completeness.AMBIGUOUS

    derived: DerivedMetrics | None = None  # filled by the deterministic stage
    extraction_notes: str | None = Field(
        default=None, description="Free-text caveats from the model for the reviewer."
    )

    @property
    def churn(self) -> bool:
        """Derived, single source of truth = completeness. The transcript was cut
        off mid-thought, which is what gates a field's UNKNOWN_CHURNED status."""
        return self.completeness is Completeness.TRUNCATED

    @property
    def on_biologic(self) -> bool | None:
        """Q1 headline, the ONE place the rule lives. None == genuinely unknown
        (e.g. churned before we could tell), so it's excluded from the % rather
        than silently counted as a 'no'."""
        match self.biologic.status.value:
            case BiologicStatus.ON:
                return True
            case BiologicStatus.NOT_ON | BiologicStatus.PLANNED | BiologicStatus.DISCUSSED_REJECTED:
                return False
            case _:
                return None
