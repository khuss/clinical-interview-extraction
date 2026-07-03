"""The golden set: hand-built PatientRecords + their label dicts, used by evaluate.py.

Kept as Python (not JSON) deliberately: every field below carries an inline comment
justifying the judgment call behind it (why a symptom was excluded, why a barrier is
comorbidity_risk and not not_severe_enough, why an encounter collapses into a prior
one). That reasoning is real signal for a reviewer — JSON has no comment syntax, so
converting would mean throwing it away or bolting on a parallel notes structure that
can drift out of sync with the field it explains. Pydantic construction also
validates every record at import time for free.

IMPORTANT — what's actually "gold" here vs what isn't:
  - P001/P002/P047/P048/P015/P031/P004/P038/P024/P045 (`RECORDS`, `HAND_LABELS`) are
    Matsumoto's own hand labels — genuine independent human ground truth. THIS is the
    golden set `evaluate.py` scores against by default.
  - P010/P019/P020/P044/P049/P050 (`AI_DRAFT_RECORDS`, `AI_DRAFT_LABELS`) were
    drafted by an AI assistant reading the transcripts during this session, NOT
    independently human-verified. Checking the extraction model against labels
    an LLM wrote defeats the point of a golden set (model-checking-model, not a
    real independent check) — these are kept separate, clearly marked, and
    EXCLUDED from `evaluate.py`'s default golden set. They're here as a
    starting draft in case they're useful once reviewed/corrected/replaced by
    real hand labels — not to be cited as evaluation results on their own.

The PatientRecord objects still prove the schema can represent the real, messy
data and exercise the deterministic derive stage regardless of which bucket
they're in — that part doesn't depend on who wrote the labels.

Run:  python src/golden_set.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schema import (  # noqa: E402
    Barrier,
    BarrierType,
    BiologicAssessment,
    BiologicStatus,
    Completeness,
    DiseaseControl,
    DiseaseLocation,
    DiseaseProfile,
    DrugClass,
    Encounter,
    EncounterType,
    PatientRecord,
    PersonalProfile,
    Severity,
    Signal,
    Status,
    SymptomFlag,
    Treatment,
    TreatmentOutcome,
)


def S(value=None, status=Status.STATED, evidence=None, confidence=None):  # noqa: N802
    """Terse Signal constructor for readability below."""
    return Signal(value=value, status=status, evidence=evidence, confidence=confidence)


# ---------------------------------------------------------------------------
# P001 — Sarah: complete, on biologic, barrier overcome
# ---------------------------------------------------------------------------

P001 = PatientRecord(
    patient_id="P001",
    profile=PersonalProfile(
        name="Sarah",
        age=S(34, evidence="34 years old", confidence=0.99),
        years_since_diagnosis=S(8.0, evidence="Crohn's for about eight years now"),
        occupation=S("marketing manager", evidence="working as a marketing manager in Chicago"),
        comorbidities=["psoriasis"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="inflammation in my ileum and parts of my colon"),
        severity=S(Severity.MODERATE_TO_SEVERE, evidence="moderate to severe Crohn's like mine", confidence=0.9),
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.DIARRHEA, SymptomFlag.WEIGHT_LOSS, SymptomFlag.SKIN],
        disease_control=S(DiseaseControl.REMISSION, evidence="mostly in remission"),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="not much improvement after three months", is_current=False, sequence_index=1, evidence="started with mesalamine ... wasn't much improvement"),
        Treatment(name_raw="azathioprine", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="severe nausea, liver enzymes up", is_current=False, sequence_index=2, evidence="tried azathioprine, which actually made me feel worse"),
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.EFFECTIVE, is_current=True, sequence_index=3, evidence="on Humira for five years now"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="Dr. Martinez (primary care)", sequence_index=1, evidence="My primary care doctor, Dr. Martinez"),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="Dr. Chen (gastroenterologist)", biologic_decision_point=True, sequence_index=2, evidence="referred me to a gastroenterologist, Dr. Rebecca Chen"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="on Humira for five years now", confidence=0.98),
        drug_name="Humira",
        barriers=[
            Barrier(type=BarrierType.PATIENT_FEAR, evidence="I was terrified of the injections and the potential side effects - increased infection risk, possible lymphoma", overcome=True),
            Barrier(type=BarrierType.COST_INSURANCE, evidence="insurance initially denied coverage", overcome=True),
        ],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P002 — Michael: complete, NOT on biologic, blocked by cost (unresolved)
# ---------------------------------------------------------------------------

P002 = PatientRecord(
    patient_id="P002",
    profile=PersonalProfile(
        name="Michael",
        age=S(42, evidence="I'm 42"),
        years_since_diagnosis=S(3.0, evidence="digestive issues started about three years ago"),
        occupation=S("high school teacher", evidence="I'm a high school teacher"),
        comorbidities=["type 2 diabetes", "anemia", "arthritis", "prior minor stroke"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="terminal ileum and ascending colon"),
        # severity is INFERRED (biologic recommended, multiple failed lines),
        # never stated verbatim -> Status.INFERRED + low confidence.
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="Dr. Patel wanted to start me on a biologic", confidence=0.45),
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.BLOODY_DIARRHEA, SymptomFlag.ANEMIA, SymptomFlag.NAUSEA_VOMITING, SymptomFlag.JOINT],
        disease_control=S(DiseaseControl.POOR, evidence="Sometimes I wonder if the cure is worse than the disease"),
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, reason_stopped="wreaked havoc on blood sugar control", is_current=False, sequence_index=1, evidence="Started on prednisone for immediate relief"),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="no improvement", is_current=False, sequence_index=2, evidence="tried mesalamine, no improvement"),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="liver wasn't happy", is_current=False, sequence_index=3, evidence="Moved to methotrexate, but my liver wasn't happy"),
        Treatment(name_raw="budesonide", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=4, evidence="I'm on budesonide, which helps somewhat"),
        Treatment(name_raw="Stelara", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=5, evidence="wanted to start me on a biologic, specifically Stelara"),
    ],
    encounters=[
        # NOTE: entry point is an endocrinologist, NOT a GP -> Q4 'GP-to-prescriber' is ill-posed here.
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="endocrinologist", sequence_index=1, evidence="my endocrinologist thought it was diabetic gastroparesis"),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="Dr. Patel (gastroenterologist)", biologic_decision_point=True, sequence_index=2, evidence="Dr. Patel keeps pushing for the biologic"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.NOT_ON, evidence="currently trying to get financial assistance ... Meanwhile, I'm on budesonide", confidence=0.9),
        drug_name="Stelara",
        barriers=[Barrier(type=BarrierType.COST_INSURANCE, evidence="tier 4 ... $800 per month ... not feasible", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P047 — Donna: CHURNED, not on biologic (never raised), early in journey
# ---------------------------------------------------------------------------

P047 = PatientRecord(
    patient_id="P047",
    profile=PersonalProfile(
        name="Donna",
        age=S(41, evidence="Donna, 41, female"),  # transcript says 41 (hand-label said 45)
        years_since_diagnosis=S(0.42, evidence="Diagnosed with Crohn's five months ago"),
        occupation=S("office manager", evidence="working as office manager"),
        comorbidities=["hypothyroid", "depression", "obesity"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.SMALL_INTESTINE_OTHER, evidence="Found Crohn's in small intestine"),
        severity=S(status=Status.NOT_MENTIONED),  # never characterized
        symptoms=[],  # no hallmark Crohn symptoms enumerated (years of misdiagnosis)
        disease_control=S(DiseaseControl.POOR, evidence="some improvement in symptoms. Maybe 30%"),
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, reason_stopped="gained 30 pounds", is_current=False, sequence_index=1, evidence="Started on prednisone ... gain 30 pounds"),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=False, sequence_index=2, evidence="Started on prednisone and mesalamine"),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.PARTIAL, reason_stopped=None, is_current=True, sequence_index=3, evidence="Trying methotrexate now ... some improvement, maybe 30%"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="various (3 yrs misdiagnosis as IBS)", sequence_index=1, evidence="Spent three years being told it was IBS"),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="good GI", biologic_decision_point=False, sequence_index=2, evidence="Finally got to good GI who actually listened"),
    ],
    biologic=BiologicAssessment(
        # INFERRED not_on: within one sitting she lists current meds in present tense
        # (methotrexate) with no biologic among them -> probably not on one. Truncation
        # caps confidence (the list may be unfinished), but doesn't erase the inference.
        # The REASON she's not on one was never reached, so the barrier stays churned.
        status=S(BiologicStatus.NOT_ON, status=Status.INFERRED, confidence=0.55,
                 evidence="Trying methotrexate now (present tense); no biologic mentioned"),
        drug_name=None,
        barriers=[],
        barriers_status=Status.UNKNOWN_CHURNED,
    ),
    completeness=Completeness.TRUNCATED,
)


# ---------------------------------------------------------------------------
# P048 — Larry: CHURNED, biologic PLANNED (on hold for surgery)
# ---------------------------------------------------------------------------

P048 = PatientRecord(
    patient_id="P048",
    profile=PersonalProfile(
        name="Larry",
        age=S(56, evidence="Larry, 56"),
        years_since_diagnosis=S(0.06, evidence="found out I have Crohn's three weeks ago"),
        occupation=S("former construction company owner", evidence="man who used to run construction company"),
        comorbidities=["diabetes", "heart disease"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # "inflammation in intestines", no segment named
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="Emergency surgery to remove blocked section", confidence=0.5),
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.BOWEL_OBSTRUCTION],  # "severe pain", "blocked section"
        disease_control=S(DiseaseControl.POOR, evidence="Now can barely make it to bathroom"),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=True, sequence_index=1, evidence="Mesalamine and antibiotics for now"),
        Treatment(name_raw="antibiotics", drug_class=DrugClass.ANTIBIOTIC, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=True, sequence_index=2, evidence="Mesalamine and antibiotics for now"),
        Treatment(name_raw="biologic (unnamed)", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=3, evidence="Doctor wants biologics after I heal from surgery"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.ER, specialty_raw="ER (suspected heart attack)", sequence_index=1, evidence="Went to ER with severe pain"),
        # Surgery happened DURING the same ER admission -> same node, not a separate
        # referral hop, so it's off-path and doesn't add to the chain length.
        Encounter(encounter_type=EncounterType.SURGEON, specialty_raw="emergency surgery (same ER admission)", on_referral_path=False, sequence_index=2, evidence="Emergency surgery to remove blocked section"),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="doctor managing biologic plan", biologic_decision_point=True, sequence_index=3, evidence="Doctor wants biologics after I heal"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.PLANNED, evidence="Doctor wants biologics after I heal from surgery", confidence=0.85),
        drug_name=None,
        # #4: NOT just surgery — cost/insurance is a second, active barrier.
        barriers=[
            Barrier(type=BarrierType.SURGERY_FIRST, evidence="biologics after I heal from surgery", overcome=False),
            Barrier(type=BarrierType.COST_INSURANCE, evidence="Insurance fighting everything ... May have to sell house", overcome=False),
        ],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.TRUNCATED,
)


# ---------------------------------------------------------------------------
# P015 — Caroline: complete, ON biologic (3rd line), 15-year journey, insurance
# denial overcome. Matsumoto's hand label.
# ---------------------------------------------------------------------------

P015 = PatientRecord(
    patient_id="P015",
    profile=PersonalProfile(
        name="Caroline",
        age=S(37, evidence="I'm 37 years old"),
        years_since_diagnosis=S(15.0, evidence="been dealing with Crohn's disease for almost 15 years now"),
        occupation=S("teacher", evidence="starting my first real job as a teacher"),
        comorbidities=["chronic migraine syndrome", "fibromyalgia"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # "cobblestone road" imagery, no segment named
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="ended up hospitalized with partial bowel obstruction ... some days I can barely function", confidence=0.5),
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.DIARRHEA, SymptomFlag.WEIGHT_LOSS, SymptomFlag.FATIGUE, SymptomFlag.BOWEL_OBSTRUCTION, SymptomFlag.ANEMIA],
        disease_control=S(DiseaseControl.PARTIAL, evidence="It's okay, not great. Still have frequent symptoms but avoiding surgery."),
    ),
    treatments=[
        Treatment(name_raw="fiber supplements", drug_class=DrugClass.OTHER, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="made everything worse (given for a misdiagnosis of IBS)", is_current=False, sequence_index=1, evidence="she said IBS and gave me fiber supplements. Made everything worse."),
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, reason_stopped="gained 30 pounds, couldn't sleep, mood swings — tapered once mesalamine took over", is_current=False, sequence_index=2, evidence="The prednisone was horrible ... But it did calm the inflammation."),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="flaring again within three months", is_current=False, sequence_index=3, evidence="Tried to maintain on just mesalamine but within three months I was flaring again."),
        Treatment(name_raw="azathioprine", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="developed severe joint pain", is_current=False, sequence_index=4, evidence="Added azathioprine. Worked for about a year until I developed severe joint pain ... had to stop the medication."),
        Treatment(name_raw="Remicade", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="stopped working during her second pregnancy", is_current=False, sequence_index=5, evidence="Remicade worked wonderfully for about four years ... during my second pregnancy, it stopped working."),
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="failed after three years", is_current=False, sequence_index=6, evidence="started Humira. ... Humira worked for three years ... When Humira failed"),
        Treatment(name_raw="Cimzia", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, reason_stopped="insurance denied it three times, never started", is_current=False, sequence_index=7, evidence="tried Cimzia. Insurance denied it three times."),
        Treatment(name_raw="Stelara", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=8, evidence="Finally got approved for Stelara. Been on it two years. It's okay, not great. Still have frequent symptoms."),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=9, evidence="Doctor mentioned combination therapy next - adding methotrexate to Stelara."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="doctor (misdiagnosed IBS)", sequence_index=1, evidence="When I finally saw a doctor, she said IBS and gave me fiber supplements."),
        # One continuous ~15-year GI relationship (colonoscopy through Remicade,
        # Humira, the Cimzia denial, and current Stelara) -- kept as ONE encounter,
        # not one per medication change, per the provider-continuity rule: this is
        # the same specialist over time, not a new referral each time treatment changes.
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI doctor", biologic_decision_point=True, sequence_index=2, evidence="demanded a referral to a specialist. The GI doctor ordered a colonoscopy immediately."),
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="rheumatologist", on_referral_path=False, sequence_index=3, evidence="Rheumatologist couldn't determine if it was drug-induced or Crohn's-related arthritis."),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="Finally got approved for Stelara. Been on it two years.", confidence=0.95),
        drug_name="Stelara",
        barriers=[Barrier(type=BarrierType.COST_INSURANCE, evidence="tried Cimzia. Insurance denied it three times ... Finally got approved for Stelara.", overcome=True)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P031 — Patricia: complete, ON biologic (2nd line, Stelara), cancer interruption.
# Matsumoto's hand label.
# ---------------------------------------------------------------------------

P031 = PatientRecord(
    patient_id="P031",
    profile=PersonalProfile(
        name="Patricia",
        age=S(46, evidence="Patricia, 46"),
        years_since_diagnosis=S(15.0, evidence="Crohn's for fifteen years ... Crohn's diagnosis at 31"),
        occupation=S("teacher (now on disability)", evidence="Was teacher. Loved it. But between three conditions, can't maintain schedule. On disability."),
        comorbidities=["fibromyalgia", "breast cancer (survivor, cancer-free two years)", "depression (untreated)"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # no segment named
        severity=S(Severity.MODERATE_TO_SEVERE, evidence="Moderate to severe Crohn's throughout"),
        symptoms=[SymptomFlag.BOWEL_OBSTRUCTION, SymptomFlag.FATIGUE],
        # NOTE: "Pain everywhere" is attributed to fibromyalgia, not the Crohn's itself
        # -> not flagged as ABDOMINAL_PAIN to avoid conflating a comorbidity's symptom
        # with the disease under study.
        disease_control=S(DiseaseControl.POOR, evidence="nowhere near remission. Probably never will be ... Quality of life terrible."),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.PARTIAL, reason_stopped=None, is_current=False, sequence_index=1, evidence="Started standard treatment. Mesalamine, prednisone for flares. Managed okay for few years."),
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, reason_stopped=None, is_current=False, sequence_index=2, evidence="Mesalamine, prednisone for flares."),
        Treatment(name_raw="Remicade", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.EFFECTIVE, reason_stopped="had to stop all immunosuppressants for chemo (breast cancer diagnosis)", is_current=False, sequence_index=3, evidence="Moved to biologics seven years ago. Remicade first. Worked well for three years. Then cancer diagnosis ... Had to stop all immunosuppressants for chemo."),
        Treatment(name_raw="Stelara", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=4, evidence="Started Stelara after cancer treatment. ... Working somewhat. Better than nothing. But nowhere near remission."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="misdiagnosed as stomach flu", sequence_index=1, evidence="Started with presumed stomach flu that wouldn't resolve. Six months of misdiagnosis before finally getting colonoscopy."),
        # One continuing GI relationship spans diagnosis through Remicade and the
        # later Stelara restart -- kept as ONE encounter per the provider-continuity
        # rule (same specialist over time, not a new referral each time treatment changes).
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI (colonoscopy, Remicade, Stelara)", biologic_decision_point=True, sequence_index=2, evidence="finally getting colonoscopy. Moderate to severe Crohn's throughout. ... Moved to biologics seven years ago. Remicade first."),
        # Off-path: comorbidity workup (fibromyalgia), never referred onward re: the biologic.
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="rheumatologist (fibromyalgia)", on_referral_path=False, sequence_index=3, evidence="Doctors initially dismissive. Finally rheumatologist confirmed."),
        # Off-path: oncology consult during/after cancer treatment -- raised the
        # comorbidity-risk question but didn't refer her anywhere new for the Crohn's.
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="oncologist (breast cancer)", on_referral_path=False, sequence_index=4, evidence="Oncologist nervous about immunosuppression but agreed quality of life matters."),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="Started Stelara after cancer treatment ... but now she's on it", confidence=0.9),
        drug_name="Stelara",
        # The cancer/immunosuppression risk was a real, stated obstacle to restarting
        # a biologic -- ultimately overridden by the oncologist, so overcome=True.
        # NOTE: no cost_insurance barrier recorded -- the transcript's cost mention
        # ("cost was high") is explicitly hedged ("not sure if Crohn's or cancer")
        # and general financial strain (disability income) isn't a stated reason she
        # wasn't on a biologic, per the stated-only barrier rule.
        barriers=[Barrier(type=BarrierType.COMORBIDITY_RISK, evidence="Oncologist nervous about immunosuppression but agreed quality of life matters.", overcome=True)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P004 — Robert: complete, biologic PLANNED (Entyvio recommended), blocked by
# catastrophic-coverage cost. Matsumoto's hand label.
# ---------------------------------------------------------------------------

P004 = PatientRecord(
    patient_id="P004",
    profile=PersonalProfile(
        name="Robert",
        age=S(55, evidence="Robert here, 55 years old"),
        years_since_diagnosis=S(0.5, evidence="just diagnosed with Crohn's six months ago"),
        occupation=S("self-employed construction business owner", evidence="running my construction business ... I'm self-employed"),
        comorbidities=["hypertension", "high cholesterol", "depression (about the diagnosis)"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.COLONIC, evidence="pretty extensive involvement of my colon"),
        # No severity word ever used verbatim -- inferred from the biologic being
        # recommended plus "extensive involvement", same reasoning basis as P002.
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="extensive involvement of my colon ... now he wants me on a biologic", confidence=0.45),
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.FATIGUE, SymptomFlag.JOINT],
        disease_control=S(DiseaseControl.POOR, evidence="trying to manage with diet changes and budesonide, but I know it's not enough"),
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, reason_stopped="not a long-term solution", is_current=False, sequence_index=1, evidence="Started me on prednisone immediately, which helped, but he said it's not a long-term solution."),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.PARTIAL, reason_stopped="minimal improvement", is_current=False, sequence_index=2, evidence="Tried mesalamine next, minimal improvement."),
        Treatment(name_raw="sulfasalazine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="terrible headaches", is_current=False, sequence_index=3, evidence="Then sulfasalazine, which gave me terrible headaches."),
        Treatment(name_raw="budesonide", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=4, evidence="trying to manage with diet changes and budesonide, but I know it's not enough"),
        Treatment(name_raw="Entyvio", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=5, evidence="Now he wants me on a biologic, recommended Entyvio."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.ER, specialty_raw="ER (collapsed at job site)", sequence_index=1, evidence="Emergency room visit led to CT scan showing significant inflammation. They referred me to a GI specialist."),
        # One continuing GI relationship (colonoscopy diagnosis through the Entyvio
        # recommendation) -- kept as ONE encounter per the provider-continuity rule.
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="Dr. Williams (gastroenterologist)", biologic_decision_point=True, sequence_index=2, evidence="Dr. Williams did a colonoscopy and confirmed Crohn's disease ... Now he wants me on a biologic, recommended Entyvio."),
    ],
    biologic=BiologicAssessment(
        # Recommended as the clear next step, not yet started (blocked by cost) --
        # same semantics as P048's PLANNED, not NOT_ON (which implies no biologic is
        # even on the table).
        status=S(BiologicStatus.PLANNED, evidence="Now he wants me on a biologic, recommended Entyvio", confidence=0.85),
        drug_name="Entyvio",
        barriers=[Barrier(type=BarrierType.COST_INSURANCE, evidence="catastrophic coverage only. The medication would cost me $5,000 per infusion, every 8 weeks. That's impossible.", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P038 — Daniel: complete, ON biologic (Humira, 2nd line), triple-comorbidity
# gridlock blocking escalation to Stelara. Matsumoto's hand label.
# ---------------------------------------------------------------------------

P038 = PatientRecord(
    patient_id="P038",
    profile=PersonalProfile(
        name="Daniel",
        age=S(37, evidence="Daniel, 37"),
        years_since_diagnosis=S(10.0, evidence="Crohn's for ten years now"),
        occupation=S("software architect (now on disability)", evidence="Was software architect. High-paying job. Had to go on disability."),
        comorbidities=["ankylosing spondylitis", "chronic kidney disease (from prior NSAID use)"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="Crohn's throughout small bowel, some colon"),
        # No severity word ever used for the Crohn's itself -- inferred from biologics
        # being started immediately at diagnosis (same reasoning basis as P002/P004).
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="Started biologics immediately ... Crohn's throughout small bowel, some colon", confidence=0.4),
        # NOTE: fatigue is explicitly attributed to the kidney disease ("Fatigue from
        # kidney disease"), not the Crohn's -- excluded here for the same reason
        # P031's fibromyalgia pain was excluded, to avoid crediting a comorbidity's
        # symptom to the disease under study.
        symptoms=[SymptomFlag.URGENCY],
        disease_control=S(DiseaseControl.POOR, evidence="Quality of life terrible. Can't sit or stand long from AS. Need bathroom constantly from Crohn's."),
    ),
    treatments=[
        # NSAIDs (rheumatologist, for AS/back pain, pre-diagnosis, the cause of his
        # kidney disease) are excluded -- comorbidity treatment, not an IBD treatment.
        Treatment(name_raw="Remicade", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.EFFECTIVE, reason_stopped="stopped working", is_current=False, sequence_index=1, evidence="Started biologics immediately. Remicade for both conditions. Worked initially. ... Then Remicade stopped working."),
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=2, evidence="Switched to Humira. Less effective. Maybe 40% improvement."),
        Treatment(name_raw="Stelara", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=3, evidence="Doc wants to try Stelara. Might help both conditions. Kidney doctor reluctant."),
    ],
    encounters=[
        # Off-path: the rheumatologist visit was for AS/back pain (a different
        # condition) -- it indirectly caused the kidney damage but didn't itself
        # refer him toward the Crohn's diagnosis/biologic decision. The transcript
        # never narrates an explicit referral chain to the Crohn's diagnosis (unlike
        # other patients' "referred me to a GI specialist" language) -- Matsumoto
        # flagged this ambiguity directly during dictation ("I think the second
        # step... but we're not sure"), so this is a documented judgment call, not
        # a clean-cut fact.
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="rheumatologist (AS/back pain, pre-Crohn's-diagnosis)", on_referral_path=False, sequence_index=1, evidence="Back pain first. Rheumatologist prescribed NSAIDs."),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI (diagnosed both conditions)", biologic_decision_point=True, sequence_index=2, evidence="Finally diagnosed with both conditions. ... Started biologics immediately."),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="he did try biologics ... he is on biologics", confidence=0.95),
        drug_name="Humira",
        # He IS on a biologic (Humira) -- this barrier blocks the ESCALATION to a
        # potentially better one (Stelara), not initial access. Recorded for
        # completeness; Q2's aggregate excludes him anyway since on_biologic=True.
        barriers=[Barrier(type=BarrierType.COMORBIDITY_RISK, evidence="Doc wants to try Stelara ... Kidney doctor reluctant ... nephrologist said kidneys couldn't handle anything stronger", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
    extraction_notes=(
        "Patient expresses passive suicidal ideation tied to disease burden and "
        "specialist-treatment gridlock (kidney disease blocking biologic escalation); "
        "noted for clinical-review awareness, not used in any business-question scoring."
    ),
)


# ---------------------------------------------------------------------------
# P024 — Derek: complete, biologic PLANNED (step-therapy blocked), likely
# UNDERCOUNTED referral chain (see extraction_notes). Matsumoto's hand label.
# ---------------------------------------------------------------------------

P024 = PatientRecord(
    patient_id="P024",
    profile=PersonalProfile(
        name="Derek",
        age=S(40, evidence="Derek, 40"),
        # NOTE: dictation said "he had it for four years" but the actual transcript
        # only says "just diagnosed three months ago but probably had it for years"
        # -- no specific "four years" figure appears anywhere. Using the stated
        # diagnosis date, not the misremembered number.
        years_since_diagnosis=S(0.25, evidence="just diagnosed three months ago"),
        occupation=S("plumber", evidence="I'm a plumber, need to be at job sites, not toilets"),
        comorbidities=["pre-diabetic", "sleep apnea"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # "inflammation everywhere" -- no segment named
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="Found inflammation everywhere. Said I'm lucky no complications given how long untreated.", confidence=0.45),
        symptoms=[SymptomFlag.DIARRHEA, SymptomFlag.URGENCY, SymptomFlag.WEIGHT_LOSS],
        disease_control=S(DiseaseControl.POOR, evidence="Mesalamine alone not enough. Having 15-20 bowel movements daily. ... Losing weight rapidly."),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="alone not enough", is_current=False, sequence_index=1, evidence="Started mesalamine and prednisone. ... Mesalamine alone not enough."),
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="blood sugar went crazy (pre-diabetic to almost diabetic); tapered off quick", is_current=False, sequence_index=2, evidence="Prednisone helped but blood sugar went crazy. ... Endocrinologist not happy. Tapered off quick."),
        Treatment(name_raw="sulfasalazine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=True, sequence_index=3, evidence="Currently trying sulfasalazine. Headaches terrible but pushing through."),
        Treatment(name_raw="biologic (unnamed)", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=4, evidence="Doctor wants to try biologics but insurance saying need to fail more drugs first."),
    ],
    encounters=[
        # NOTE (Matsumoto flagged this directly during dictation): the transcript
        # says "doctors blame everything on weight" (plural, implying an unspecified
        # earlier diagnostic odyssey with prior doctors) before "Finally, new doctor
        # took symptoms seriously" -- those earlier visits are never itemized (no
        # count, specialty, or sequence given), so only the one doctor who actually
        # ordered the colonoscopy AND is now discussing biologics is encoded here.
        # referral_steps=1 below is therefore a likely UNDERCOUNT, not a confident 1
        # -- see extraction_notes.
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="new doctor (ordered colonoscopy, now discussing biologics)", biologic_decision_point=True, sequence_index=1, evidence="Finally, new doctor took symptoms seriously. Ordered colonoscopy. ... Doctor wants to try biologics but insurance saying need to fail more drugs first."),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.PLANNED, evidence="Doctor wants to try biologics but insurance saying need to fail more drugs first ... Going to push hard for biologics", confidence=0.85),
        drug_name=None,
        barriers=[Barrier(type=BarrierType.STEP_THERAPY, evidence="insurance saying need to fail more drugs first", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
    extraction_notes=(
        "Transcript references an unspecified number of earlier dismissive doctors "
        "('doctors blame everything on weight') before the 'new doctor' who diagnosed "
        "him -- not itemized (no count/specialty/sequence given), so referral_steps=1 "
        "here is a likely undercount of the true referral chain, not a confident count."
    ),
)


# ---------------------------------------------------------------------------
# P045 — Rosa: TRUNCATED, biologic PLANNED (step-therapy blocked, methotrexate
# currently required first). Matsumoto's hand label.
# ---------------------------------------------------------------------------

P045 = PatientRecord(
    patient_id="P045",
    profile=PersonalProfile(
        name="Rosa",
        age=S(33, evidence="Rosa here, 33 years old"),
        # NOTE: diagnosis (4 months ago) and symptom-onset (~2 years ago) are
        # different dates -- the schema only has one `years_since_diagnosis` slot,
        # so the ~2-year symptom-onset gap isn't captured anywhere. Worth logging
        # as a further-improvement candidate (a distinct "years_since_symptom_onset"
        # field), per Matsumoto's own note during dictation.
        years_since_diagnosis=S(0.33, evidence="Got diagnosed with Crohn's about four months ago"),
        occupation=S("elementary school teacher", evidence="Work as elementary school teacher."),
        comorbidities=["PCOS", "insulin resistance"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="inflammation in terminal ileum and ascending colon"),
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="Doctor mentioned biologics might be next step ... Some improvement but not much ... Already had two close calls.", confidence=0.45),
        # NOTE: the 40 lbs weight loss was patient-initiated (doctors told her to
        # lose weight to fix the "digestive issues"), not clearly a disease-driven
        # symptom -- excluded from WEIGHT_LOSS to avoid over-crediting an
        # intentional diet change as a Crohn's symptom.
        symptoms=[SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.URGENCY],
        disease_control=S(DiseaseControl.POOR, evidence="Some improvement but not much. ... Already had two close calls. ... Sometimes the pain is so bad I can't move."),
    ),
    treatments=[
        Treatment(name_raw="budesonide", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=1, evidence="Started on budesonide and mesalamine. Some improvement but not much."),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=2, evidence="Started on budesonide and mesalamine. Some improvement but not much."),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=True, sequence_index=3, evidence="Currently trying methotrexate. The nausea is terrible. Taking folic acid helps a bit."),
        Treatment(name_raw="biologic (unnamed)", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=4, evidence="Doctor mentioned biologics might be next step but insurance wants me to fail more medications first."),
    ],
    encounters=[
        # An unspecified number of dismissive prior doctors preceded diagnosis
        # ("Doctors kept blaming everything on my weight") -- not itemized (no
        # distinct names/specialties/sequence given), so collapsed into one GP
        # encounter rather than inventing separate nodes. referral_steps=2 below
        # may undercount the true pre-diagnosis odyssey (same class of ambiguity
        # as P024) -- see extraction_notes.
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="doctor(s), pre-diagnosis (weight blamed for symptoms)", sequence_index=1, evidence="Doctors kept blaming everything on my weight. ... Finally demanded colonoscopy."),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI (colonoscopy, biologics discussion)", biologic_decision_point=True, sequence_index=2, evidence="Found inflammation in terminal ileum and ascending colon. ... Doctor mentioned biologics might be next step."),
    ],
    biologic=BiologicAssessment(
        # Stated clearly earlier in the transcript (not an open deliberation cut off
        # by the truncation, which happens later during her emotional reflection) --
        # stays STATED/PLANNED despite the transcript being truncated overall.
        status=S(BiologicStatus.PLANNED, evidence="Doctor mentioned biologics might be next step but insurance wants me to fail more medications first. Currently trying methotrexate.", confidence=0.8),
        drug_name=None,
        barriers=[Barrier(type=BarrierType.STEP_THERAPY, evidence="insurance wants me to fail more medications first", overcome=False)],
        barriers_status=Status.STATED,
    ),
    # Transcript trails off mid-thought ("Trying to stay positive but it's hard...")
    # -- the textbook TRUNCATED case.
    completeness=Completeness.TRUNCATED,
    extraction_notes=(
        "Symptom onset (~2 years before the interview) predates the 4-month-old "
        "diagnosis -- the schema has no field for this gap (only years_since_diagnosis); "
        "flagged as a further-improvement candidate, not captured elsewhere in this record."
    ),
)


# ---------------------------------------------------------------------------
# AI-DRAFTED, NOT YET HUMAN-VERIFIED (see module docstring). Everything below
# this line down to AI_DRAFT_RECORDS/AI_DRAFT_LABELS was read and labeled by an
# AI assistant, not Matsumoto — pending review before it can count as gold.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P010 — Mark: complete, ON biologic, insurance step-therapy overcome
# ---------------------------------------------------------------------------

P010 = PatientRecord(
    patient_id="P010",
    profile=PersonalProfile(
        name="Mark",
        age=S(48, evidence="Mark here, 48"),
        years_since_diagnosis=S(5.0, evidence="diagnosed with Crohn's five years ago"),
        occupation=S("middle management", evidence="Work in middle management"),
        comorbidities=["type 2 diabetes", "enteropathic arthritis"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="Colonoscopy revealed severe ileocolonic Crohn's"),
        severity=S(Severity.SEVERE, evidence="severe ileocolonic Crohn's", confidence=0.85),
        symptoms=[SymptomFlag.DIARRHEA, SymptomFlag.WEIGHT_LOSS, SymptomFlag.JOINT],
        disease_control=S(DiseaseControl.PARTIAL, evidence="Life is manageable. Not great, but manageable."),
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="sent blood sugars through roof; tapered after two months", is_current=False, sequence_index=1, evidence="Started triple therapy - prednisone ... Prednisone sent blood sugars through roof"),
        Treatment(name_raw="azathioprine", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.EFFECTIVE, reason_stopped="switched approach when joint pain started", is_current=False, sequence_index=2, evidence="Azathioprine and mesalamine maintained remission for about a year"),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="wasn't really helping", is_current=False, sequence_index=3, evidence="Stopped mesalamine, wasn't really helping"),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.PARTIAL, reason_stopped=None, is_current=False, sequence_index=4, evidence="Helped joints but Crohn's flared"),
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.EFFECTIVE, is_current=True, sequence_index=5, evidence="Been on Humira two years. Life is manageable."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="primary care", sequence_index=1, evidence="Primary care ran basic tests, all normal except elevated CRP. Referred to gastroenterologist."),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="gastroenterologist", biologic_decision_point=True, sequence_index=2, evidence="Finally agreed to biologics ... Finally approved for Humira"),
        # Side consults that never referred onward toward the biologic decision — don't add to chain length.
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="endocrinologist", on_referral_path=False, sequence_index=3, evidence="Endocrinologist had to add insulin to my diabetes regimen"),
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="rheumatologist", on_referral_path=False, sequence_index=4, evidence="Rheumatologist said it's enteropathic arthritis"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="Been on Humira two years", confidence=0.98),
        drug_name="Humira",
        barriers=[
            Barrier(type=BarrierType.STEP_THERAPY, evidence="Required documentation of failure of multiple therapies", overcome=True),
            Barrier(type=BarrierType.COST_INSURANCE, evidence="Insurance nightmare began ... Appeals process took four months", overcome=True),
        ],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P019 — Natalie: complete, ON biologic, straight-to-specialist (1 step)
# ---------------------------------------------------------------------------

P019 = PatientRecord(
    patient_id="P019",
    profile=PersonalProfile(
        name="Natalie",
        age=S(24, evidence="I'm Natalie, 24"),
        years_since_diagnosis=S(status=Status.NOT_MENTIONED),  # diagnosed "recently" post-graduation, no explicit duration
        occupation=S("marketing", evidence="first job out of college is at marketing firm"),
        comorbidities=["generalized anxiety disorder"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.ILEOCOLONIC, evidence="found inflammation in small intestine and colon", confidence=0.6),
        severity=S(Severity.MODERATE_TO_SEVERE, evidence="Moderate to severe, he said"),
        symptoms=[SymptomFlag.URGENCY, SymptomFlag.FATIGUE],
        disease_control=S(DiseaseControl.PARTIAL, evidence="Humira working okay. Not perfect but better than before. Still have breakthrough symptoms."),
    ),
    treatments=[
        # No IBD treatment was tried before the biologic — she skipped straight to it.
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=1, evidence="Started Humira right away, skipping the traditional steps"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="family-friend gastroenterologist", biologic_decision_point=True, sequence_index=1, evidence="Family friend who's a GI ... Started Humira right away, skipping the traditional steps"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="Humira working okay", confidence=0.95),
        drug_name="Humira",
        barriers=[
            Barrier(type=BarrierType.STEP_THERAPY, evidence="They wanted me to fail other drugs first", overcome=True),
            Barrier(type=BarrierType.COST_INSURANCE, evidence="Insurance fight was real ... Doctor wrote letters, appeals. Finally approved.", overcome=True),
        ],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P020 — Gary: complete, ON biologic (2nd line), classic step-therapy fight
# ---------------------------------------------------------------------------

P020 = PatientRecord(
    patient_id="P020",
    profile=PersonalProfile(
        name="Gary",
        age=S(46, evidence="Gary, 46"),
        years_since_diagnosis=S(7.0, evidence="diagnosed with Crohn's about seven years ago"),
        occupation=S("construction worker", evidence="Construction worker, so this disease really impacts my livelihood"),
        comorbidities=["hypertension"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # "ulcers everywhere" — no segment named
        severity=S(Severity.SEVERE, evidence="Colonoscopy showed severe inflammation, ulcers everywhere"),
        symptoms=[SymptomFlag.BLOODY_DIARRHEA, SymptomFlag.WEIGHT_LOSS, SymptomFlag.JOINT],
        disease_control=S(DiseaseControl.POOR, evidence="Starting to notice it's not working as well. More bathroom trips, more pain."),
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="made me rage, almost lost my job", is_current=False, sequence_index=1, evidence="Prednisone made me rage ... Got off that quick"),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="didn't do much alone", is_current=False, sequence_index=2, evidence="Mesalamine didn't do much alone"),
        Treatment(name_raw="sulfasalazine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="gave me killer headaches", is_current=False, sequence_index=3, evidence="Tried sulfasalazine, gave me killer headaches"),
        Treatment(name_raw="methotrexate", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="so nauseous I couldn't work", is_current=False, sequence_index=4, evidence="methotrexate, made me so nauseous I couldn't work"),
        Treatment(name_raw="Remicade", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="effectiveness dropped, then stopped working completely", is_current=False, sequence_index=5, evidence="Finally got approved for Remicade ... Worked great for about two years, then effectiveness dropped"),
        Treatment(name_raw="Humira", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=6, evidence="Switched to Humira ... Starting to notice it's not working as well"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GP_PCP, specialty_raw="family doctor", sequence_index=1, evidence="Wife finally made me see doctor when I dropped 20 pounds"),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="gastroenterologist", biologic_decision_point=True, sequence_index=2, evidence="Doc pushed for biologics ... Finally got approved for Remicade"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.ON, evidence="Been on Humira three years now", confidence=0.95),
        drug_name="Humira",
        barriers=[
            Barrier(type=BarrierType.STEP_THERAPY, evidence="insurance wanted documentation of failing everything else first", overcome=True),
            Barrier(type=BarrierType.COST_INSURANCE, evidence="Took eight months of appeals while I suffered", overcome=True),
        ],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P044 — Peter: complete, biologic PLANNED, blocked by needle/immune fear
# ---------------------------------------------------------------------------

P044 = PatientRecord(
    patient_id="P044",
    profile=PersonalProfile(
        name="Peter",
        age=S(27, evidence="Peter, 27"),
        years_since_diagnosis=S(2.0, evidence="diagnosed two years ago"),
        occupation=S("junior architect", evidence="working as junior architect"),
        comorbidities=["ADHD", "generalized anxiety disorder"],
    ),
    disease=DiseaseProfile(
        location=S(DiseaseLocation.COLONIC, evidence="inflammation throughout sigmoid colon and rectum"),
        severity=S(status=Status.NOT_MENTIONED),  # no severity word used, only functional impact ("40% functional")
        symptoms=[SymptomFlag.URGENCY, SymptomFlag.ABDOMINAL_PAIN, SymptomFlag.BLOODY_DIARRHEA],
        disease_control=S(DiseaseControl.POOR, evidence="Can't continue at 40% functional."),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="awkward and minimal help", is_current=False, sequence_index=1, evidence="Started mesalamine suppositories and oral. Awkward and minimal help."),
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped=None, is_current=None, sequence_index=2, evidence="Added prednisone. Energy through roof but anxiety went crazy. Had panic attack."),
        Treatment(name_raw="azathioprine", drug_class=DrugClass.IMMUNOMODULATOR, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=3, evidence="Started azathioprine six months ago. Helps maybe 40%."),
        Treatment(name_raw="biologic (unnamed)", drug_class=DrugClass.BIOLOGIC, outcome=TreatmentOutcome.NOT_STARTED, is_current=False, sequence_index=4, evidence="Doctor wants to try biologics but I'm terrified"),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="gastroenterologist", biologic_decision_point=True, sequence_index=1, evidence="Finally saw GI ... Doctor wants to try biologics ... Will probably agree to biologics"),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.PLANNED, status=Status.INFERRED, evidence="Will probably agree to biologics. Can't continue at 40% functional.", confidence=0.6),
        drug_name=None,
        barriers=[Barrier(type=BarrierType.PATIENT_FEAR, evidence="terrified ... needles trigger anxiety ... What if I get sick and miss more work?", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# ---------------------------------------------------------------------------
# P049 — Tammy: CHURNED mid-decision (biologic discussed, fear stated, outcome
# never reached — the "discussed but not decided" case the prompt calls out)
# ---------------------------------------------------------------------------

P049 = PatientRecord(
    patient_id="P049",
    profile=PersonalProfile(
        name="Tammy",
        age=S(28, evidence="Tammy, 28, female"),
        years_since_diagnosis=S(0.08, evidence="Crohn's diagnosed last month"),
        occupation=S(status=Status.NOT_MENTIONED),
        comorbidities=["eating disorder (in recovery)", "anxiety"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),  # "throughout intestines" — too vague to map to a segment
        severity=S(Severity.MODERATE_TO_SEVERE, status=Status.INFERRED, evidence="GI wants to be aggressive given my age and extent of disease", confidence=0.4),
        symptoms=[],
        disease_control=S(status=Status.NOT_MENTIONED),  # "too early to tell" — not yet assessable
    ),
    treatments=[
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="made me manic, didn't sleep for four days", is_current=False, sequence_index=1, evidence="Started on prednisone in hospital. Made me manic."),
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.ONGOING_UNKNOWN, is_current=True, sequence_index=2, evidence="Now on mesalamine. Too early to tell if working."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.SURGEON, specialty_raw="surgeon (incidental diagnosis during appendectomy)", sequence_index=1, evidence="Crohn's diagnosed last month during what should have been routine appendectomy. Surgeon found inflammation throughout intestines."),
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI", biologic_decision_point=True, sequence_index=2, evidence="GI wants to be aggressive given my age and extent of disease. Mentioned biologics."),
    ],
    biologic=BiologicAssessment(
        # Discussed, but the accept/reject outcome was never reached before the cut-off
        # (transcript ends mid-word, "Chronic...") — exactly the null-value/unknown_churned case.
        status=S(status=Status.UNKNOWN_CHURNED, evidence="Mentioned biologics. But reading side effects online. Cancer risk? Serious infections?"),
        drug_name=None,
        # The REASON she's hesitating is known even though the final decision is churned —
        # barriers_status stays STATED, not unknown_churned (that's the P047 case, not this one).
        barriers=[Barrier(type=BarrierType.PATIENT_FEAR, evidence="reading side effects online. Cancer risk? Serious infections? I'm 28. Whole life ahead. Or maybe not.", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.TRUNCATED,
)


# ---------------------------------------------------------------------------
# P050 — Victor: complete, biologic DISCUSSED then blocked by a comorbidity
# (COPD) — the new COMORBIDITY_RISK barrier type
# ---------------------------------------------------------------------------

P050 = PatientRecord(
    patient_id="P050",
    profile=PersonalProfile(
        name="Victor",
        age=S(62, evidence="Victor, 62, male"),
        years_since_diagnosis=S(0.67, evidence="Diagnosed with Crohn's eight months ago"),
        occupation=S(status=Status.NOT_MENTIONED),
        comorbidities=["COPD", "arthritis"],
    ),
    disease=DiseaseProfile(
        location=S(status=Status.NOT_MENTIONED),
        severity=S(Severity.SEVERE, status=Status.INFERRED, evidence="Lost 50 pounds ... Quality of life terrible", confidence=0.5),
        symptoms=[SymptomFlag.WEIGHT_LOSS, SymptomFlag.DIARRHEA],
        disease_control=S(DiseaseControl.POOR, evidence="Quality of life terrible. Can't breathe, can't eat, can't leave house much."),
    ),
    treatments=[
        Treatment(name_raw="mesalamine", drug_class=DrugClass.AMINOSALICYLATE, outcome=TreatmentOutcome.INEFFECTIVE, reason_stopped="nothing", is_current=False, sequence_index=1, evidence="Started treatment with mesalamine. Nothing."),
        Treatment(name_raw="prednisone", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.ADVERSE_REACTION, reason_stopped="breathing got worse; respiratory doctor said absolutely not", is_current=False, sequence_index=2, evidence="Tried prednisone but breathing got worse. Respiratory doctor said absolutely not."),
        Treatment(name_raw="budesonide", drug_class=DrugClass.CORTICOSTEROID, outcome=TreatmentOutcome.PARTIAL, is_current=True, sequence_index=3, evidence="Now on budesonide. Helps a little. Maybe 20%."),
    ],
    encounters=[
        Encounter(encounter_type=EncounterType.GI_SPECIALIST, specialty_raw="GI", biologic_decision_point=True, sequence_index=1, evidence="GI wants to try biologics but lung doctor says too risky with COPD"),
        Encounter(encounter_type=EncounterType.OTHER_SPECIALIST, specialty_raw="pulmonologist (lung doctor)", on_referral_path=False, sequence_index=2, evidence="lung doctor says too risky with COPD. Might get infection that kills me."),
    ],
    biologic=BiologicAssessment(
        status=S(BiologicStatus.DISCUSSED_REJECTED, evidence="lung doctor says too risky with COPD ... So stuck with inadequate treatment.", confidence=0.75),
        drug_name=None,
        barriers=[Barrier(type=BarrierType.COMORBIDITY_RISK, evidence="lung doctor says too risky with COPD. Might get infection that kills me.", overcome=False)],
        barriers_status=Status.STATED,
    ),
    completeness=Completeness.COMPLETE,
)


# The TRUE golden set — Matsumoto's own hand labels. This is what evaluate.py
# scores against by default.
RECORDS = [P001, P002, P047, P048, P015, P031, P004, P038, P024, P045]

# AI-drafted, not human-verified — see module docstring. Kept separate so
# nothing accidentally treats these as gold. Combine explicitly
# (RECORDS + AI_DRAFT_RECORDS) if you want the larger set for a quick schema/
# derive smoke-test, but NOT for reporting evaluation numbers.
AI_DRAFT_RECORDS = [P010, P019, P020, P044, P049, P050]


# ---------------------------------------------------------------------------
# Deterministic derive lives in derive.py (single source of truth).
# ---------------------------------------------------------------------------
from derive import q1_range, referral_steps, treatment_sequence  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Human hand-labels (Matsumoto), for the diff
# ---------------------------------------------------------------------------

HAND_LABELS = {
    "P001": {"on_biologic": True, "barriers": ["cost_insurance", "patient_fear"], "referral_steps": 2, "churn": False, "disease_control": "remission", "treatment_classes": ["aminosalicylate", "immunomodulator"]},
    "P002": {"on_biologic": False, "barriers": ["cost_insurance"], "referral_steps": 2, "churn": False, "disease_control": "poor", "treatment_classes": ["corticosteroid", "aminosalicylate", "immunomodulator", "corticosteroid"]},
    "P047": {"on_biologic": False, "barriers": [], "referral_steps": None, "churn": True, "disease_control": "poor", "treatment_classes": ["corticosteroid", "aminosalicylate", "immunomodulator"]},
    "P048": {"on_biologic": False, "barriers": ["surgery_first", "cost_insurance"], "referral_steps": 2, "churn": True, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "antibiotic"]},
    "P015": {"on_biologic": True, "barriers": ["cost_insurance"], "referral_steps": 2, "churn": False, "disease_control": "partial", "treatment_classes": ["other", "corticosteroid", "aminosalicylate", "immunomodulator"]},
    "P031": {"on_biologic": True, "barriers": ["comorbidity_risk"], "referral_steps": 2, "churn": False, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "corticosteroid"]},
    "P004": {"on_biologic": False, "barriers": ["cost_insurance"], "referral_steps": 2, "churn": False, "disease_control": "poor", "treatment_classes": ["corticosteroid", "aminosalicylate", "aminosalicylate", "corticosteroid"]},
    "P038": {"on_biologic": True, "barriers": ["comorbidity_risk"], "referral_steps": 1, "churn": False, "disease_control": "poor", "treatment_classes": []},
    "P024": {"on_biologic": False, "barriers": ["step_therapy"], "referral_steps": 1, "churn": False, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "corticosteroid", "aminosalicylate"]},
    "P045": {"on_biologic": False, "barriers": ["step_therapy"], "referral_steps": 2, "churn": True, "disease_control": "poor", "treatment_classes": ["corticosteroid", "aminosalicylate", "immunomodulator"]},
}

# AI-drafted, NOT human-verified (see module docstring) — excluded from
# evaluate.py's default golden set. Not to be quoted as evaluation results.
AI_DRAFT_LABELS = {
    "P010": {"on_biologic": True, "barriers": ["cost_insurance", "step_therapy"], "referral_steps": 2, "churn": False, "disease_control": "partial", "treatment_classes": ["aminosalicylate", "corticosteroid", "immunomodulator"]},
    "P019": {"on_biologic": True, "barriers": ["cost_insurance", "step_therapy"], "referral_steps": 1, "churn": False, "disease_control": "partial", "treatment_classes": []},
    "P020": {"on_biologic": True, "barriers": ["cost_insurance", "step_therapy"], "referral_steps": 2, "churn": False, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "corticosteroid", "immunomodulator"]},
    "P044": {"on_biologic": False, "barriers": ["patient_fear"], "referral_steps": 1, "churn": False, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "corticosteroid", "immunomodulator"]},
    "P049": {"on_biologic": None, "barriers": ["patient_fear"], "referral_steps": 2, "churn": True, "disease_control": None, "treatment_classes": ["aminosalicylate", "corticosteroid"]},
    "P050": {"on_biologic": False, "barriers": ["comorbidity_risk"], "referral_steps": 1, "churn": False, "disease_control": "poor", "treatment_classes": ["aminosalicylate", "corticosteroid"]},
}


def _barriers_str(rec: PatientRecord) -> str:
    if rec.biologic.barriers:
        return ",".join(b.type.value for b in rec.biologic.barriers)
    return f"<{rec.biologic.barriers_status.value}>"


def _print_table(records: list[PatientRecord]) -> None:
    header = f"{'pid':5}{'on_bio':>8}{'barriers':>28}{'steps':>7}{'churn':>7}{'control':>10}"
    print(header)
    print("-" * len(header))
    for rec in records:
        steps = referral_steps(rec)
        steps_str = str(steps.value) if steps.value is not None else f"<{steps.status.value}>"
        print(f"{rec.patient_id:5}{str(rec.on_biologic):>8}{_barriers_str(rec):>28}"
              f"{steps_str:>7}{str(rec.churn):>7}"
              f"{(rec.disease.disease_control.value.value if rec.disease.disease_control.value else '-'):>10}")


def main() -> None:
    print(f"TRUE golden set — Matsumoto's hand labels ({len(RECORDS)} patients):\n")
    _print_table(RECORDS)
    lo, hi, brk = q1_range(RECORDS)
    print(f"\nQ1 (% on biologic) as a band: {lo:.0%}–{hi:.0%}   {brk}")

    print(f"\n\nAI-DRAFTED, NOT human-verified ({len(AI_DRAFT_RECORDS)} patients — "
          "shown for transparency only, NOT used by evaluate.py's default golden set):\n")
    _print_table(AI_DRAFT_RECORDS)

    print("\nDIFF vs hand-labels  (match / disagreement worth a look):")
    for rec in RECORDS:
        h = HAND_LABELS[rec.patient_id]
        steps = referral_steps(rec)
        model = {
            "on_biologic": rec.on_biologic,
            "barriers": sorted(b.type.value for b in rec.biologic.barriers),
            "churn": rec.churn,
            "disease_control": rec.disease.disease_control.value.value if rec.disease.disease_control.value else None,
            "referral_steps": steps.value,
        }
        diffs = []
        for k in model:
            hv = sorted(h[k]) if k == "barriers" else h[k]
            if model[k] != hv:
                diffs.append(f"{k}: model={model[k]} / hand={hv}")
        mark = "OK " if not diffs else "/!\\"
        print(f"  {mark} {rec.patient_id}: " + ("all key fields agree" if not diffs else "; ".join(diffs)))

    print("\nTreatment sequences before biologic (Q3):")
    for rec in RECORDS:
        print(f"  {rec.patient_id}: {' -> '.join(treatment_sequence(rec)) or '(none reached)'}")


if __name__ == "__main__":
    main()
