\
import re
from dataclasses import dataclass
from typing import List, Optional
from .matchers import extract_number

DIAG_SYMPTOMS = [
    "fever", "cough", "sore throat", "shortness of breath", "chest pain",
    "headache", "nausea", "vomiting", "diarrhea", "fatigue", "dizziness",
    "hypertension", "diabetes", "asthma", "covid", "influenza", "otitis",
]
NEGATION_CUES = ["no", "denies", "without", "not", "negative for", "nkda", "no known drug allergies"]

VITAL_PATTERNS = {
    "temperature_f": r"(?:temp(?:erature)?[:\s]*)(\d{2,3}(?:\.\d)?)\s*(?:f|°f|degrees f)?",
    "temperature_c": r"(?:temp(?:erature)?[:\s]*)(\d{2}(?:\.\d)?)\s*(?:c|°c|degrees c)",
    "hr": r"(?:heart rate|hr)[:\s]*(\d{2,3})\b",
    "rr": r"(?:respiratory rate|rr)[:\s]*(\d{1,2})\b",
    "spo2": r"(?:sp[o0]2|oxygen saturation)[:\s]*(\d{2,3})\s*%",
    "bp": r"(?:bp|blood pressure)[:\s]*(\d{2,3})\s*/\s*(\d{2,3})\b",
}

@dataclass(frozen=True)
class Fact:
    type: str
    key: str
    value: Optional[str]
    negated: bool
    raw: str

def _negated(text: str, start: int) -> bool:
    window = text[max(0, start-30):start].lower()
    return any(cue in window for cue in NEGATION_CUES)

def extract_vitals(text: str) -> List[Fact]:
    out: List[Fact] = []
    lowered = text.lower()
    for k, pat in VITAL_PATTERNS.items():
        for m in re.finditer(pat, lowered):
            if k == "bp":
                sys, dia = m.group(1), m.group(2)
                out.append(Fact("vital", "bp", f"{sys}/{dia}", False, m.group(0)))
            elif k == "temperature_f":
                t = m.group(1); out.append(Fact("vital", "temperature_f", f"{t} F", False, m.group(0)))
            elif k == "temperature_c":
                t = m.group(1); out.append(Fact("vital", "temperature_c", f"{t} C", False, m.group(0)))
            else:
                val = m.group(1); out.append(Fact("vital", k, val, False, m.group(0)))
    return out

def extract_allergies(text: str) -> List[Fact]:
    out: List[Fact] = []
    if re.search(r"\b(no known drug allergies|nkda)\b", text, flags=re.I):
        out.append(Fact("allergy", "drug", "none", False, "NKDA"))
        return out
    for m in re.finditer(r"allergic to\s+([a-zA-Z0-9\-\s]+?)([\.,;\n]|$)", text, flags=re.I):
        drug = m.group(1).strip().lower()
        out.append(Fact("allergy", drug, "present", False, m.group(0)))
    return out

def extract_meds(text: str) -> List[Fact]:
    out: List[Fact] = []
    for m in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9\-]{1,30})\s+(\d{1,4})\s*(mg|mcg)\b([^\n\.;]*)", text, flags=re.I):
        name = m.group(1).lower()
        dose = m.group(2) + " " + m.group(3).lower()
        tail = " ".join(m.group(4).split()[:4])
        out.append(Fact("medication", name, (dose + " " + tail).strip(), False, m.group(0)))
    return out

def extract_diags_symptoms(text: str) -> List[Fact]:
    out: List[Fact] = []
    lowered = text.lower()
    for term in DIAG_SYMPTOMS:
        for m in re.finditer(rf"\b{re.escape(term)}\b", lowered):
            neg = _negated(lowered, m.start())
            out.append(Fact("diagnosis" if term in ('hypertension','diabetes','asthma','covid','influenza','otitis')
                            else "symptom", term, "present" if not neg else "absent", neg, m.group(0)))
    return out

def extract_all(text: str) -> List[Fact]:
    facts: List[Fact] = []
    facts.extend(extract_vitals(text))
    facts.extend(extract_allergies(text))
    facts.extend(extract_meds(text))
    facts.extend(extract_diags_symptoms(text))
    return facts
