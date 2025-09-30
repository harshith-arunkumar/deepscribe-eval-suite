\
import re
from typing import Dict

def split_soap_sections(text: str) -> Dict[str, str]:
    t = text or ""
    lines = t.splitlines()
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    current = None

    def header_of(line: str):
        l = line.strip().lower()
        l = re.sub(r"[:\-\s]+$", "", l)
        if l in ("subjective", "s"):
            return "subjective"
        if l in ("objective", "o"):
            return "objective"
        if l in ("assessment", "a"):
            return "assessment"
        if l in ("plan", "p"):
            return "plan"
        return None

    for line in lines:
        h = header_of(line)
        if h is not None:
            current = h
            continue
        if current is None:
            current = "subjective"
        sections[current] += (line + "\n")

    if not any(v.strip() for v in sections.values()):
        return {"body": t}
    return sections
