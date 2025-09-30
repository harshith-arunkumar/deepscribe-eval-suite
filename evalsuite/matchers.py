\
import re
from typing import Set

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9%\.\-/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a: str, b: str) -> float:
    A: Set[str] = set(normalize(a).split())
    B: Set[str] = set(normalize(b).split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def approx_equal_num(x: float, y: float, rel_tol: float = 0.05, abs_tol: float = 0.5) -> bool:
    return abs(x - y) <= max(abs_tol, rel_tol * max(abs(x), abs(y)))

def extract_number(s: str):
    try:
        return float(re.findall(r"-?\d+(?:\.\d+)?", s)[0])
    except Exception:
        return None
