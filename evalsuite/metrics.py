# evalsuite/metrics.py
from typing import List, Dict, Tuple
from .extractors import Fact
from .matchers import jaccard, approx_equal_num, extract_number

RANGES = {
    "temperature_f": (95.0, 107.0),
    "temperature_c": (35.0, 41.7),
    "hr": (30, 220),
    "rr": (5, 60),
    "spo2": (50, 100),
    "bp_sys": (70, 250),
    "bp_dia": (30, 150),
}

CRITICAL_TERMS = set(["fever","chest pain","shortness of breath","hypertension","diabetes","asthma"])

def _vital_equal(a: str, b: str) -> bool:
    if "/" in a and "/" in b:
        try:
            asys, adia = a.split("/"); bsys, bdia = b.split("/")
            return approx_equal_num(float(asys), float(bsys)) and approx_equal_num(float(adia), float(bdia))
        except Exception:
            return a == b
    ax = extract_number(a); bx = extract_number(b)
    if ax is None or bx is None:
        return a == b
    return approx_equal_num(ax, bx)

def _fact_match(a: Fact, b: Fact) -> bool:
    if a.type != b.type:
        return False
    if jaccard(a.key, b.key) < 0.8:
        return False
    if a.type in ("symptom","diagnosis","allergy"):
        if a.negated != b.negated: return False
        return True
    if a.type == "vital":
        return _vital_equal(a.value or "", b.value or "")
    if a.type == "medication":
        return jaccard(a.value or "", b.value or "") >= 0.8
    return jaccard(a.value or "", b.value or "") >= 0.8

def find_missing(transcript_facts: List[Fact], note_facts: List[Fact]) -> List[Fact]:
    missing: List[Fact] = []
    for tf in transcript_facts:
        critical = (tf.type in ("vital","allergy","medication")) or \
                   (tf.type in ("symptom","diagnosis") and tf.key in CRITICAL_TERMS and not tf.negated)
        if not critical: continue
        if not any(_fact_match(tf, nf) for nf in note_facts):
            missing.append(tf)
    return missing

def find_hallucinated(transcript_facts: List[Fact], note_facts: List[Fact]) -> List[Fact]:
    halluc: List[Fact] = []
    for nf in note_facts:
        if not any(_fact_match(nf, tf) for tf in transcript_facts):
            halluc.append(nf)
    return halluc

def find_contradictions(note_facts: List[Fact]) -> List[str]:
    issues: List[str] = []
    for f in note_facts:
        if f.type == "vital":
            if f.key == "bp" and isinstance(f.value, str) and "/" in f.value:
                try:
                    sys, dia = f.value.split("/"); sys, dia = float(sys), float(dia)
                    lo_s, hi_s = RANGES["bp_sys"]; lo_d, hi_d = RANGES["bp_dia"]
                    if not (lo_s <= sys <= hi_s and lo_d <= dia <= hi_d):
                        issues.append(f"Implausible BP: {f.value}")
                except Exception:
                    issues.append(f"Malformed BP: {f.value}")
            elif f.key in ("temperature_f","temperature_c","hr","rr","spo2"):
                rng = RANGES[f.key]; val = extract_number(f.value or "")
                if val is None or not (rng[0] <= val <= rng[1]):
                    issues.append(f"Implausible {f.key}: {f.value}")
        if f.type in ("symptom","diagnosis"):
            if any(g for g in note_facts if g.type == f.type and g.key == f.key and g.negated != f.negated):
                issues.append(f"Contradiction for {f.type} '{f.key}': both present and absent")
    return issues

def prf1(pred_facts: List[Fact], ref_facts: List[Fact]) -> Dict[str, float]:
    matched = 0; used = set()
    for p in pred_facts:
        for j, r in enumerate(ref_facts):
            if j in used: continue
            if _fact_match(p, r):
                matched += 1; used.add(j); break
    P = matched / max(1, len(pred_facts))
    R = matched / max(1, len(ref_facts))
    F1 = 0.0 if (P == 0.0 and R == 0.0) else (2 * P * R) / (P + R)
    return {"precision": P, "recall": R, "f1": F1}

# ---------------------------------------------------------------
# Text overlap metrics (BLEU, ROUGE-L) -- deterministic, no deps
# ---------------------------------------------------------------

def _tok(s: str) -> List[str]:
    # simple whitespace tokenization; could be replaced with smarter tokenizer
    return [t for t in s.strip().split() if t]

def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    from collections import Counter
    return Counter(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1)))

def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """
    Corpus BLEU (single ref) with uniform n-gram weights and brevity penalty.
    Deterministic, dependency-free. Range ~[0,1].
    """
    import math
    c = _tok(candidate)
    r = _tok(reference)
    if not c or not r:
        return 0.0

    precisions = []
    for n in range(1, max_n+1):
        c_counts = _ngram_counts(c, n)
        r_counts = _ngram_counts(r, n)
        match = 0
        total = 0
        for ng, cnt in c_counts.items():
            total += cnt
            match += min(cnt, r_counts.get(ng, 0))
        precisions.append((match / total) if total > 0 else 0.0)

    # geometric mean of precisions
    if any(p == 0.0 for p in precisions):
        geo = 0.0
    else:
        geo = math.exp(sum((1.0/max_n) * math.log(p) for p in precisions))

    # brevity penalty
    c_len = len(c); r_len = len(r)
    bp = 1.0 if c_len > r_len else math.exp(1.0 - r_len / max(1, c_len))
    return bp * geo

def rouge_l_f(candidate: str, reference: str) -> float:
    """
    ROUGE-L F-measure (LCS-based), single-ref, deterministic, no deps.
    Range ~[0,1].
    """
    c = _tok(candidate)
    r = _tok(reference)
    if not c or not r:
        return 0.0

    # LCS length via DP
    m, n = len(c), len(r)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if c[i] == r[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    if prec == 0.0 or rec == 0.0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)
