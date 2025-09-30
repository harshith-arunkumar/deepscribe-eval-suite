# tools/prepare_datasets.py
"""
Normalize multiple public datasets into the suite JSONL schema:
{"id": "...", "transcript": "...", "generated_note": "", "reference_note": "..."}

Decision on the adesouza row you provided:
- We **should not** fold extra columns (e.g., demographics, full_patient_data) into the
  transcript text because they weren’t literally said in the encounter and would
  contaminate grounding checks.
- We **will** keep those as **metadata** in a separate `meta` field (ignored by the
  evaluator), so you still have them for slicing/analysis without affecting the eval.
"""
import argparse
import json
import pathlib
import glob
import os
from typing import Iterable, Dict, Any, List, Optional


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_record(idx: int, transcript: str, reference: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": f"ex_{idx:06d}",
        "transcript": (transcript or "").strip(),
        "generated_note": "",
        "reference_note": (reference or "").strip(),
    }
    if meta:
        rec["meta"] = meta
    return rec


def load_adesouza(split: str) -> List[Dict[str, Any]]:
    """Loader for `adesouza1/soap_notes`"""
    from datasets import load_dataset
    ds = load_dataset("adesouza1/soap_notes", split=split)

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        transcript = ex.get("patient_convo") or ""
        reference = ex.get("soap_notes") or ""
        if not reference.strip():
            continue

        # Keep other info as metadata only
        meta: Dict[str, Any] = {
            "age": ex.get("age"),
            "patient_name": ex.get("patient_name"),
            "gender": ex.get("gender"),
            "dob": ex.get("dob"),
            "phone": ex.get("phone"),
            "health_problem": ex.get("health_problem"),
            "doctor_name": ex.get("doctor_name"),
            "address": ex.get("address"),
            "has_full_patient_data": bool(ex.get("full_patient_data")),
        }
        rows.append(norm_record(i, transcript, reference, meta))
    return rows


def load_omi(split: str) -> List[Dict[str, Any]]:
    """
    Load the Omi-Health dataset and normalize into our common schema:
      { id, transcript, generated_note, reference_note, meta }

    - transcript       <= 'dialogue'
    - reference_note   <= 'soap'
    - meta             <= { 'prompt', 'messages', 'messages_nosystem' } (kept as-is)
    """
    from datasets import load_dataset
    ds = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        # Required fields in this dataset
        dialogue = (ex.get("dialogue") or "").strip()
        soap = (ex.get("soap") or "").strip()
        if not dialogue or not soap:
            # Skip incomplete rows
            continue

        # Keep auxiliary fields as metadata
        meta: Dict[str, Any] = {
            "prompt": ex.get("prompt"),
            "messages": ex.get("messages"),
            "messages_nosystem": ex.get("messages_nosystem"),
        }

        rows.append(norm_record(i, dialogue, soap, meta))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["adesouza", "omi", "mts"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--mts-path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "adesouza":
        rows = load_adesouza(args.split)
    elif args.source == "omi":
        rows = load_omi(args.split)
    else:
        ap.error("{Provide path to dataset}")

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
