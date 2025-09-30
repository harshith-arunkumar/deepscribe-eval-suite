# main.py
import argparse, os, json
from typing import Dict, Any, List
from evalsuite.extractors import extract_all, Fact
from evalsuite.metrics import find_missing, find_hallucinated, find_contradictions, prf1, bleu, rouge_l_f
from evalsuite.report import write_per_case_jsonl, write_summary, write_dashboard
from evalsuite.judge import judge_dispatch

def to_fact(f: Fact) -> Dict[str, Any]:
    return {"type": f.type, "key": f.key, "value": f.value, "negated": f.negated, "raw": f.raw}

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def run(input_path: str, out_dir: str, llm_backend: str = "none", llm_model: str = "", num_rows = None):
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    idx = 0
    n = None if num_rows is None else int(num_rows)
    for ex in load_jsonl(input_path):
        if n is not None and idx >= n:
            break
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1} examples ...")
        idx += 1
        cid = ex.get("id")
        transcript = ex.get("transcript","")
        note = ex.get("generated_note","")
        reference = ex.get("reference_note","")

        tf = extract_all(transcript)
        nf = extract_all(note)
        rf = extract_all(reference)

        missing = find_missing(tf, nf)
        halluc = find_hallucinated(tf, nf)
        contra = find_contradictions(nf)
        align = prf1(nf, rf)
        
        bleu4 = bleu(note, reference, max_n=4)
        rougeL = rouge_l_f(note, reference)

        judged = None
        if llm_backend.lower() != "none":
            judged = judge_dispatch(transcript, note, reference, backend=llm_backend, model_name=llm_model)

        rows.append({
            "id": cid,
            "missing_count": len(missing),
            "hallucinated_count": len(halluc),
            "contradictions_count": len(contra),
            "missing": [to_fact(x) for x in missing],
            "hallucinated": [to_fact(x) for x in halluc],
            "contradictions": contra,
            "ref_align": align,
            "text_overlap": {"bleu": bleu4, "rouge_l_f": rougeL},
            "llm_judge": judged,
        })

    write_per_case_jsonl(out_dir, rows)
    summary = write_summary(out_dir, rows)
    write_dashboard(out_dir, rows, summary)
    print(f"Wrote reports -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-judge", default="none", choices=["none","openai","openrouter"],
                    help="Select LLM-judge backend")
    ap.add_argument("--llm-model", default="x-ai/grok-4-fast:free", help="Model name for 'openrouter' backend")
    ap.add_argument("--num-rows", default=None, help="Give value to limit number of rows processed")
    args = ap.parse_args()
    run(args.input, args.out, llm_backend=args.llm_judge, llm_model=args.llm_model, num_rows=args.num_rows)
