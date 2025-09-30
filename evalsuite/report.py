# evalsuite/report.py (replace write_summary & write_dashboard)
import os, json, csv
from typing import List, Dict, Any

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0

def write_per_case_jsonl(out_dir: str, rows: List[Dict[str, Any]]) -> None:
    path = os.path.join(out_dir, "per_case.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_summary(out_dir: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "num_cases": len(rows),
        "avg_missing": _mean([r["missing_count"] for r in rows]),
        "avg_hallucinated": _mean([r["hallucinated_count"] for r in rows]),
        "avg_contradictions": _mean([r["contradictions_count"] for r in rows]),
        "avg_ref_precision": _mean([r["ref_align"]["precision"] for r in rows]),
        "avg_ref_recall": _mean([r["ref_align"]["recall"] for r in rows]),
        "avg_ref_f1": _mean([r["ref_align"]["f1"] for r in rows]),
        # NEW: text overlap
        "avg_bleu": _mean([(r.get("text_overlap") or {}).get("bleu") for r in rows]),
        "avg_rouge_l_f": _mean([(r.get("text_overlap") or {}).get("rouge_l_f") for r in rows]),
    }

    # Optional LLM aggregates if present
    comp, ground, clin = [], [], []
    for r in rows:
        j = r.get("llm_judge")
        if isinstance(j, dict):
            comp.append(j.get("completeness"))
            ground.append(j.get("grounding"))
            clin.append(j.get("clinical_accuracy"))
    summary.update({
        "avg_llm_completeness": _mean(comp) if comp else None,
        "avg_llm_grounding": _mean(ground) if ground else None,
        "avg_llm_clinical_accuracy": _mean(clin) if clin else None,
    })

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV per case (include new text metrics and LLM if present)
    any_llm = any(r.get("llm_judge") for r in rows)
    cols = [
        "id","missing_count","hallucinated_count","contradictions_count",
        "ref_precision","ref_recall","ref_f1",
        "bleu","rouge_l_f",
    ] + (["llm_completeness","llm_grounding","llm_clinical_accuracy"] if any_llm else [])

    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            t = r.get("text_overlap") or {}
            row = [
                r["id"],
                r["missing_count"],
                r["hallucinated_count"],
                r["contradictions_count"],
                r["ref_align"]["precision"],
                r["ref_align"]["recall"],
                r["ref_align"]["f1"],
                t.get("bleu"),
                t.get("rouge_l_f"),
            ]
            if any_llm:
                j = r.get("llm_judge") or {}
                row += [j.get("completeness"), j.get("grounding"), j.get("clinical_accuracy")]
            w.writerow(row)
    return summary

def write_dashboard(out_dir: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    any_llm = any(r.get("llm_judge") for r in rows)
    trs = []
    for r in rows:
        t = r.get("text_overlap") or {}
        j = r.get("llm_judge") or {}
        llm_cells = ""
        if any_llm:
            llm_cells = f"<td>{(j.get('completeness') or '')}</td><td>{(j.get('grounding') or '')}</td><td>{(j.get('clinical_accuracy') or '')}</td>"
        trs.append(
            "<tr>"
            f"<td>{r['id']}</td>"
            f"<td>{r['missing_count']}</td><td>{r['hallucinated_count']}</td><td>{r['contradictions_count']}</td>"
            f"<td>{r['ref_align']['precision']:.2f}</td><td>{r['ref_align']['recall']:.2f}</td><td>{r['ref_align']['f1']:.2f}</td>"
            f"<td>{(t.get('bleu') or 0):.3f}</td><td>{(t.get('rouge_l_f') or 0):.3f}</td>"
            f"{llm_cells}</tr>"
        )

    llm_kv = ""
    if summary.get("avg_llm_completeness") is not None:
        llm_kv = (
            f"<div class='item'><div class='k'>Avg LLM Completeness</div><div class='v'>{summary['avg_llm_completeness']:.2f}</div></div>"
            f"<div class='item'><div class='k'>Avg LLM Grounding</div><div class='v'>{summary['avg_llm_grounding']:.2f}</div></div>"
            f"<div class='item'><div class='k'>Avg LLM Clinical</div><div class='v'>{summary['avg_llm_clinical_accuracy']:.2f}</div></div>"
        )

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Evals Dashboard</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; }}
.card {{ border:1px solid #e5e8ef; border-radius: 12px; padding: 16px; margin: 16px 0; }}
table {{ width:100%; border-collapse: collapse; }}
th, td {{ border-bottom:1px solid #eef2f7; padding:8px; font-size:14px; text-align:left; }}
th {{ background:#f8fafc; }}
.kv {{ display:flex; gap:24px; flex-wrap:wrap; }}
.kv .item {{ min-width:180px; }}
.kv .k {{ color:#556; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.kv .v {{ font-size:20px; font-weight:600; }}
</style></head><body>
<h1>DeepScribe Evals Dashboard</h1>
<div class="card kv">
  <div class="item"><div class="k">Cases</div><div class="v">{summary['num_cases']}</div></div>
  <div class="item"><div class="k">Avg Missing</div><div class="v">{summary['avg_missing']:.2f}</div></div>
  <div class="item"><div class="k">Avg Hallucinated</div><div class="v">{summary['avg_hallucinated']:.2f}</div></div>
  <div class="item"><div class="k">Avg Contradictions</div><div class="v">{summary['avg_contradictions']:.2f}</div></div>
  <div class="item"><div class="k">Avg Ref F1</div><div class="v">{summary['avg_ref_f1']:.2f}</div></div>
  <div class="item"><div class="k">Avg BLEU</div><div class="v">{summary['avg_bleu']:.3f}</div></div>
  <div class="item"><div class="k">Avg ROUGE-L(F)</div><div class="v">{summary['avg_rouge_l_f']:.3f}</div></div>
  {llm_kv}
</div>
<div class="card">
  <h3>Per-Case Metrics</h3>
  <table>
    <thead><tr>
      <th>Case</th><th>Missing</th><th>Hallucinated</th><th>Contradictions</th>
      <th>Ref P</th><th>Ref R</th><th>Ref F1</th>
      <th>BLEU</th><th>ROUGE-L(F)</th>{'<th>LLM Comp</th><th>LLM Ground</th><th>LLM Clin</th>' if any_llm else ''}
    </tr></thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</div>
<p style="color:#789">BLEU/ROUGE-L complement fact-level metrics; they are not substitutes for clinical correctness.</p>
</body></html>"""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
