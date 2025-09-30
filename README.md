# DeepScribe Evaluation Suite

This repository contains my implementation of an evaluation framework for AI‑generated SOAP notes, built as part of the DeepScribe coding assessment.

## Overview

The goal of this project was to build a minimal but meaningful evaluation suite that can:
- Detect **missing findings**, **hallucinations**, and **clinical inaccuracies** in generated notes.
- Provide both **deterministic metrics** (precision, recall, F1, BLEU, ROUGE) and **LLM‑as‑a‑judge metrics** (completeness, grounding, clinical accuracy).
- Run quickly so new model changes can be tested within minutes.

The design emphasizes **reproducibility** and **ease of extension**, so it can slot directly into a production CI/CD pipeline.

## Datasets Used

I integrated two public datasets of transcripts paired with SOAP notes:
- [adesouza1/soap_notes](https://huggingface.co/datasets/adesouza1/soap_notes)
- [Omi-Health SOAP Dataset](https://huggingface.co/datasets/omi-health/medical-dialogue-to-soap-summary)

Each dataset has a slightly different schema, so I wrote normalization functions (`prepare_datasets.py`) to bring them into a consistent format.

## Metrics Implemented

1. **Deterministic Metrics**
   - Precision, Recall, F1 → checks overlap of entities and facts between generated note and reference note.
   - BLEU and ROUGE → capture fluency and overlap at the n‑gram level.
   - Contradiction / Negation check → flags mismatches like “no swelling” vs. “swelling present”.

2. **LLM-as-a-Judge (via OpenRouter)**
   - Completeness → whether critical information from transcript is preserved.
   - Grounding → whether facts are actually supported by the transcript.
   - Clinical Accuracy → whether the note is medically plausible and internally consistent.

I replaced the earlier OpenAI/HuggingFace calls with **OpenRouter** so the suite can query models through a single unified API (`OPENROUTER_API_KEY`). This makes evaluation cheaper and avoids managing large local models.

## Workflow

1. **Normalize data** from each dataset into JSONL format.
2. **Proxy model** (`proxy_model.py`) generates synthetic “mild/medium/spicy” notes with missing or hallucinated info to test evaluator sensitivity.
3. **Concatenate** datasets into one combined JSONL.
4. **Run main evaluator** (`main.py`) with chosen backends (deterministic only, or with LLM scoring through OpenRouter).
5. **Inspect results** in JSON reports and optional dashboards.

## Why This Design

- **Move Fast:** Deterministic metrics are lightweight, and proxy corruption lets me test eval quality without waiting on new models. The suite can be run end‑to‑end with one or two commands.
- **Understand Production Quality:** By combining deterministic checks with semantic judgment from an LLM (via OpenRouter), the system surfaces both quantitative overlap and clinical trustworthiness. This triangulation gives better signals about regression risk in production.

## Example Commands

### Deterministic Baseline (no LLM judge)

For a single dataset (Adesouza):

```bash
python tools/proxy_model.py --input data/adesouza.jsonl --out data/adesouza.gen.jsonl --mode mild
python main.py --input data/adesouza.gen.jsonl --out out_adesouza
```

For all datasets combined:

```bash
python tools/proxy_model.py --input data/adesouza.jsonl --out data/adesouza.gen.jsonl --mode mild
python tools/proxy_model.py --input data/omi.jsonl --out data/omi.gen.jsonl --mode medium
python tools/concat_jsonl.py data/adesouza.gen.jsonl data/omi.gen.jsonl --out data/all.gen.jsonl
python main.py --input data/all.gen.jsonl --out out_all
```

### With LLM Judge (via OpenRouter)

For a single dataset (Adesouza):

```bash
export OPENROUTER_API_KEY=sk-...

python tools/proxy_model.py --input data/adesouza.jsonl --out data/adesouza.gen.jsonl --mode mild
python main.py --input data/adesouza.gen.jsonl --out out_adesouza --llm-judge openrouter --llm-model x-ai/grok-4-fast:free
```

For all datasets combined:

```bash
export OPENROUTER_API_KEY=sk-...

python tools/proxy_model.py --input data/adesouza.jsonl --out data/adesouza.gen.jsonl --mode mild
python tools/proxy_model.py --input data/omi.jsonl --out data/omi.gen.jsonl --mode medium
python tools/concat_jsonl.py data/adesouza.gen.jsonl data/omi.gen.jsonl --out data/all.gen.jsonl
python main.py --input data/all.gen.jsonl --out out_all --llm-judge openrouter --llm-model x-ai/grok-4-fast:free --num-rows 10
```

## DISCLAIMER
The OpenRouter version is slower due to API rate limits. For testing, you can use `--num-rows` to limit input size.

## Measuring the Evaluator

I validated the evaluator by:
- Checking it detects issues in proxy‑corrupted notes (mild/medium/spicy).
- Comparing scores to human‑edited reference notes.
- Ensuring that LLM‑judge scores correlate with deterministic metrics but also highlight semantic/clinical errors those miss.

## Conclusion

This framework is not just about scoring text overlap — it is designed to ensure generated clinical notes remain faithful to transcripts, avoid hallucinations, and maintain clinical safety. It is fast, reproducible, and adaptable for production monitoring.
