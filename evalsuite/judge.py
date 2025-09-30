# evalsuite/judge.py
import os, json, re, sys
from typing import Optional, Dict, Any

PROMPT = """You are a clinical documentation auditor. Given a transcript, a generated SOAP note, and the clinician reference,
rate the note on:
1) Completeness (did it miss important findings from transcript?) [1-5]
2) Grounding (does every factual claim appear in transcript?) [1-5]
3) Clinical accuracy (any incorrect medical statements?) [1-5]

Respond as strict JSON ONLY (no prose, no markdown), exactly:
{"completeness": <int 1..5>, "grounding": <int 1..5>, "clinical_accuracy": <int 1..5>, "rationale": "<short reason>"}
"""

def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

# ------------ OpenAI backend (optional, unchanged) ------------
def judge_with_openai(transcript: str, note: str, reference: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("[judge] OPENAI_API_KEY not set; skipping OpenAI judge.\n")
        return None
    try:
        import openai  # type: ignore
    except Exception:
        sys.stderr.write("[judge] openai package not installed; skipping OpenAI judge.\n")
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        content = f"TRANSCRIPT:\n{transcript}\n\nNOTE:\n{note}\n\nREFERENCE:\n{reference}\n\nRubric:\n{PROMPT}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        parsed = _safe_parse_json(txt)
        if not isinstance(parsed, dict):
            sys.stderr.write("[judge] OpenAI returned non-JSON; skipping.\n")
            return None
        for k in ("completeness","grounding","clinical_accuracy"):
            if k in parsed:
                try: parsed[k] = int(parsed[k])
                except Exception: pass
        return parsed
    except Exception as e:
        sys.stderr.write(f"[judge] OpenAI exception: {e}\n")
        return None

# ------------ OpenRouter backend (new) ------------
def judge_with_openrouter(transcript: str, note: str, reference: str, model_name: Optional[str]):
    """
    Calls OpenRouter's /chat/completions endpoint.
    Docs: https://openrouter.ai/docs
    """
    try:
        import requests  # type: ignore
    except Exception as e:
        sys.stderr.write(f"[judge] requests not installed for OpenRouter backend: {e}\n")
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write("[judge] OPENROUTER_API_KEY not set; skipping OpenRouter judge.\n")
        return None

    if not model_name:
        # pick a reasonable instruct model; you can override via --llm-model
        model_name = "meta-llama/llama-3.1-8b-instruct"

    # compose prompt
    content = f"TRANSCRIPT:\n{transcript}\n\nNOTE:\n{note}\n\nREFERENCE:\n{reference}\n\nRubric:\n{PROMPT}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # optional analytics headers (won't break if missing)
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai/api/v1"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "DeepScribe Evals"),
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
    }

    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        txt = txt.strip()
        parsed = _safe_parse_json(txt)
        if not isinstance(parsed, dict):
            sys.stderr.write("[judge] OpenRouter returned non-JSON; skipping.\n")
            return None
        for k in ("completeness","grounding","clinical_accuracy"):
            if k in parsed:
                try: parsed[k] = int(parsed[k])
                except Exception: pass
        return parsed
    except Exception as e:
        sys.stderr.write(f"[judge] OpenRouter error: {e}\n")
        return None

# ------------ Dispatcher ------------
def judge_dispatch(transcript: str, note: str, reference: str, backend: str, model_name: Optional[str] = None):
    backend = (backend or "none").lower()
    if backend == "openai":
        return judge_with_openai(transcript, note, reference)
    if backend == "openrouter":
        return judge_with_openrouter(transcript, note, reference, model_name)
    # 'hf' (local) removed per your request to avoid CUDA/local setup
    return None
