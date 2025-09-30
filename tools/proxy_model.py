\
import argparse, json, pathlib, random, re

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p, rows):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def jitter_numbers(text, jitter=0.03):
    def repl(m):
        x = float(m.group(0)); y = x*(1.0+random.uniform(-jitter,jitter))
        return f"{y:.1f}" if abs(x)>=50 else f"{round(y)}"
    return re.sub(r"\b\d+(?:\.\d+)?\b", repl, text)

def maybe_drop_sections(text, p=0.15):
    # naive drop of some sections by header tokens
    out = []
    for sec in ["S:", "O:", "A:", "P:"]:
        m = re.search(rf"(?ms)^ *{sec}.*?(?=^\s*[SOAP]:|\Z)", text)
        if not m: 
            continue
        block = m.group(0)
        if random.random() > p:
            out.append(block.strip())
    return "\n".join(out) if out else text

def flip_negations(text, p=0.2):
    pairs = [(" no ", " yes "), (" denies ", " reports "), (" without ", " with ")]
    for a,b in pairs:
        if random.random() < p:
            text = re.sub(a, b, text, flags=re.I)
    return text

def add_hallucination(text, p=0.2):
    if random.random() < p:
        text += "\nO: Temperature 120 F, SpO2 85%."
    return text

def transform(ref, mode):
    t = ref
    if mode == "mild":
        t = jitter_numbers(t, 0.02); t = maybe_drop_sections(t, 0.10)
    elif mode == "medium":
        t = jitter_numbers(t, 0.05); t = maybe_drop_sections(t, 0.25); t = flip_negations(t, 0.25)
    else:
        t = jitter_numbers(t, 0.08); t = maybe_drop_sections(t, 0.35); t = flip_negations(t, 0.4); t = add_hallucination(t, 0.5)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="medium", choices=["mild","medium","spicy"])
    args = ap.parse_args()

    rows = []
    for ex in load_jsonl(args.input):
        ref = ex.get("reference_note","")
        ex["generated_note"] = transform(ref, args.mode)
        rows.append(ex)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} -> {args.out}")

if __name__ == "__main__":
    main()
