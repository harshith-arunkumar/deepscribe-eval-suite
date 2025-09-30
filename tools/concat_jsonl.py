# tools/concat_jsonl.py
import sys, os, gzip

def iter_lines(path: str):
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line

def main():
    if len(sys.argv) < 4 or ("-o" not in sys.argv and "--out" not in sys.argv):
        print("Usage: python tools/concat_jsonl.py <in1> <in2> [<in3> ...] -o <out>", file=sys.stderr)
        sys.exit(1)

    if "-o" in sys.argv:
        oidx = sys.argv.index("-o")
    else:
        oidx = sys.argv.index("--out")

    out_path = sys.argv[oidx + 1]
    in_paths = sys.argv[1:oidx]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as w:
        for p in in_paths:
            for line in iter_lines(p):
                w.write(line + "\n")

if __name__ == "__main__":
    main()
