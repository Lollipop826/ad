import os
import json
import argparse

from src.tools.retrieval.filters import filter_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter JSONL chunks and write to output directory")
    parser.add_argument("--in_dir", type=str, default=os.path.abspath("kb/chunks_md"))
    parser.add_argument("--out_dir", type=str, default=os.path.abspath("kb/chunks_md_filtered"))
    parser.add_argument("--min_len", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kept_total = 0
    dropped_total = 0

    for root, _, files in os.walk(args.in_dir):
        for name in files:
            if not name.lower().endswith(".jsonl"):
                continue
            in_path = os.path.join(root, name)
            out_path = os.path.join(args.out_dir, name)
            kept = 0
            dropped = 0
            with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text") or ""
                    metadata = rec.get("metadata") or {}
                    if filter_text(text, metadata, min_len=args.min_len):
                        fout.write(json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        dropped += 1
            print(f"{name}: kept={kept} dropped={dropped}")
            kept_total += kept
            dropped_total += dropped

    print(f"DONE: kept={kept_total} dropped={dropped_total} → {args.out_dir}")


if __name__ == "__main__":
    main()


