import os
import json
import argparse

from src.tools.retrieval.sentence_filter import SentenceFilter


def main():
    parser = argparse.ArgumentParser(description="Apply sentence-level filter to JSONL chunks")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-base")
    args = parser.parse_args()

    flt = SentenceFilter(model_name=args.model, device=args.device)

    kept = 0
    total = 0
    with open(args.jsonl, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            rec = json.loads(line)
            text = rec.get("text", "")
            filtered = flt.keep(args.query, text, threshold=args.threshold)
            if not filtered:
                continue
            rec["text"] = filtered
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Filtered {kept}/{total} records → {args.out}")


if __name__ == "__main__":
    main()


