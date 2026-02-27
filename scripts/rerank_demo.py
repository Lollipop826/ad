import os
import argparse
import json

from src.tools.retrieval.reranker import CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder rerank demo over a JSONL file")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to a JSONL file with {text, metadata}")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-base")
    args = parser.parse_args()

    items = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append((rec.get("text", ""), rec.get("metadata", {})))

    rr = CrossEncoderReranker(model_name=args.model, device=args.device)
    ranked = rr.rerank(args.query, items, top_k=args.top_k)
    for t, m, s in ranked:
        print(f"score={s:.4f} | file={m.get('filename')} | text={t[:120]}...")


if __name__ == "__main__":
    main()


