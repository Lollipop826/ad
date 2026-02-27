import os
import argparse
import json
from src.tools.retrieval.ingest import load_all_paragraphs


def main():
    parser = argparse.ArgumentParser(description="Dump paragraph chunks to JSONL without embeddings")
    parser.add_argument("--pdf_dir", type=str, default=os.path.abspath("kb/pdfs"))
    parser.add_argument("--out_path", type=str, default=os.path.abspath("kb/paragraph_chunks.jsonl"))
    parser.add_argument("--backend", type=str, default="pdfplumber", choices=["pymupdf4llm", "pdfplumber"])  # default to pdfplumber per request
    parser.add_argument("--min_chars", type=int, default=30)
    args = parser.parse_args()

    docs = load_all_paragraphs(pdf_dir=args.pdf_dir, min_chars=args.min_chars, backend=args.backend)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "text": d.page_content,
                "metadata": d.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} chunks to {args.out_path} using backend={args.backend}")


if __name__ == "__main__":
    main()


