from __future__ import annotations

import os
import re
import glob
import shutil
import logging
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import pymupdf4llm
import pdfplumber


LOGGER = logging.getLogger("tools.retrieval.ingest")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


DEFAULT_COLLECTION = "ad_kb"
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_BASE_URL = os.getenv("OHMYGPT_BASE_URL", os.getenv("OPENAI_BASE_URL"))
DEFAULT_API_KEY = os.getenv("OHMYGPT_API_KEY", os.getenv("OPENAI_API_KEY"))


def _iter_pdf_files(pdf_dir: str) -> Iterable[str]:
    pattern = os.path.join(os.path.abspath(pdf_dir), "**", "*.pdf")
    for path in glob.iglob(pattern, recursive=True):
        if os.path.isfile(path):
            yield path


def _extract_with_pymupdf4llm(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)


def _extract_with_pdfplumber(pdf_path: str) -> str:
    # Use PDFMiner via pdfplumber to respect reading order/layout as much as possible
    # Combine pages with a blank line between them to help paragraph splitting
    parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            parts.append(text.strip())
    return "\n\n".join([p for p in parts if p])


def _looks_like_reference_header(text: str) -> bool:
    t = text.strip().lower()
    patterns = [
        r"^references\b",
        r"^bibliography\b",
        r"^works\s+cited\b",
        r"^参考文献$",
        r"^参考资料$",
    ]
    return any(re.match(p, t) for p in patterns)


def _looks_like_figure_or_table(text: str) -> bool:
    t = text.strip()
    patterns = [
        r"^(Fig(?:ure)?\.?\s*\d+)",
        r"^(Table\s*\d+)",
        r"^(图\s*\d+)",
        r"^(表\s*\d+)",
        r"^图表\s*\d+",
    ]
    if any(re.match(p, t, flags=re.IGNORECASE) for p in patterns):
        return True
    # common caption starters
    if re.match(r"^(图|表|Fig|Figure|Table)[:：]", t, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_equation(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # drop standalone equation numbers like (1), [1]
    if re.match(r"^[\(\[]?\d+[\)\]]?$", t):
        return True
    # symbol density heuristic
    math_chars = set("=+-*/^%<>≤≥≈±×÷∑∏∫√∞→←λμσπθΔΩαβγδ•·")
    symbol_count = sum(ch in math_chars for ch in t)
    if symbol_count >= 3 and symbol_count / max(len(t), 1) > 0.25:
        return True
    # LaTeX-like
    if re.search(r"\\(?:frac|sum|int|alpha|beta|gamma|delta|pi|theta)", t):
        return True
    return False


def _looks_like_meta_header(text: str) -> bool:
    t = text.strip()
    tl = t.lower()
    keywords = [
        "doi", "issn", "received", "accepted", "published",
        "corresponding author", "affiliation", "email", "@",
        "department", "university", "hospital", "institute", "address",
        "elsevier", "springer", "wiley", "sciencedirect",
        "单位", "作者", "通讯作者", "邮箱", "基金", "项目", "地址", "省", "市", "中国",
    ]
    if any(k in tl for k in keywords):
        return True
    # likely journal header/footer or copyright
    if ("©" in t) or ("http" in tl):
        return True
    # year patterns in header-like short lines
    if len(t) <= 120 and re.search(r"20\d{2}", t):
        return True
    return False


def _clean_academic_paragraphs(segments: List[str]) -> List[str]:
    cleaned: List[str] = []
    references_started = False
    for idx, seg in enumerate(segments):
        if references_started:
            # drop everything after references
            continue
        if _looks_like_reference_header(seg):
            references_started = True
            continue
        # drop figure/table captions
        if _looks_like_figure_or_table(seg):
            continue
        # drop equations
        if _looks_like_equation(seg):
            continue
        # drop early meta blocks (title/authors/affiliations/dates/locations)
        if idx <= 8 and _looks_like_meta_header(seg):
            continue
        # drop pure page numbers
        if re.match(r"^\s*\d+\s*$", seg):
            continue
        cleaned.append(seg)
    return cleaned


def extract_paragraphs_from_pdf(
    pdf_path: str,
    min_chars: int = 30,
    backend: str = "pymupdf4llm",
) -> List[Document]:
    """Extract paragraph-level chunks from a PDF.

    backends:
    - "pymupdf4llm": fast, robust markdown extraction
    - "pdfplumber": PDFMiner-based reading-order text extraction

    Splits by blank lines, filters very short fragments, returns LangChain Documents.
    """
    try:
        if backend == "pdfplumber":
            full_text: str = _extract_with_pdfplumber(pdf_path)
        else:
            full_text = _extract_with_pymupdf4llm(pdf_path)
    except Exception as exc:
        LOGGER.error(f"Failed to parse PDF: {pdf_path} | {exc}")
        return []

    segments = [seg.strip() for seg in re.split(r"\n\s*\n+", full_text) if seg and seg.strip()]
    # academic-specific cleanup
    segments = _clean_academic_paragraphs(segments)
    docs: List[Document] = []
    filename = os.path.basename(pdf_path)
    for idx, seg in enumerate(segments):
        if len(seg) < min_chars:
            continue
        docs.append(
            Document(
                page_content=seg,
                metadata={
                    "source": os.path.abspath(pdf_path),
                    "filename": filename,
                    "paragraph_index": idx,
                    "chunking": "paragraph",
                    "backend": backend,
                    "cleaned": True,
                    "clean_rules": "academic_v1",
                },
            )
        )
    return docs


def load_all_paragraphs(pdf_dir: str, min_chars: int = 30, backend: str = "pymupdf4llm") -> List[Document]:
    """Load paragraphs for all PDFs under a directory (recursively)."""
    all_docs: List[Document] = []
    for pdf_path in _iter_pdf_files(pdf_dir):
        docs = extract_paragraphs_from_pdf(pdf_path, min_chars=min_chars, backend=backend)
        if docs:
            LOGGER.info(f"Parsed {len(docs)} paragraphs from {os.path.basename(pdf_path)}")
            all_docs.extend(docs)
    LOGGER.info(f"Total paragraphs: {len(all_docs)} from dir: {pdf_dir}")
    return all_docs


def build_chroma_from_paragraphs(
    docs: List[Document],
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = DEFAULT_BASE_URL,
    api_key: Optional[str] = DEFAULT_API_KEY,
) -> Tuple[Chroma, int]:
    """Create/update a Chroma vector store from paragraph Documents."""
    if not docs:
        LOGGER.warning("No documents provided; skipping Chroma build.")
        os.makedirs(persist_dir, exist_ok=True)
        embeddings = OpenAIEmbeddings(model=embedding_model, base_url=base_url, api_key=api_key)
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        return vectordb, 0

    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=embedding_model, base_url=base_url, api_key=api_key)

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vectordb.add_documents(docs)
    vectordb.persist()
    return vectordb, len(docs)


def rebuild_chroma_from_pdfs(
    pdf_dir: str,
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = DEFAULT_BASE_URL,
    api_key: Optional[str] = DEFAULT_API_KEY,
    min_chars: int = 30,
    wipe: bool = True,
    backend: str = "pymupdf4llm",
) -> Tuple[Chroma, int]:
    """Convenience helper to rebuild a Chroma store from PDFs.

    - Optionally wipes the existing persist directory
    - Extracts paragraph-level chunks and indexes into Chroma
    """
    if wipe and os.path.exists(persist_dir):
        LOGGER.info(f"Wiping existing Chroma at {persist_dir}")
        shutil.rmtree(persist_dir)

    docs = load_all_paragraphs(pdf_dir, min_chars=min_chars, backend=backend)
    return build_chroma_from_paragraphs(
        docs=docs,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )


def get_chroma_retriever(
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = DEFAULT_BASE_URL,
    api_key: Optional[str] = DEFAULT_API_KEY,
    k: int = 5,
):
    """Return a LangChain retriever for an existing Chroma store."""
    embeddings = OpenAIEmbeddings(model=embedding_model, base_url=base_url, api_key=api_key)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


