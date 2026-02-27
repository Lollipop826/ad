from __future__ import annotations

import os
import re
from typing import Iterable, List

from langchain_core.documents import Document


REFERENCE_HEADERS = [
    r"^references\b",
    r"^bibliography\b",
    r"^works\s+cited\b",
    r"^参考文献$",
    r"^参考资料$",
]


def _looks_like_reference_header(text: str) -> bool:
    lowered = text.lower().strip()
    return any(re.match(pattern, lowered) for pattern in REFERENCE_HEADERS)


def _looks_like_table_line(text: str) -> bool:
    stripped = text.strip()
    if "|" in stripped and stripped.count("|") >= 2:
        return True
    if re.match(r"^:?-{3,}:?\s*(\|\s*:?-{3,}:?)*$", stripped):
        return True
    return False


def _looks_like_figure_or_caption(text: str) -> bool:
    stripped = text.strip()
    patterns = [
        r"^(图|表)\s*\d+",
        r"^(Fig(?:ure)?\.?\s*\d+)",
        r"^(Table\s*\d+)",
        r"^(Figure\s*\d+)",
        r"^(图|表)[：:]",
        r"^(Fig|Table)[：:]",
    ]
    return any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns)


def _is_metadata_line(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "作者", "单位", "来源", "刊物", "期刊", "通讯作者", "基金", "资助",
        "doi", "issn", "journal", "publisher", "corresponding", "email",
        "地址", "地址：", "mail", "received", "accepted", "published",
    ]
    if any(keyword in lowered for keyword in keywords):
        return True
    if len(text) <= 120 and re.search(r"20\d{2}", text):
        return True
    return False


def _strip_markdown_formatting(text: str) -> str:
    result = text
    result = re.sub(r"`([^`]+)`", r"\1", result)
    result = re.sub(r"\*\*([^*]+)\*\*", r"\1", result)
    result = re.sub(r"__([^_]+)__", r"\1", result)
    result = re.sub(r"\*([^*]+)\*", r"\1", result)
    result = re.sub(r"_([^_]+)_", r"\1", result)
    result = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result)
    result = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", result)
    result = re.sub(r"<[^>]+>", "", result)
    result = re.sub(r"\s+", " ", result)
    # Remove spaces between consecutive CJK characters
    result = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", result)
    return result.strip()


def _preprocess_markdown_lines(text: str) -> List[str]:
    content = text.replace("\r\n", "\n")
    content = re.sub(r"```[\s\S]*?```", "", content)
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    lines: List[str] = []
    reference_mode = False
    for raw_line in content.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            lines.append("")
            continue
        if _looks_like_reference_header(stripped):
            reference_mode = True
            continue
        if reference_mode:
            continue
        if _looks_like_table_line(stripped):
            continue
        if _looks_like_figure_or_caption(stripped):
            continue
        lines.append(stripped)
    return lines


def _paragraphs_from_lines(lines: Iterable[str], drop_meta_prefix: bool = True) -> List[str]:
    paragraphs: List[str] = []
    buffer: List[str] = []
    line_index = 0
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            line_index += 1
            continue

        cleaned_line = line
        if cleaned_line.startswith(("#", ">")):
            cleaned_line = cleaned_line.lstrip("#> ")
        cleaned_line = re.sub(r"^[-*+•]\s+", "", cleaned_line)
        cleaned_line = re.sub(r"^\d+[\.、]\s+", "", cleaned_line)

        if drop_meta_prefix and line_index < 10 and _is_metadata_line(cleaned_line):
            line_index += 1
            continue

        buffer.append(cleaned_line)
        line_index += 1

    if buffer:
        paragraphs.append(" ".join(buffer))

    return paragraphs


def _looks_like_metadata_only(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith("keywords"):
        return True
    if lowered.startswith("key words"):
        return True
    return False


def extract_paragraphs_from_markdown_file(md_path: str, min_chars: int = 40) -> List[Document]:
    with open(md_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    lines = _preprocess_markdown_lines(raw_text)
    paragraphs = _paragraphs_from_lines(lines)

    documents: List[Document] = []
    filename = os.path.basename(md_path)
    for idx, paragraph in enumerate(paragraphs):
        cleaned = _strip_markdown_formatting(paragraph)
        if not cleaned or len(cleaned) < min_chars:
            continue
        if _looks_like_metadata_only(cleaned):
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source": os.path.abspath(md_path),
                    "filename": filename,
                    "paragraph_index": idx,
                    "chunking": "paragraph",
                    "format": "markdown",
                    "clean_rules": "markdown_academic_v1",
                },
            )
        )
    return documents


def _iter_markdown_files(md_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(md_dir):
        for name in files:
            if name.lower().endswith(".md"):
                yield os.path.join(root, name)


def load_markdown_paragraphs(md_dir: str, min_chars: int = 40) -> List[Document]:
    docs: List[Document] = []
    for path in _iter_markdown_files(md_dir):
        docs.extend(extract_paragraphs_from_markdown_file(path, min_chars=min_chars))
    return docs


