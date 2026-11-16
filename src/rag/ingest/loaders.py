"""File -> Document. Each loader yields heading-scoped blocks with a section path and page."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, cast

from bs4 import BeautifulSoup, Tag
from markdown_it import MarkdownIt

from rag.errors import IngestError
from rag.types import Block, Document

SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".html", ".htm"})
# A PDF span this much larger than the body font size is treated as a heading.
_PDF_HEADING_RATIO = 1.15
_HTML_DROP = ("script", "style", "nav", "header", "footer", "aside", "noscript", "form")
_HTML_BLOCKS = ("p", "li", "pre", "blockquote", "td", "th", "dd", "dt")


def doc_id_for(rel_path: str) -> str:
    return "d_" + hashlib.sha1(rel_path.encode()).hexdigest()[:10]


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return tuple(t for _, t in stack)


def load_markdown(text: str) -> tuple[str | None, list[Block]]:
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    stack: list[tuple[int, str]] = []
    section: tuple[str, ...] = ()
    title: str | None = None
    blocks: list[Block] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            heading = tokens[i + 1].content.strip()
            if level == 1 and title is None:
                title = heading
            section = _push_heading(stack, level, heading)
            i += 3
            continue
        if tok.type == "inline" and tok.content.strip():
            blocks.append(Block(text=tok.content.strip(), section_path=section))
        elif tok.type in ("fence", "code_block") and tok.content.strip():
            blocks.append(Block(text=tok.content.rstrip(), section_path=section))
        i += 1
    return title, blocks


def load_html(text: str) -> tuple[str | None, list[Block]]:
    soup = BeautifulSoup(text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    for tag in soup.find_all(_HTML_DROP):
        tag.decompose()
    root = soup.body or soup
    stack: list[tuple[int, str]] = []
    section: tuple[str, ...] = ()
    blocks: list[Block] = []
    headings = tuple(f"h{n}" for n in range(1, 7))
    for el in root.find_all((*headings, *_HTML_BLOCKS)):
        if not isinstance(el, Tag):
            continue
        if el.name in headings:
            heading = el.get_text(" ", strip=True)
            if heading:
                section = _push_heading(stack, int(el.name[1]), heading)
                title = title or heading
            continue
        if el.find_parent(_HTML_BLOCKS) is not None:
            continue  # nested block; its text is part of the parent
        content = el.get_text(" ", strip=True)
        if content:
            blocks.append(Block(text=content, section_path=section))
    return title, blocks


def load_pdf(path: Path) -> tuple[str | None, list[Block]]:
    import pymupdf

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise IngestError(f"{path.name}: cannot open PDF: {exc}") from exc
    with pdf:
        pages: list[dict[str, Any]] = [cast(dict[str, Any], pg.get_text("dict")) for pg in pdf]
        meta_title = (pdf.metadata or {}).get("title") or None
    # Body size = the font size that carries the most characters (headers and footers are short).
    chars: Counter[float] = Counter()
    for page in pages:
        for blk in page["blocks"]:
            for line in blk.get("lines", []):
                for span in line["spans"]:
                    chars[round(span["size"], 1)] += len(span["text"].strip())
    body_size = chars.most_common(1)[0][0] if chars else 0.0
    stack: list[tuple[int, str]] = []
    section: tuple[str, ...] = ()
    blocks: list[Block] = []
    first_heading: str | None = None
    for page_no, page in enumerate(pages, start=1):
        for blk in page["blocks"]:
            lines = blk.get("lines", [])
            text = "\n".join("".join(s["text"] for s in ln["spans"]) for ln in lines).strip()
            if not text:
                continue
            size = max(s["size"] for ln in lines for s in ln["spans"])
            if body_size and size >= body_size * _PDF_HEADING_RATIO and len(text) < 120:
                level = 1 if size >= body_size * 1.6 else 2 if size >= body_size * 1.3 else 3
                heading = " ".join(text.split())
                section = _push_heading(stack, level, heading)
                first_heading = first_heading or heading
                continue
            blocks.append(Block(text=text, section_path=section, page=page_no))
    return meta_title or first_heading, blocks


def load_file(path: Path, corpus_dir: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"{path.name}: unsupported file type {suffix}")
    rel = path.relative_to(corpus_dir).as_posix()
    data = path.read_bytes()
    if suffix == ".pdf":
        title, blocks = load_pdf(path)
    else:
        text = data.decode("utf-8", errors="replace")
        title, blocks = load_markdown(text) if suffix in (".md", ".markdown") else load_html(text)
    return Document(
        doc_id=doc_id_for(rel),
        path=rel,
        title=title or path.stem,
        blocks=tuple(blocks),
        sha256=hashlib.sha256(data).hexdigest(),
    )
