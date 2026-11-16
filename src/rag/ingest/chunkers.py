"""Document -> Chunks. Sizes are in tokens; the limit applies to the embedded text (header + body),
so neither the embedded nor the displayed text can exceed chunk_tokens."""

from __future__ import annotations

from itertools import groupby

from rag.tokens import Tokenizer
from rag.types import Block, Chunk, Document

SECTION_SEP = " > "
# A header longer than this share of the chunk budget is truncated so a body always fits.
_MAX_HEADER_SHARE = 0.4
_JOIN_SLACK = 1


def context_header(title: str, section_path: str) -> str:
    return f"{title}{SECTION_SEP}{section_path}" if section_path else title


def window_split(text: str, tok: Tokenizer, budget: int, overlap: int) -> list[str]:
    """Cut text into windows of at most `budget` tokens that overlap by `overlap` tokens."""
    ids = tok.encode(text)
    if len(ids) <= budget:
        return [text]
    step = max(1, budget - overlap)
    pieces: list[str] = []
    start = 0
    while start < len(ids):
        size = budget
        piece = tok.decode(ids[start : start + size])
        while size > 1 and tok.count(piece) > budget:  # decode/encode is not always a round trip
            size -= 1
            piece = tok.decode(ids[start : start + size])
        if piece.strip():
            pieces.append(piece.strip())
        if start + size >= len(ids):
            break
        start += min(step, size)
    return pieces


def _tail(text: str, tok: Tokenizer, overlap: int) -> str:
    if overlap == 0:
        return ""
    ids = tok.encode(text)
    return tok.decode(ids[-overlap:]).strip() if len(ids) > overlap else text


def pack_section(paragraphs: list[str], tok: Tokenizer, budget: int, overlap: int) -> list[str]:
    """Merge small paragraphs and cut large ones; consecutive chunks share `overlap` tokens."""
    pieces = [p for para in paragraphs for p in window_split(para, tok, budget, overlap)]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if tok.count(candidate) <= budget:
            current = candidate
            continue
        if current:
            chunks.append(current)
            carry = _tail(current, tok, overlap)
            joined = f"{carry}\n\n{piece}" if carry else piece
            current = joined if tok.count(joined) <= budget else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _section_groups(doc: Document, strategy: str) -> list[tuple[tuple[str, ...], list[Block]]]:
    if strategy == "fixed":
        return [((), list(doc.blocks))]
    return [(k, list(g)) for k, g in groupby(doc.blocks, key=lambda b: b.section_path)]


def chunk_document(
    doc: Document,
    tok: Tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
    add_context_header: bool = True,
    strategy: str = "structure",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, blocks in _section_groups(doc, strategy):
        section_path = SECTION_SEP.join(section)
        header = context_header(doc.title, section_path) if add_context_header else ""
        max_header = int(chunk_tokens * _MAX_HEADER_SHARE)
        if header and tok.count(header) > max_header:
            header = tok.decode(tok.encode(header)[:max_header]).strip()
        # One token of slack: BPE can merge across the header/body boundary.
        header_cost = tok.count(header + "\n") + _JOIN_SLACK if header else 0
        budget = chunk_tokens - header_cost
        overlap = min(overlap_tokens, budget // 2)
        if strategy == "fixed":
            bodies = window_split(" ".join(b.text for b in blocks), tok, budget, overlap)
        else:
            bodies = pack_section([b.text for b in blocks], tok, budget, overlap)
        first_page = next((b.page for b in blocks if b.page is not None), None)
        for body in bodies:
            ord_ = len(chunks)
            page = _page_of(body, blocks) or first_page
            embed_text = f"{header}\n{body}" if header else body
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{ord_:04d}",
                    doc_id=doc.doc_id,
                    ord=ord_,
                    text=body,
                    embed_text=embed_text,
                    section_path=section_path,
                    page=page,
                    n_tokens=tok.count(embed_text),
                )
            )
    return chunks


def _page_of(body: str, blocks: list[Block]) -> int | None:
    probe = body[:60]
    for block in blocks:
        if probe and probe in block.text:
            return block.page
    return None
