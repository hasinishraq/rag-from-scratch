import itertools

from rag.ingest.chunkers import chunk_document, pack_section, window_split
from rag.tokens import SimpleTokenizer
from rag.types import Block, Document

TOK = SimpleTokenizer()


def _doc(n_paras: int = 12, words: int = 40) -> Document:
    blocks = tuple(
        Block(
            text=" ".join(f"w{p}_{i}" for i in range(words)),
            section_path=("Guide", f"Part {p // 4}"),
        )
        for p in range(n_paras)
    )
    return Document(doc_id="d_x", path="x.md", title="Guide", blocks=blocks)


def test_no_chunk_exceeds_limit() -> None:
    for size, overlap in ((60, 10), (100, 20), (350, 50)):
        for strategy in ("structure", "fixed"):
            chunks = chunk_document(_doc(words=130), TOK, size, overlap, True, strategy)
            assert chunks
            for c in chunks:
                assert c.n_tokens <= size
                assert TOK.count(c.embed_text) <= size
                assert TOK.count(c.text) <= size


def test_window_overlap_is_exact() -> None:
    text = " ".join(f"t{i}" for i in range(100))
    pieces = window_split(text, TOK, budget=30, overlap=10)
    for a, b in itertools.pairwise(pieces):
        assert a.split()[-10:] == b.split()[:10]
    assert pieces[0].split()[0] == "t0" and pieces[-1].split()[-1] == "t99"


def test_pack_merges_small_and_carries_overlap() -> None:
    paras = [f"p{i} " + " ".join(f"x{i}_{j}" for j in range(9)) for i in range(6)]
    chunks = pack_section(paras, TOK, budget=25, overlap=5)
    assert len(chunks) > 1
    assert all(TOK.count(c) <= 25 for c in chunks)
    assert chunks[1].split()[:5] == chunks[0].split()[-5:]


def test_header_in_embedded_text_only() -> None:
    chunks = chunk_document(_doc(), TOK, 80, 10)
    first = chunks[0]
    assert first.embed_text.startswith("Guide > Guide > Part 0\n")
    assert not first.text.startswith("Guide")
    assert first.section_path == "Guide > Part 0"


def test_structure_strategy_does_not_mix_sections() -> None:
    chunks = chunk_document(_doc(), TOK, 200, 20)
    for c in chunks:
        part = c.section_path.split(" > ")[-1]
        index = int(part.split()[-1])
        allowed = {f"w{p}_" for p in range(index * 4, index * 4 + 4)}
        assert all(any(w.startswith(a) for a in allowed) for w in c.text.split())


def test_chunk_ids_are_stable() -> None:
    a = [c.chunk_id for c in chunk_document(_doc(), TOK, 80, 10)]
    b = [c.chunk_id for c in chunk_document(_doc(), TOK, 80, 10)]
    assert a == b and a[0] == "d_x_0000"
