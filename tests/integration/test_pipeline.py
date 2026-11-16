"""Ingest the fixture corpus and answer with the fake provider."""

from pathlib import Path
from typing import Any

import pytest

from rag.errors import InvalidRequest
from rag.types import Filters, Mode
from tests.conftest import GROQ


def test_ingest_is_incremental(make_env: Any) -> None:
    env = make_env()
    first = env.pipeline.ingest()
    assert (first.added, first.updated, first.deleted) == (3, 0, 0)
    assert first.chunks > 0
    again = env.pipeline.ingest()
    assert (again.added, again.updated, again.deleted, again.chunks) == (0, 0, 0, 0)


def test_editing_one_file_updates_only_its_chunks(make_env: Any) -> None:
    env = make_env()
    env.pipeline.ingest()
    s = env.services
    before = {d.path: s.chunks.chunk_ids_for_doc(d.doc_id) for d in s.chunks.documents()}
    faq_text_before = {cid: s.chunks.get(cid) for cid in before["policies/payments_faq.html"]}
    md = env.cfg.corpus_dir / "policies/refund_policy.md"
    md.write_text(
        md.read_text(encoding="utf-8") + "\n## Exchanges\n\nExchanges are free.\n", encoding="utf-8"
    )
    report = env.pipeline.ingest()
    assert (report.added, report.updated, report.deleted) == (0, 1, 0)
    after = {cid: s.chunks.get(cid) for cid in before["policies/payments_faq.html"]}
    assert after == faq_text_before
    refund_doc = s.chunks.document_by_path("policies/refund_policy.md")
    assert refund_doc is not None
    stored = s.chunks.get_many(s.chunks.chunk_ids_for_doc(refund_doc.doc_id))
    assert any("Exchanges are free" in c.text for c in stored.values())


def test_deleted_file_is_removed_from_all_three_stores(make_env: Any) -> None:
    env = make_env()
    env.pipeline.ingest()
    s = env.services
    doc = s.chunks.document_by_path("policies/payments_faq.html")
    assert doc is not None
    ids = s.chunks.chunk_ids_for_doc(doc.doc_id)
    (env.cfg.corpus_dir / "policies/payments_faq.html").unlink()
    report = env.pipeline.ingest()
    assert report.deleted == 1
    assert s.chunks.document_by_path("policies/payments_faq.html") is None
    assert not set(ids) & set(s.indexes.vectors.ids)
    assert not set(ids) & set(s.indexes.bm25.docs)
    # The saved files agree with memory after a restart.
    from rag.index.vector_store import VectorStore

    assert not set(ids) & set(VectorStore.load(Path(env.cfg.store_dir)).ids)


def test_cited_answer_end_to_end(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    ans = env.pipeline.answer("Within how many days of approval are approved refunds issued?")
    assert not ans.refused
    assert "14 days" in ans.answer
    assert any(src.cited for src in ans.sources)
    cited = next(src for src in ans.sources if src.cited)
    assert cited.path == "policies/refund_policy.md"
    assert ans.provider == "groq" and ans.n_candidates > 0
    trace = env.services.chunks.get_trace(ans.trace_id)
    assert trace is not None
    assert '<source id="1"' in trace.prompt and trace.answer == ans.answer
    assert trace.provider == "groq" and trace.tokens_in > 0 and trace.config_hash


def test_trace_keeps_reranker_scores_for_every_scored_candidate(make_env: Any) -> None:
    env = make_env()
    env.pipeline.ingest()
    ans = env.pipeline.answer("card refunds bank statement", retrieve_only=True)
    trace = env.services.chunks.get_trace(ans.trace_id)
    assert trace is not None
    pool = trace.candidates[: env.cfg.rerank.k_rerank]
    assert len(pool) > len(ans.sources)  # more candidates than were sent on
    assert all(c.rerank_score is not None for c in pool)


def test_low_score_refuses_without_calling_generator(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    ans = env.pipeline.answer("What is the boiling point of mercury on Venus?")
    assert ans.refused
    assert ans.answer == env.cfg.generation.refusal_text
    assert env.server.chat_requests() == []
    assert env.services.chunks.get_trace(ans.trace_id) is not None


def test_model_refusal_returns_refusal_text(make_env: Any) -> None:
    env = make_env(rerank={"min_score": 0.0})
    env.mark_tested("groq")
    env.pipeline.ingest()
    ans = env.pipeline.answer(
        "Which refunds policy covers lunar mining permits and asteroid taxes?"
    )
    assert len(env.server.chat_requests(GROQ)) == 1
    assert ans.refused and ans.answer == env.cfg.generation.refusal_text


def test_retrieve_only_and_filters(make_env: Any) -> None:
    env = make_env()
    env.pipeline.ingest()
    ans = env.pipeline.answer("card refunds bank statement", retrieve_only=True)
    assert ans.sources and ans.answer == "" and env.server.requests == []
    only = env.pipeline.answer(
        "card refunds", retrieve_only=True, filters=Filters(path_prefixes=("handbook/",))
    )
    assert all(s.path.startswith("handbook/") for s in only.sources)


def test_cache_serves_repeated_question(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    q = "Within how many days of approval are approved refunds issued?"
    first = env.pipeline.answer(q)
    second = env.pipeline.answer(q)
    assert not first.cached and second.cached
    assert len(env.server.chat_requests()) == 1


def test_input_limits(make_env: Any) -> None:
    env = make_env()
    with pytest.raises(InvalidRequest):
        env.pipeline.answer("   ")
    with pytest.raises(InvalidRequest):
        env.pipeline.answer("x" * 5000)
    with pytest.raises(InvalidRequest):
        env.pipeline.ingest(path="../../etc")


def test_mode_override_per_query(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq", "ollama")
    env.pipeline.ingest()
    ans = env.pipeline.answer(
        "Within how many days of approval are approved refunds issued?", mode=Mode.OLLAMA_ONLY
    )
    assert ans.provider == "ollama"
