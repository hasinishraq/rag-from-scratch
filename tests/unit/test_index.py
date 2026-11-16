import math
from pathlib import Path
from typing import Any

import numpy as np

from rag.index.bm25 import BM25Index, tokenize
from rag.index.embedder import LocalEmbedder, normalise
from rag.index.vector_store import VectorStore
from tests.fixtures.fakes import FakeEmbedder


def test_vectors_have_unit_norm() -> None:
    vecs = FakeEmbedder().embed_passages(["refund timelines", "card payments", "x"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(normalise([[3.0, 4.0]]), axis=1), 1.0)
    assert not np.isnan(normalise([[0.0, 0.0]])).any()


def test_batch_equals_single() -> None:
    emb = FakeEmbedder()
    texts = ["refund timelines", "card payments take days", "invoices"]
    batch = emb.embed_passages(texts)
    single = np.stack([emb.embed_passages([t])[0] for t in texts])
    assert np.allclose(batch, single)


def test_local_embedder_applies_prefixes() -> None:
    seen: list[list[str]] = []

    class Model:
        def encode(self, texts: list[str], **_: Any) -> np.ndarray:
            seen.append(texts)
            return np.ones((len(texts), 4))

    emb = LocalEmbedder("m", query_prefix="query: ", passage_prefix="passage: ")
    emb._model = Model()
    emb.embed_passages(["a"])
    q = emb.embed_query("b")
    assert seen == [["passage: a"], ["query: b"]]
    assert math.isclose(float(np.linalg.norm(q)), 1.0, rel_tol=1e-5)


def test_topk_equals_naive_loop() -> None:
    rng = np.random.default_rng(7)
    mat = normalise(rng.normal(size=(500, 32)))
    ids = [f"c{i}" for i in range(500)]
    store = VectorStore()
    store.add(ids, mat)
    for _ in range(20):
        q = normalise(rng.normal(size=(1, 32)))[0]
        naive = sorted(((ids[i], float(mat[i] @ q)) for i in range(500)), key=lambda x: -x[1])
        got = store.search(q, 10)
        assert [g[0] for g in got] == [n[0] for n in naive[:10]]
        assert np.allclose([g[1] for g in got], [n[1] for n in naive[:10]], atol=1e-5)


def test_vector_store_delete_filter_save_load(tmp_path: Path) -> None:
    store = VectorStore()
    store.add(["a", "b", "c"], normalise(np.eye(3)))
    store.delete(["b"])
    assert store.ids == ["a", "c"]
    q = normalise([[0.0, 0.0, 1.0]])[0]
    assert store.search(q, 1)[0][0] == "c"
    assert store.search(q, 2, allowed={"a"}) == [("a", 0.0)]
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.ids == ["a", "c"] and np.allclose(loaded.matrix, store.matrix)
    store.add(["a"], normalise([[0.0, 1.0, 0.0]]))  # re-adding replaces
    assert store.ids.count("a") == 1


def _reference_bm25(corpus: dict[str, str], query: str, k1: float, b: float) -> dict[str, float]:
    docs = {d: tokenize(t) for d, t in corpus.items()}
    n = len(docs)
    avgdl = sum(len(t) for t in docs.values()) / n
    out: dict[str, float] = {}
    for d, terms in docs.items():
        s = 0.0
        for q in set(tokenize(query)):
            df = sum(1 for t in docs.values() if q in t)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            f = terms.count(q)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(terms) / avgdl))
        if s > 0:
            out[d] = s
    return out


def test_bm25_matches_reference_on_toy_corpus(tmp_path: Path) -> None:
    corpus = {
        "1": "refunds are issued within fourteen days",
        "2": "card refunds take three to five working days",
        "3": "invoices have a payment term of thirty days days",
        "4": "shipping uses the regional postal carrier",
    }
    index = BM25Index(k1=1.5, b=0.75)
    index.add(corpus.items())
    for query in ("card refunds days", "postal carrier", "payment", "nothing here"):
        ref = _reference_bm25(corpus, query, 1.5, 0.75)
        got = index.scores(query)
        assert got.keys() == ref.keys()
        for d in ref:
            assert math.isclose(got[d], ref[d], rel_tol=1e-9)
    index.remove(["2"])
    assert "2" not in index.scores("card")
    index.save(tmp_path / "bm25.json")
    again = BM25Index.load(tmp_path / "bm25.json", 1.5, 0.75)
    assert again.scores("days") == index.scores("days")
