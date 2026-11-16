from evals.metrics import (
    Question,
    QuestionResult,
    changed_questions,
    recall_at_k,
    reciprocal_rank,
    summarise,
)


def _r(
    qid: str,
    retrieved: list[str],
    refused: bool = False,
    answerable: bool = True,
    faithful: bool | None = None,
) -> QuestionResult:
    return QuestionResult(
        id=qid,
        answerable=answerable,
        retrieved=retrieved,
        refused=refused,
        answer="",
        latency_ms=100,
        faithful=faithful,
    )


def test_recall_and_mrr() -> None:
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], ["f"]) == 0.0
    assert recall_at_k(["a", "b"], ["b", "z"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_summary() -> None:
    qs = [
        Question(id="q1", question="?", gold_chunk_ids=["a"]),
        Question(id="q2", question="?", gold_chunk_ids=["b"]),
        Question(id="q3", question="?", answerable=False),
    ]
    rs = [
        _r("q1", ["a"], faithful=True),
        _r("q2", ["x", "b"], faithful=False),
        _r("q3", ["x"], refused=True, answerable=False),
    ]
    m = summarise(qs, rs)
    assert m["recall_at_5"] == 1.0
    assert m["mrr"] == 0.75
    assert m["refusal_accuracy"] == 1.0
    assert m["false_refusal_rate"] == 0.0
    assert m["faithfulness"] == 0.5


def test_changed_questions() -> None:
    old = [_r("q1", ["a"]), _r("q2", ["x"])]
    new = [_r("q1", ["a"]), _r("q2", ["b"])]
    diff = changed_questions(old, new, {"q1": ["a"], "q2": ["b"]})
    assert len(diff) == 1 and diff[0].startswith("q2")
