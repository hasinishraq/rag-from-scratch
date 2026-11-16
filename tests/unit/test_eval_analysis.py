from typing import Any

from evals.analysis import analyse, gold_rank, latency_summary, std_error
from evals.metrics import Question


def _q(qid: str, topic: str, gold: list[str], answerable: bool = True) -> Question:
    data = {
        "id": qid,
        "question": f"{qid}?",
        "gold_chunk_ids": gold,
        "answerable": answerable,
        "topic": topic,
    }
    return Question.model_validate(data)


def _r(
    qid: str,
    retrieved: list[str],
    answerable: bool = True,
    latency: int = 100,
    refused: bool = False,
) -> dict[str, Any]:
    return {
        "id": qid,
        "answerable": answerable,
        "retrieved": retrieved,
        "refused": refused,
        "answer": "",
        "latency_ms": latency,
    }


QS = {
    q.id: q
    for q in [
        _q("a", "Policies", ["g1"]),
        _q("b", "Policies", ["g2"]),
        _q("c", "Incidents", ["g3"]),
        _q("u", "Incidents", [], answerable=False),
    ]
}
RUN = {
    "run": "r2",
    "config": "c.yaml",
    "split": "dev",
    "retrieval_only": True,
    "metrics": {"recall_at_5": 2 / 3, "mrr": 0.5},
    "results": [
        _r("a", ["g1", "x", "y"], latency=100),
        _r("b", ["x", "g2", "y"], latency=300),
        _r("c", ["x", "y", "z"], latency=200),
        _r("u", ["x"], answerable=False, latency=150),
    ],
}


def test_rank_curve_and_histogram() -> None:
    a = analyse(RUN, QS)
    assert a["n_answerable"] == 3 and a["n_unanswerable"] == 1
    assert a["n"] == 4  # every question, not only the scored ones
    assert [p["value"] for p in a["recall_at_k"]] == [
        round(1 / 3, 4),
        round(2 / 3, 4),
        round(2 / 3, 4),
    ]
    hist = {h["rank"]: h["count"] for h in a["rank_hist"]}
    assert hist == {"1": 1, "2": 1, "3": 0, "not found": 1}
    assert [m["id"] for m in a["misses"]] == ["c"]
    assert a["generation"] is None  # retrieval-only run


def test_by_topic_and_latency() -> None:
    a = analyse(RUN, QS)
    topics = {t["topic"]: t for t in a["by_topic"]}
    assert topics["Policies"]["recall_at_5"] == 1.0 and topics["Policies"]["mrr"] == 0.75
    assert topics["Incidents"]["misses"] == 1 and topics["Incidents"]["unanswerable"] == 1
    lat = a["latency"]
    assert (lat["min"], lat["max"], lat["p50"]) == (100, 300, 200)
    assert sum(b["count"] for b in lat["hist"]) == 4


def test_comparison_with_previous_run() -> None:
    previous = {**RUN, "run": "r1", "results": [_r("a", ["x", "y", "z"]), *RUN["results"][1:]]}
    a = analyse(RUN, QS, previous)
    assert a["previous"]["run"] == "r1"
    assert len(a["changes"]) == 1 and a["changes"][0].startswith("a:")


def test_generation_summary() -> None:
    run = {
        **RUN,
        "retrieval_only": False,
        "results": [
            {
                **_r("a", ["g1"]),
                "faithful": True,
                "correct": True,
                "provider": "groq",
                "tokens": 500,
            },
            {**_r("u", ["x"], answerable=False, refused=True), "provider": None},
        ],
    }
    g = analyse(run, QS)["generation"]
    assert g["correct_refusals"] == 1 and g["faithful"] == 1 and g["tokens"] == 500


def test_helpers() -> None:
    assert gold_rank(["a", "b"], ["b"]) == 2 and gold_rank(["a"], ["z"]) is None
    assert std_error(0.5, 100) == 0.05 and std_error(None, 10) is None
    assert latency_summary([]) == {"n": 0, "hist": []}


def test_latency_outlier_goes_to_an_overflow_bucket() -> None:
    values = [100 + i for i in range(200)] + [10_000]
    lat = latency_summary(values)
    assert lat["max"] == 10_000 and lat["hist"][-1].get("overflow") is True
    assert lat["hist"][-1]["count"] >= 1
    assert sum(b["count"] for b in lat["hist"]) == len(values)
    assert lat["hist"][-2]["to_ms"] < 1_000  # the regular buckets stay near the bulk


def test_misses_record_whether_the_same_document_came_first() -> None:
    qs = {"m": _q("m", "Policies", ["d1_0004"]), "n": _q("n", "Policies", ["d1_0009"])}
    run = {**RUN, "results": [_r("m", ["d1_0001", "d2_0001"]), _r("n", ["d7_0000"])]}
    a = analyse(run, qs)
    same = {m["id"]: m["same_document"] for m in a["misses"]}
    assert same == {"m": True, "n": False} and a["misses_same_document"] == 1
