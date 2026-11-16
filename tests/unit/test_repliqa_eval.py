from scripts.build_repliqa_eval import gold_chunks, is_unanswerable, split_for, words

PARAGRAPH = (
    "Employees of Veltrano Logistics may carry over at most twelve days of unused leave into "
    "the next calendar year, provided their manager approves the request before 15 December."
)


def test_gold_is_the_chunk_holding_the_supporting_paragraph() -> None:
    chunks = {
        "d_0000": "Veltrano Logistics was founded in a coastal town and ships furniture.",
        # PDF extraction breaks lines and hyphenates words; matching must survive that.
        "d_0001": "Leave policy. Employees of Veltrano Logistics may carry over at most twelve "
        "days of un-\nused leave into the next calendar year, provided their manager approves "
        "the request before 15 December. Sick leave is separate.",
        "d_0002": "Parking spaces are assigned by seniority.",
    }
    assert gold_chunks(PARAGRAPH, chunks) == ["d_0001"]


def test_paragraph_split_across_two_chunks_gives_both() -> None:
    half = len(PARAGRAPH) // 2
    chunks = {"d_0003": PARAGRAPH[: half + 20], "d_0004": PARAGRAPH[half - 20 :]}
    assert sorted(gold_chunks(PARAGRAPH, chunks)) == ["d_0003", "d_0004"]


def test_no_match_gives_no_gold() -> None:
    assert gold_chunks(PARAGRAPH, {"d_0000": "Completely unrelated text about recipes."}) == []


def test_split_is_stable_and_roughly_a_quarter_test() -> None:
    ids = [f"doc{i}-q{j}" for i in range(200) for j in range(1, 6)]
    test_share = sum(split_for(i) == "test" for i in ids) / len(ids)
    assert 0.18 < test_share < 0.32
    assert split_for("abc-q1") == split_for("abc-q1")
    assert words("Un-\nused") == ["unused"]


def test_unanswerable_detection_covers_both_markers() -> None:
    assert is_unanswerable({"answer": "UNANSWERABLE", "long_answer": "NA"})
    assert is_unanswerable(
        {"answer": "The answer is not found in the document.", "long_answer": "NA"}
    )
    assert not is_unanswerable({"answer": "Twelve days.", "long_answer": PARAGRAPH})
