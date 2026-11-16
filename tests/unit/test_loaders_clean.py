from pathlib import Path

import pymupdf

from rag.ingest.clean import clean_document
from rag.ingest.loaders import load_file

DOCS = Path(__file__).parents[1] / "fixtures" / "docs"


def test_markdown_sections_and_title() -> None:
    doc = load_file(DOCS / "policies/refund_policy.md", DOCS)
    assert doc.title == "Refund policy"
    assert doc.path == "policies/refund_policy.md"
    timeline = [b for b in doc.blocks if "14 days" in b.text]
    assert timeline and timeline[0].section_path == ("Refund policy", "Timelines")
    assert len(doc.sha256) == 64


def test_html_drops_navigation_and_footer() -> None:
    doc = clean_document(load_file(DOCS / "policies/payments_faq.html", DOCS))
    text = " ".join(b.text for b in doc.blocks)
    assert doc.title == "Payments FAQ"
    assert "3 to 5 working days" in text
    for junk in ("Sign in", "All rights reserved", "trackVisitor", "customer portal"):
        assert junk not in text
    card = next(b for b in doc.blocks if "3 to 5" in b.text)
    assert card.section_path == ("Payments FAQ", "Card payments")


def _make_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    for n in range(1, 5):
        page = pdf.new_page()
        page.insert_text((72, 40), "ACME Internal Handbook", fontsize=9)
        if n == 1:
            page.insert_text((72, 100), "Leave policy", fontsize=20)
        page.insert_text((72, 140), f"Body text on page {n} about annual leave days.", fontsize=11)
        page.insert_text((72, 800), f"Page {n} of 4", fontsize=9)
    pdf.save(path)


def test_pdf_headers_footers_and_page_numbers_removed(tmp_path: Path) -> None:
    pdf_path = tmp_path / "handbook.pdf"
    _make_pdf(pdf_path)
    doc = clean_document(load_file(pdf_path, tmp_path))
    text = " ".join(b.text for b in doc.blocks)
    assert "ACME Internal Handbook" not in text
    assert "of 4" not in text
    assert "Body text on page 3" in text
    assert doc.title == "Leave policy"
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2, 3, 4}
    assert all(b.section_path == ("Leave policy",) for b in doc.blocks)


def test_hyphenation_is_joined() -> None:
    from rag.ingest.clean import normalise_text

    assert normalise_text("infor-\nmation  is\nhere") == "information is here"
