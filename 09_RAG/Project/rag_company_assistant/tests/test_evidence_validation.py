
from app.rag.generator import evidence_is_present


def test_accepts_exact_evidence():
    context = "Employees receive 20 days of annual leave."
    evidence = "20 days of annual leave"

    assert evidence_is_present(evidence, context)


def test_rejects_invented_evidence():
    context = "Employees receive 20 days of annual leave."
    evidence = "Employees receive 14 days of maternity leave."

    assert not evidence_is_present(evidence, context)


def test_rejects_empty_evidence():
    assert not evidence_is_present("", "Some document text.")


def test_rejects_whitespace_evidence():
    assert not evidence_is_present("   ", "Some document text.")
