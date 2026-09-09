from pathlib import Path

from postergen.models import Figure, PaperDocument, Section, Table
from postergen.parser_validation import validate_document


def test_parser_validation_reports_entity_metrics(tmp_path):
    document = PaperDocument(
        title="Demo Paper",
        source_path=Path(tmp_path / "demo.pdf"),
        sections=[
            Section(title="Introduction", text="introductory words " * 10, level=1),
            Section(title="Method", text="method words " * 10, level=1),
        ],
        figures=[Figure(figure_id="Figure 1", caption="Method overview")],
        tables=[Table(table_id="Table 1", caption="Evaluation results")],
    )
    expected = {
        "file": "demo.pdf",
        "title": "Demo Paper",
        "sections": [
            {"title": "Introduction", "level": 1},
            {"title": "Method", "level": 1},
        ],
        "figure_ids": ["Figure 1"],
        "table_ids": ["Table 1"],
    }

    result = validate_document(document, expected)

    assert result.title_correct
    assert result.section_f1 == 1.0
    assert result.required_body_pass_rate == 1.0
    assert result.figure_recall == 1.0
    assert result.table_recall == 1.0
