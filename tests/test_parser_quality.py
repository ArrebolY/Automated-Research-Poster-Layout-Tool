from pathlib import Path

from postergen.models import Figure, PaperDocument, Section, Table
from postergen.parser_quality import build_parser_quality_report


def test_parser_quality_flags_suspicious_titles_and_missing_assets(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(title="Introduction", text="Valid text."),
            Section(title="Figure 3: Incorrect heading", text="Noise."),
        ],
        figures=[Figure(figure_id="Figure 1", caption="", path=tmp_path / "missing.png")],
        tables=[Table(table_id="Table 1", caption="Figure 4: Wrongly associated caption", path=None)],
    )

    report = build_parser_quality_report(
        paper,
        backend="docling",
        extraction_warnings=["Excluded uncaptioned picture"],
    )

    assert report.section_count == 2
    assert report.suspicious_section_titles == ["Figure 3: Incorrect heading"]
    assert report.figures_without_captions == ["Figure 1"]
    assert report.suspicious_captions == ["Table 1: Figure 4: Wrongly associated caption"]
    assert report.extraction_warnings == ["Excluded uncaptioned picture"]
    assert str(tmp_path / "missing.png") in report.missing_files
    assert "Table 1: no image path" in report.missing_files
