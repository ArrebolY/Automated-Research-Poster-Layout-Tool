import json
from pathlib import Path

from postergen.models import PaperDocument, Section
from postergen.stages.docling_parser import DoclingPaperParser


class FakeImage:
    def save(self, path, _format):
        Path(path).write_bytes(b"fake-png")


class FakeCell:
    def __init__(self, text):
        self.text = text


class FakeTableData:
    grid = [[FakeCell("Metric"), FakeCell("Score")], [FakeCell("F1"), FakeCell("0.91")]]


class FakePictureItem:
    def caption_text(self, _document):
        return "Figure 2: Overview of the proposed pipeline."

    def get_image(self, _document):
        return FakeImage()


class FakeTableItem:
    data = FakeTableData()

    def caption_text(self, _document):
        return "Table 1: Main evaluation results."

    def get_image(self, _document):
        return FakeImage()


class FakeDoclingDocument:
    def export_to_markdown(self):
        return """## Demo Paper

## Demo Author

## Abstract

This paper presents a compact parsing pipeline.

## 1 Introduction

The introduction explains the task and motivation.

### 1.1 Method

The method extracts structured text, figures, and tables.

## References

References should not become poster sections.
"""

    def iterate_items(self):
        yield FakePictureItem(), 1
        yield FakeTableItem(), 1


class FakeFallback:
    def parse(self, pdf_path):
        return PaperDocument(
            title="Fallback Paper",
            source_path=pdf_path,
            abstract="Fallback abstract.",
            sections=[Section(title="Introduction", text="Fallback section text.")],
        )


def test_docling_parser_maps_markdown_and_assets_to_existing_models(tmp_path):
    parser = DoclingPaperParser(asset_dir=tmp_path / "assets")
    pdf_path = tmp_path / "demo.pdf"

    paper = parser._paper_from_docling_document(
        pdf_path,
        FakeDoclingDocument(),
        picture_type=FakePictureItem,
        table_type=FakeTableItem,
    )

    assert paper.title == "Demo Paper"
    assert paper.abstract == "This paper presents a compact parsing pipeline."
    assert [(section.title, section.level) for section in paper.sections] == [
        ("Introduction", 1),
        ("Method", 2),
    ]
    assert paper.figures[0].figure_id == "Figure 2"
    assert paper.figures[0].path.is_file()
    assert paper.tables[0].table_id == "Table 1"
    assert paper.tables[0].cells == [["Metric", "Score"], ["F1", "0.91"]]
    assert paper.tables[0].path.is_file()
    assert (tmp_path / "assets" / "demo" / "paper.md").is_file()


def test_docling_parser_uses_legacy_fallback_and_writes_report(tmp_path, monkeypatch):
    parser = DoclingPaperParser(asset_dir=tmp_path / "assets", fallback=FakeFallback())
    pdf_path = tmp_path / "fallback.pdf"

    def fail_docling(_pdf_path):
        raise ModuleNotFoundError("docling is not installed")

    monkeypatch.setattr(parser, "_parse_with_docling", fail_docling)
    paper = parser.parse(pdf_path)

    report_path = tmp_path / "assets" / "fallback" / "parser_quality.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert paper.title == "Fallback Paper"
    assert report["backend"] == "legacy"
    assert "docling is not installed" in report["fallback_reason"]
