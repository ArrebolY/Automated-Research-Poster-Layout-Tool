import fitz

from postergen.stages.parser import PaperParser


def _write_test_pdf(pdf_path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=8)
    document.save(pdf_path)
    document.close()


def test_parser_extracts_sections_abstract_and_figure_caption(tmp_path):
    pdf_path = tmp_path / "demo-paper.pdf"
    _write_test_pdf(
        pdf_path,
        """A Demo Paper for Poster Generation
Abstract
This paper studies automatic poster generation from research papers.
Introduction
The introduction explains the motivation and cites Figure 1.
Method
The method describes a staged pipeline for content and layout generation.
Figure 1: Overview of the proposed paper-to-poster pipeline.
Table 1: Main evaluation metrics used in the experiment.
Results
The results show that the generated poster preserves important information.
Conclusion
The conclusion highlights editable PPTX output.""",
    )

    paper = PaperParser().parse(pdf_path)

    assert paper.title == "A Demo Paper for Poster Generation"
    assert "automatic poster generation" in paper.abstract
    assert [section.title for section in paper.sections] == ["Introduction", "Method", "Results", "Conclusion"]
    assert paper.figures[0].figure_id == "Figure 1"
    assert "paper-to-poster pipeline" in paper.figures[0].caption
    assert paper.tables[0].table_id == "Table 1"
    assert "evaluation metrics" in paper.tables[0].caption


def test_parser_preserves_original_section_titles_with_categories(tmp_path):
    pdf_path = tmp_path / "demo-paper.pdf"
    _write_test_pdf(
        pdf_path,
        """Demo Paper
Abstract
This is an abstract.
1. Introduction
The paper introduces the task.
2. Methodology
The method is described here.
4. Experiments and Analysis
The experiments are reported here.
6. Conclusion
The paper concludes.""",
    )

    paper = PaperParser().parse(pdf_path)

    assert [section.title for section in paper.sections] == [
        "Introduction",
        "Methodology",
        "Experiments and Analysis",
        "Conclusion",
    ]
    assert [section.category for section in paper.sections] == ["Introduction", "Method", "Results", "Conclusion"]


def test_parser_matches_long_evaluation_section_titles(tmp_path):
    pdf_path = tmp_path / "demo-paper.pdf"
    _write_test_pdf(
        pdf_path,
        """Demo Paper
Abstract
This is an abstract.
1. Introduction
The paper introduces the task.
2. Methodology
The method is described here.
3. P2P EVAL: A Fine-grained Benchmark for Poster Evaluation
The benchmark evaluates generated posters.
4. Experiments and Analysis
The experiments are reported here.
6. Conclusion
The paper concludes.""",
    )

    paper = PaperParser().parse(pdf_path)

    assert "P2P EVAL: A Fine-grained Benchmark for Poster Evaluation" in [
        section.title for section in paper.sections
    ]
    assert [section.category for section in paper.sections] == [
        "Introduction",
        "Method",
        "Evaluation",
        "Results",
        "Conclusion",
    ]


def test_parser_cleans_pdfplumber_table_cells():
    parser = PaperParser()

    cells = parser._clean_table_cells(
        [
            ["Metric", "Score", None],
            ["Quality", " 0.82 ", ""],
            ["", "", None],
        ]
    )

    assert cells == [["Metric", "Score", ""], ["Quality", "0.82", ""]]


def test_parser_combines_split_numbered_headings(tmp_path):
    pdf_path = tmp_path / "demo-paper.pdf"
    _write_test_pdf(
        pdf_path,
        """Demo Paper
Abstract
This is an abstract.
1. Introduction
The paper introduces the task.
3.2
Poster Evaluation Framework
The framework evaluates poster quality.
6. Conclusion
The paper concludes.""",
    )

    paper = PaperParser().parse(pdf_path)

    assert [section.title for section in paper.sections] == [
        "Introduction",
        "Poster Evaluation Framework",
        "Conclusion",
    ]


def test_parser_ignores_table_notes_and_references(tmp_path):
    pdf_path = tmp_path / "demo-paper.pdf"
    _write_test_pdf(
        pdf_path,
        """Demo Paper
Abstract
This is an abstract.
1. Introduction
The paper introduces the task.
4.1 Experimental Setup
The experiment compares several models.
0.2745
Claude-3.7-Sonnet
1 F1 scores of BERTScore.
4 Percentage of total poster area occupied by blank space.
References
1. A Reference Title
Reference text should not become a section.
6. Conclusion
This text appears after references and should be ignored.""",
    )

    paper = PaperParser().parse(pdf_path)

    assert [section.title for section in paper.sections] == ["Introduction", "Experimental Setup"]


def test_parser_uses_font_style_for_unnumbered_headings(tmp_path):
    pdf_path = tmp_path / "styled-paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 55), "Styled Research Paper", fontsize=14, fontname="hebo")
    page.insert_text((72, 90), "Abstract", fontsize=10, fontname="hebo")
    page.insert_text((72, 110), "This paper studies structured PDF parsing.", fontsize=8)
    page.insert_text((72, 150), "Introduction", fontsize=12, fontname="hebo")
    page.insert_text((72, 170), "The introduction explains the research problem and cites Figure 1.", fontsize=8)
    page.insert_text((72, 210), "System Overview", fontsize=12, fontname="hebo")
    page.insert_text((72, 230), "The system extracts sections, figures, tables, and captions from papers.", fontsize=8)
    page.insert_text((72, 270), "Figure 1: Overview of the parsing system.", fontsize=8)
    document.save(pdf_path)
    document.close()

    paper = PaperParser().parse(pdf_path)

    assert [section.title for section in paper.sections] == ["Introduction", "System Overview"]
    assert paper.figures[0].referenced_in == ["Introduction"]


def test_parser_does_not_treat_table_decimals_as_split_section_numbers(tmp_path):
    pdf_path = tmp_path / "table-noise.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 55), "Table Noise Paper", fontsize=14, fontname="hebo")
    page.insert_text((72, 90), "Introduction", fontsize=12, fontname="hebo")
    page.insert_text((72, 110), "The introduction contains enough text for section parsing.", fontsize=8)
    page.insert_text((72, 150), "6.54", fontsize=8)
    page.insert_text((150, 150), "Original posters", fontsize=8)
    page.insert_text((72, 190), "Conclusion", fontsize=12, fontname="hebo")
    page.insert_text((72, 210), "The conclusion summarizes the parsing experiment.", fontsize=8)
    document.save(pdf_path)
    document.close()

    paper = PaperParser().parse(pdf_path)

    assert [section.title for section in paper.sections] == ["Introduction", "Conclusion"]
