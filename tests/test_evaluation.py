import json

from postergen.batch_eval import aggregate_evaluations, discover_pdfs, read_paper_list
from postergen.evaluation import evaluate_report_dir, write_evaluation


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_evaluate_report_dir_computes_four_module_metrics(tmp_path):
    report_dir = tmp_path / "assets" / "paper_a"
    _write_json(
        report_dir / "parser_quality.json",
        {
            "source_pdf": "paper_a.pdf",
            "backend": "docling",
            "title_present": True,
            "abstract_present": True,
            "section_count": 6,
            "suspicious_section_titles": ["References"],
            "figure_count": 2,
            "table_count": 1,
            "figures_without_captions": [],
            "tables_without_captions": ["Table 1"],
            "missing_files": [],
            "extraction_warnings": ["Excluded uncaptioned picture"],
        },
    )
    _write_json(
        report_dir / "section_selection.json",
        {
            "selection_method": "llm",
            "parsed_section_count": 6,
            "candidate_section_count": 5,
            "selected_section_count": 3,
            "selected_sections": [
                {
                    "title": "Introduction",
                    "effective_category": "introduction",
                    "source_excerpt": "We use 10 papers.",
                    "summary_bullets": ["- We use 10 papers."],
                },
                {
                    "title": "Method",
                    "effective_category": "method",
                    "source_excerpt": "The method builds a parser.",
                    "summary_bullets": ["- The method builds a parser."],
                },
                {
                    "title": "Results",
                    "effective_category": "results",
                    "source_excerpt": "The model scores 90.",
                    "summary_bullets": ["- The model scores 91."],
                },
            ],
        },
    )
    _write_json(
        report_dir / "asset_selection.json",
        {
            "selection_method": "llm",
            "candidate_count": 3,
            "selected_figure_count": 1,
            "selected_table_count": 1,
            "assets": [
                {
                    "type": "figure",
                    "id": "Figure 1",
                    "selected": True,
                    "caption": "Architecture figure.",
                    "evidence": ["explicitly referenced by Method"],
                },
                {
                    "type": "table",
                    "id": "Table 1",
                    "selected": True,
                    "caption": "Results table.",
                    "table_rows": 4,
                    "evidence": ["explicitly referenced by Results"],
                },
                {"type": "figure", "id": "Figure 2", "selected": False, "caption": "Logo."},
            ],
        },
    )
    _write_json(
        report_dir / "asset_matching.json",
        {
            "matching_method": "llm",
            "assignments": [
                {"type": "figure", "id": "Figure 1", "section": "Method"},
                {"type": "table", "id": "Table 1", "section": "Results"},
            ],
        },
    )
    _write_json(
        report_dir / "panel_quality.json",
        {
            "summary": {"issue_count": 1, "issue_counts": {"text_overflow_risk": 1}},
            "panels": [
                {"text_occupancy": 1.2, "visual_area_ratio": 0.0},
                {"text_occupancy": 0.7, "visual_area_ratio": 0.3},
            ],
        },
    )
    _write_json(
        report_dir / "content_fitting.json",
        {
            "panels": [
                {"method": "llm", "action": "compress"},
                {"method": "llm_trimmed", "action": "expand"},
            ]
        },
    )
    _write_json(
        report_dir / "refinement_history.json",
        {"iterations": [{"issue_count": 3}, {"issue_count": 1}]},
    )

    result = write_evaluation(report_dir)

    assert (report_dir / "module_eval.json").is_file()
    assert result["pdf_parsing"]["caption_coverage"] == 0.667
    assert result["section_planning"]["unsupported_number_count"] == 1
    assert result["asset_planning"]["matched_asset_count"] == 2
    assert result["layout_refinement"]["issue_reduction"] == 2


def test_discover_pdfs_supports_paper2poster_style_folders(tmp_path):
    (tmp_path / "sample1").mkdir()
    (tmp_path / "sample1" / "paper.pdf").write_bytes(b"pdf")
    (tmp_path / "sample1" / "poster.pdf").write_bytes(b"poster")
    (tmp_path / "loose.pdf").write_bytes(b"pdf")

    names = [path.name for path in discover_pdfs(tmp_path)]

    assert names.count("paper.pdf") == 1
    assert "loose.pdf" in names
    assert "poster.pdf" not in names


def test_read_paper_list_supports_fixed_evaluation_sets(tmp_path):
    paper = tmp_path / "sample" / "paper.pdf"
    paper.parent.mkdir()
    paper.write_bytes(b"pdf")
    paper_list = tmp_path / "papers.txt"
    paper_list.write_text("# fixed set\n\nsample/paper.pdf\n", encoding="utf-8")

    assert read_paper_list(paper_list) == [paper]


def test_read_paper_list_rejects_empty_pdfs(tmp_path):
    paper = tmp_path / "sample" / "paper.pdf"
    paper.parent.mkdir()
    paper.write_bytes(b"")
    paper_list = tmp_path / "papers.txt"
    paper_list.write_text("sample/paper.pdf\n", encoding="utf-8")

    try:
        read_paper_list(paper_list)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty PDFs to be rejected")


def test_aggregate_evaluations_averages_numeric_fields():
    evaluations = [
        {
            "paper_id": "a",
            "pdf_parsing": {"section_count": 4},
            "section_planning": {},
            "asset_planning": {},
            "layout_refinement": {"final_issue_count": 2},
            "overall": {"generation_ready": True},
        },
        {
            "paper_id": "b",
            "pdf_parsing": {"section_count": 8},
            "section_planning": {},
            "asset_planning": {},
            "layout_refinement": {"final_issue_count": 0},
            "overall": {"generation_ready": False},
        },
    ]

    aggregate = aggregate_evaluations(evaluations)

    assert aggregate["avg_pdf_parsing.section_count"] == 6.0
    assert aggregate["avg_layout_refinement.final_issue_count"] == 1.0
