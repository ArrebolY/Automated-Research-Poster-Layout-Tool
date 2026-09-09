import json
import pytest

from postergen.config import AssetSelectionConfig
from postergen.models import Figure, PaperDocument, SectionSummary, Table
from postergen.stages.asset_selection import AssetSelectionStage


class FakeAssetSelector:
    def generate(self, prompt: str) -> str:
        assert "Select the figures and tables" in prompt
        return json.dumps(
            {
                "selected_assets": [
                    {
                        "type": "figure",
                        "id": "Figure 2",
                        "reason": "It explains the proposed architecture.",
                    }
                ]
            }
        )


class FailingAssetSelector:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("provider unavailable")


class PlainTextAssetSelector:
    def generate(self, prompt: str) -> str:
        return "I recommend Figure 2 and Table 1 because they present the method and results."


class DescribingAssetSelector:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Generate a concise semantic description" in prompt:
            if "Figure 2" in prompt:
                return "This figure explains the proposed architecture and generation pipeline."
            return "This table reports the main evaluation results for model comparison."
        assert "description" in prompt
        return json.dumps(
            {
                "selected_assets": [
                    {
                        "type": "figure",
                        "id": "Figure 2",
                        "reason": "The generated description identifies it as the method pipeline.",
                    }
                ]
            }
        )


def _summaries():
    return [
        SectionSummary(
            title="Methodology",
            summary_sentences=["- The method uses a structured generation pipeline."],
            source_text="The proposed architecture is illustrated in Figure 2.",
        ),
        SectionSummary(
            title="Results",
            summary_sentences=["- The experiments compare poster quality."],
            source_text="Table 1 reports the main evaluation results.",
        ),
    ]


def _paper(tmp_path):
    figure_path = tmp_path / "figure.png"
    table_path = tmp_path / "table.png"
    figure_path.write_bytes(b"image")
    table_path.write_bytes(b"table")
    return PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        figures=[
            Figure(figure_id="Figure 1", caption="Company logo", path=figure_path),
            Figure(
                figure_id="Figure 2",
                caption="Overview of the proposed architecture and generation pipeline.",
                path=figure_path,
            ),
        ],
        tables=[
            Table(
                table_id="Table 1",
                caption="Evaluation results and model comparison.",
                cells=[["Model", "Score"], ["Ours", "90"]],
                path=table_path,
            )
        ],
    )


def test_rule_asset_selection_keeps_referenced_assets_and_filters_logo(tmp_path):
    paper = _paper(tmp_path)
    report_dir = tmp_path / "assets"
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=2, max_figures=2, max_tables=1),
        report_dir=report_dir,
    )

    figures, tables = stage.select(paper, _summaries())

    assert [figure.figure_id for figure in figures] == ["Figure 2"]
    assert [table.table_id for table in tables] == ["Table 1"]
    report = json.loads((report_dir / "demo" / "asset_selection.json").read_text())
    assert report["selection_method"] == "rule"
    assert report["candidate_count"] == 3
    assert report["selected_figure_count"] == 1
    assert report["selected_table_count"] == 1
    rejected_logo = next(item for item in report["assets"] if item["id"] == "Figure 1")
    assert rejected_logo["selected"] is False
    assert rejected_logo["score"] < 0


def test_llm_asset_selection_uses_only_returned_valid_ids(tmp_path):
    paper = _paper(tmp_path)
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=3),
        text_generator=FakeAssetSelector(),
    )

    figures, tables = stage.select(paper, _summaries())

    assert [figure.figure_id for figure in figures] == ["Figure 2"]
    assert tables == []
    assert stage.last_selection_method == "llm"
    assert stage.last_decisions[0].reason == "It explains the proposed architecture."


def test_asset_selection_falls_back_to_rules_when_llm_fails(tmp_path):
    paper = _paper(tmp_path)
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=1),
        text_generator=FailingAssetSelector(),
    )

    figures, tables = stage.select(paper, _summaries())

    assert stage.last_selection_method == "rule"
    assert "provider unavailable" in stage.last_selection_error
    assert len(figures) + len(tables) == 1
    assert (figures and figures[0].figure_id == "Figure 2") or (tables and tables[0].table_id == "Table 1")


def test_asset_selection_strict_llm_raises_instead_of_rule_fallback(tmp_path):
    paper = _paper(tmp_path)
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=1, require_llm=True),
        text_generator=FailingAssetSelector(),
    )

    with pytest.raises(RuntimeError, match="rule fallback is disabled"):
        stage.select(paper, _summaries())


def test_asset_selection_accepts_plain_text_ids(tmp_path):
    paper = _paper(tmp_path)
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=2),
        text_generator=PlainTextAssetSelector(),
    )

    figures, tables = stage.select(paper, _summaries())

    assert [figure.figure_id for figure in figures] == ["Figure 2"]
    assert [table.table_id for table in tables] == ["Table 1"]
    assert stage.last_selection_method == "llm"


def test_asset_selection_limits_dense_tables_and_reports_roles(tmp_path):
    table_path = tmp_path / "table.png"
    table_path.write_bytes(b"table")
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "dense.pdf",
        tables=[
            Table(
                table_id="Table 1",
                caption="Full performance comparison results across datasets.",
                cells=[["Model", "Score"], *[[f"M{i}", str(i)] for i in range(25)]],
                path=table_path,
            ),
            Table(
                table_id="Table 2",
                caption="Detailed AUROC comparison results across baselines.",
                cells=[["Model", "AUROC"], *[[f"B{i}", str(i)] for i in range(24)]],
                path=table_path,
            ),
        ],
    )
    report_dir = tmp_path / "assets"
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=3, max_tables=3, max_dense_tables=1),
        report_dir=report_dir,
    )

    _figures, tables = stage.select(paper, _summaries())

    assert len(tables) == 1
    report = json.loads((report_dir / "dense" / "asset_selection.json").read_text())
    dense_assets = [item for item in report["assets"] if item["role"] == "dense_table"]
    assert len(dense_assets) == 2
    assert all(item["suitability"] == "low" for item in dense_assets)


def test_asset_selection_generates_llm_descriptions_before_selection(tmp_path):
    paper = _paper(tmp_path)
    generator = DescribingAssetSelector()
    report_dir = tmp_path / "assets"
    stage = AssetSelectionStage(
        AssetSelectionConfig(max_total_assets=2, describe_assets_with_llm=True, require_llm=True),
        text_generator=generator,
        report_dir=report_dir,
    )

    figures, tables = stage.select(paper, _summaries())

    assert [figure.figure_id for figure in figures] == ["Figure 2"]
    assert tables == []
    assert figures[0].summary == "This figure explains the proposed architecture and generation pipeline."
    assert len(stage.last_descriptions) == 3
    report = json.loads((report_dir / "demo" / "asset_selection.json").read_text())
    selected = next(item for item in report["assets"] if item["id"] == "Figure 2")
    assert selected["description"] == "This figure explains the proposed architecture and generation pipeline."
    assert "proposed architecture" in selected["representation"]
