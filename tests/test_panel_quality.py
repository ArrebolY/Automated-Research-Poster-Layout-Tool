import json

from PIL import Image

from postergen.config import LayoutConfig
from postergen.models import Figure, Panel, PosterLayout
from postergen.stages.panel_quality import PanelQualityStage


def test_panel_quality_reports_small_visual_and_writes_report(tmp_path):
    figure_path = tmp_path / "figure.png"
    Image.new("RGB", (600, 400), "white").save(figure_path)
    panel = Panel(
        title="Results",
        body="- The experiment improves performance.",
        figures=[Figure(figure_id="Figure 1", caption="Result figure.", path=figure_path)],
        tables=[],
        importance=0.2,
        x=0.0,
        y=0.12,
        w=0.25,
        h=0.2,
    )
    stage = PanelQualityStage(LayoutConfig(), report_dir=tmp_path / "assets")

    report = stage.analyze(PosterLayout("Demo", [panel]), source_path=tmp_path / "paper.pdf")

    issue_types = {issue["type"] for issue in report["panels"][0]["issues"]}
    saved = json.loads((tmp_path / "assets" / "paper" / "panel_quality.json").read_text())
    assert "image_too_small" in issue_types
    assert saved["summary"]["issue_count"] >= 1


def test_panel_quality_refinement_increases_visual_panel_importance(tmp_path):
    figure_path = tmp_path / "figure.png"
    Image.new("RGB", (600, 400), "white").save(figure_path)
    visual_panel = Panel(
        title="Method",
        body="- The method uses a compact visual.",
        figures=[Figure(figure_id="Figure 1", caption="Method figure.", path=figure_path)],
        tables=[],
        importance=0.1,
        x=0.0,
        y=0.12,
        w=0.24,
        h=0.2,
    )
    stage = PanelQualityStage(LayoutConfig())

    refined = stage.refine(PosterLayout("Demo", [visual_panel]))

    assert refined.panels[0].importance > visual_panel.importance


def test_panel_quality_refinement_reduces_sparse_text_only_panel():
    panel = Panel(
        title="Conclusion",
        body="- Short takeaway.",
        figures=[],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.8,
        h=0.5,
    )
    stage = PanelQualityStage(LayoutConfig())

    refined = stage.refine(PosterLayout("Demo", [panel]))

    assert refined.panels[0].importance < panel.importance


def test_panel_quality_flags_spacious_text_only_panel():
    panel = Panel(
        title="Motivation",
        body=" ".join(["method"] * 90),
        figures=[],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.45,
        h=0.4,
    )
    stage = PanelQualityStage(LayoutConfig())

    report = stage.analyze(PosterLayout("Demo", [panel]))

    issue_types = {issue["type"] for issue in report["panels"][0]["issues"]}
    assert "pure_text_too_spacious" in issue_types


def test_panel_quality_does_not_flag_spacious_when_text_is_already_dense():
    panel = Panel(
        title="Introduction",
        body=" ".join(["method"] * 140),
        figures=[],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.52,
        h=0.23,
    )
    stage = PanelQualityStage(LayoutConfig())

    report = stage.analyze(PosterLayout("Demo", [panel]))

    issue_types = {issue["type"] for issue in report["panels"][0]["issues"]}
    assert "pure_text_too_spacious" not in issue_types


def test_panel_quality_flags_visual_panel_with_unused_text_space(tmp_path):
    table_path = tmp_path / "table.png"
    Image.new("RGB", (900, 280), "white").save(table_path)
    panel = Panel(
        title="Results and Analysis",
        body=" ".join(["result"] * 25),
        figures=[Figure(figure_id="Table 2", caption="Result table.", path=table_path)],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.43,
        h=0.4,
    )
    stage = PanelQualityStage(LayoutConfig())

    report = stage.analyze(PosterLayout("Demo", [panel]))

    issue_types = {issue["type"] for issue in report["panels"][0]["issues"]}
    assert "visual_panel_unused_text_space" in issue_types


def test_panel_quality_refinement_increases_visual_unused_panel_importance(tmp_path):
    figure_path = tmp_path / "figure.png"
    Image.new("RGB", (900, 280), "white").save(figure_path)
    panel = Panel(
        title="Results and Analysis",
        body=" ".join(["result"] * 25),
        figures=[Figure(figure_id="Figure 2", caption="Result figure.", path=figure_path)],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.43,
        h=0.4,
    )
    stage = PanelQualityStage(LayoutConfig())

    refined = stage.refine(PosterLayout("Demo", [panel]))

    assert refined.panels[0].importance > panel.importance


def test_panel_quality_refinement_reduces_spacious_text_only_panel():
    panel = Panel(
        title="Conclusion",
        body=" ".join(["finding"] * 70),
        figures=[],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=0.7,
        h=0.35,
    )
    stage = PanelQualityStage(LayoutConfig())

    refined = stage.refine(PosterLayout("Demo", [panel]))

    assert refined.panels[0].importance < panel.importance
