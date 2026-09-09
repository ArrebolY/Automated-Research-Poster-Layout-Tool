from postergen.config import LayoutConfig
from postergen.models import Figure, SectionSummary, StructuredAsset
from postergen.stages.layout import LayoutGenerationStage
import json


def test_layout_generation_assigns_normalized_positions():
    stage = LayoutGenerationStage(LayoutConfig())
    assets = [
        StructuredAsset(
            section=SectionSummary(
                title="Introduction",
                summary_sentences=["A short summary."],
                source_text="A short summary.",
            ),
            figures=[],
        ),
        StructuredAsset(
            section=SectionSummary(
                title="Method",
                summary_sentences=["A longer summary with more tokens for testing the layout importance score."],
                source_text="A longer summary with more tokens for testing the layout importance score.",
            ),
            figures=[Figure(figure_id="Figure 1", caption="Example figure")],
        ),
    ]

    layout = stage.generate("Demo Poster", assets)

    assert len(layout.panels) == 2
    for panel in layout.panels:
        assert 0.0 <= panel.x <= 1.0
        assert 0.0 <= panel.y <= 1.0
        assert 0.0 <= panel.w <= 1.0
        assert 0.0 <= panel.h <= 1.0


def test_layout_uses_variable_panel_heights_for_content_density():
    stage = LayoutGenerationStage(LayoutConfig())
    assets = [
        StructuredAsset(
            section=SectionSummary(
                title="Introduction",
                summary_sentences=[" ".join(["intro"] * 80)],
                source_text=" ".join(["intro"] * 80),
            ),
            figures=[Figure(figure_id="Figure 1", caption="Overview figure")],
        ),
        StructuredAsset(
            section=SectionSummary(title="Method", summary_sentences=["Short method."], source_text="Short method."),
            figures=[],
        ),
        StructuredAsset(
            section=SectionSummary(title="Results", summary_sentences=["Short results."], source_text="Short results."),
            figures=[],
        ),
    ]

    layout = stage.generate("Demo Poster", assets)
    panels = {panel.title: panel for panel in layout.panels}

    assert panels["Introduction"].h > panels["Results"].h


def test_layout_uses_constrained_binary_partitioning_below_title_area():
    stage = LayoutGenerationStage(LayoutConfig())
    assets = [
        StructuredAsset(
            section=SectionSummary(title="Introduction", summary_sentences=[" ".join(["intro"] * 90)], source_text=" ".join(["intro"] * 90)),
            figures=[Figure(figure_id="Figure 1", caption="Overview figure")],
        ),
        StructuredAsset(
            section=SectionSummary(title="Method", summary_sentences=[" ".join(["method"] * 50)], source_text=" ".join(["method"] * 50)),
            figures=[Figure(figure_id="Figure 2", caption="Method figure")],
        ),
        StructuredAsset(
            section=SectionSummary(title="Results", summary_sentences=["Short results."], source_text="Short results."),
            figures=[],
        ),
        StructuredAsset(
            section=SectionSummary(title="Conclusion", summary_sentences=[" ".join(["conclusion"] * 30)], source_text=" ".join(["conclusion"] * 30)),
            figures=[],
        ),
    ]

    layout = stage.generate("Demo Poster", assets)
    widths = {round(panel.w, 3) for panel in layout.panels}
    heights = {round(panel.h, 3) for panel in layout.panels}

    assert len(widths) > 1 or len(heights) > 1
    assert all(panel.y >= 0.12 for panel in layout.panels)


def test_layout_preserves_selected_section_order_and_avoids_third_column():
    stage = LayoutGenerationStage(LayoutConfig())
    titles = ["Introduction", "Proposed System", "Dataset", "Results and Analysis", "Conclusion"]
    assets = [
        StructuredAsset(
            section=SectionSummary(title=title, summary_sentences=["A concise summary."], source_text="Text."),
            figures=[],
        )
        for title in titles
    ]

    layout = stage.generate("Demo", assets)

    assert [panel.title for panel in layout.panels] == titles
    assert len({round(panel.x, 3) for panel in layout.panels}) <= 2


def test_layout_writes_quality_report(tmp_path):
    stage = LayoutGenerationStage(LayoutConfig(), report_dir=tmp_path / "assets")
    source = tmp_path / "paper.pdf"
    assets = [
        StructuredAsset(
            section=SectionSummary(title="Conclusion", summary_sentences=["Short takeaway."], source_text="Short takeaway."),
            figures=[],
        )
    ]

    stage.generate("Demo", assets, source_path=source)

    report = json.loads((tmp_path / "assets" / "paper" / "layout_quality.json").read_text())
    assert report["layout_method"] == "heuristic_sp_rp_loss_tree_split"
    assert report["panels"][0]["title"] == "Conclusion"
    assert "preferred_aspect_ratio" in report["panels"][0]


def test_template_layout_uses_fixed_grid():
    stage = LayoutGenerationStage(LayoutConfig(strategy="template"))
    assets = [
        StructuredAsset(
            section=SectionSummary(title=f"Section {index}", summary_sentences=["Text."], source_text="Text."),
            figures=[],
        )
        for index in range(4)
    ]

    layout = stage.generate("Demo", assets)

    assert len({round(panel.w, 3) for panel in layout.panels}) == 1
    assert len({round(panel.h, 3) for panel in layout.panels}) == 1
