from postergen.config import AssetMatchingConfig
from postergen.models import Figure, SectionSummary, Table
from postergen.stages.asset_structuring import AssetStructurer


class CaptionRewriteGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if "Assign selected figures and tables" in prompt:
            return (
                '{"assignments": ['
                '{"section_id": 1, "asset_type": "figure", "asset_id": "Figure 1", '
                '"reason": "The figure shows the method."}'
                "]}"
            )
        assert "Rewrite the matched figure or table caption" in prompt
        return "Figure 1: Multi-agent architecture of the poster generation system."


class FakeAssetMatcher:
    def generate(self, prompt: str) -> str:
        assert "Assign selected figures and tables" in prompt
        return (
            '{"assignments": ['
            '{"section_id": 1, "asset_type": "figure", "asset_id": "Figure 1", '
            '"reason": "The figure directly illustrates the framework."}'
            "]}"
        )


class PartialAssetMatcher:
    def generate(self, prompt: str) -> str:
        return (
            '{"assignments": ['
            '{"section_id": 1, "asset_type": "figure", "asset_id": "Figure 1", '
            '"reason": "Illustrates the method."}'
            "]}"
        )


class DatasetAssetMatcher:
    def generate(self, prompt: str) -> str:
        return (
            '{"assignments": ['
            '{"section_id": 2, "asset_type": "figure", "asset_id": "Figure 2", '
            '"reason": "The figure shows a paper-to-poster transformation relevant to the dataset section."}'
            "]}"
        )


def test_asset_structurer_matches_fig_abbreviation_reference():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method uses a multi-agent framework."],
            source_text="The method overview is shown in Fig. 2.",
        ),
        SectionSummary(
            title="Results",
            summary_sentences=["The results compare poster quality."],
            source_text="The results compare poster quality.",
        ),
    ]
    figures = [Figure(figure_id="Figure 2", caption="Overview of the proposed framework.")]

    assets = AssetStructurer().build(summaries, figures)

    assert assets[0].figures[0].figure_id == "Figure 2"
    assert assets[0].figures[0].referenced_in == ["Method"]
    assert assets[1].figures == []


def test_asset_structurer_uses_caption_overlap_when_no_reference_exists():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method builds a layout pipeline."],
            source_text="The method builds a layout pipeline.",
        ),
        SectionSummary(
            title="Results",
            summary_sentences=["The experiments evaluate poster readability and quality."],
            source_text="The experiments evaluate poster readability and quality.",
        ),
    ]
    figures = [Figure(figure_id="Figure 3", caption="Poster readability quality scores across experiments.")]

    assets = AssetStructurer().build(summaries, figures)

    assert assets[0].figures == []
    assert assets[1].figures[0].figure_id == "Figure 3"


def test_asset_structurer_can_disable_semantic_matching_for_ablation():
    summaries = [
        SectionSummary(
            title="Results",
            summary_sentences=["The experiments evaluate poster readability and quality."],
            source_text="The experiments evaluate poster readability and quality.",
        )
    ]
    figures = [Figure(figure_id="Figure 3", caption="Poster readability quality scores across experiments.")]
    stage = AssetStructurer(config=AssetMatchingConfig(semantic_matching_enabled=False))

    assets = stage.build(summaries, figures)

    assert assets[0].figures == []
    assert stage.last_assignments[0]["method"] == "unassigned"


def test_asset_structurer_matches_table_reference():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method defines the pipeline."],
            source_text="The method defines the pipeline.",
        ),
        SectionSummary(
            title="Results",
            summary_sentences=["The results compare model scores."],
            source_text="Table 2 reports the human preference comparison.",
        ),
    ]
    tables = [Table(table_id="Table 2", caption="Results of pairwise human preference evaluations.")]

    assets = AssetStructurer().build(summaries, figures=[], tables=tables)

    assert assets[0].tables == []
    assert assets[1].tables[0].table_id == "Table 2"
    assert assets[1].tables[0].referenced_in == ["Results"]


def test_asset_structurer_does_not_treat_trailing_caption_as_body_reference():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method demonstrates the transformation from input to output."],
            source_text="The framework demonstrates the input to output transformation.",
        ),
        SectionSummary(
            title="Dataset",
            summary_sentences=["The dataset contains instruction response pairs."],
            source_text=(
                "The dataset contains instruction response pairs. "
                "Figure 2. An example of the input to output transformation."
            ),
        ),
    ]
    figures = [Figure(figure_id="Figure 2", caption="An example of the input to output transformation.")]

    assets = AssetStructurer().build(summaries, figures)

    assert assets[0].figures[0].figure_id == "Figure 2"
    assert assets[1].figures == []


def test_asset_structurer_leaves_unrelated_asset_unassigned():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method uses a multi-agent framework."],
            source_text="The method uses a multi-agent framework.",
        )
    ]
    figures = [Figure(figure_id="Figure 9", caption="University campus logo and event icon.")]

    assets = AssetStructurer().build(summaries, figures)

    assert assets[0].figures == []
    assert AssetStructurer().config.min_semantic_overlap == 2


def test_asset_structurer_prefers_result_section_for_evaluation_figure():
    summaries = [
        SectionSummary(
            title="Introduction",
            summary_sentences=["The introduction presents the evaluation motivation."],
            source_text="The paper motivates a benchmark for poster evaluation and scoring.",
        ),
        SectionSummary(
            title="Results and Analysis",
            summary_sentences=["The results report evaluation scores and model comparisons."],
            source_text="Experimental results compare model performance using evaluation scores.",
        ),
    ]
    figures = [Figure(figure_id="Figure 3", caption="Overview of the poster evaluation and scoring framework.")]

    assets = AssetStructurer().build(summaries, figures)

    assert assets[0].figures == []
    assert assets[1].figures[0].figure_id == "Figure 3"


def test_asset_structurer_does_not_put_result_table_in_introduction():
    summaries = [
        SectionSummary(
            title="Introduction",
            summary_sentences=["The introduction motivates uncertainty quantification."],
            source_text="The introduction motivates uncertainty quantification.",
        ),
        SectionSummary(
            title="Experimental Results",
            summary_sentences=["The experiments report AUROC and accuracy improvements."],
            source_text="The experiments report AUROC and accuracy improvements.",
        ),
    ]
    tables = [
        Table(
            table_id="Table 4",
            caption="AUROC and accuracy comparison results for all baselines.",
            cells=[["Model", "AUROC"], ["Ours", "0.93"]],
        )
    ]

    assets = AssetStructurer().build(summaries, figures=[], tables=tables)

    assert assets[0].tables == []
    assert assets[1].tables[0].table_id == "Table 4"


def test_asset_structurer_can_use_llm_assignment_with_rule_validation():
    summaries = [
        SectionSummary(
            title="Proposed Framework",
            summary_sentences=["The framework contains three processing agents."],
            source_text="Figure 1 illustrates the proposed framework and its three processing agents.",
        )
    ]
    figures = [Figure(figure_id="Figure 1", caption="Architecture of the proposed three-agent framework.")]
    stage = AssetStructurer(text_generator=FakeAssetMatcher())

    assets = stage.build(summaries, figures)

    assert assets[0].figures[0].figure_id == "Figure 1"
    assert stage.last_matching_method == "llm"
    assert stage.last_assignments[0]["method"] == "llm_validated_explicit_reference"
    assert "directly illustrates" in stage.last_assignments[0]["reason"]


def test_asset_structurer_strict_llm_keeps_direct_assignment_without_rule_validation():
    summaries = [
        SectionSummary(
            title="Introduction",
            summary_sentences=["The paper introduces poster generation."],
            source_text="The introduction motivates the task.",
        ),
        SectionSummary(
            title="P2PINSTRUCT Dataset",
            summary_sentences=["The dataset supports training for paper-to-poster generation."],
            source_text="The dataset contains paper-to-poster instruction pairs.",
        ),
    ]
    figures = [
        Figure(
            figure_id="Figure 2",
            caption="An example of the paper-to-poster transformation.",
        )
    ]
    stage = AssetStructurer(
        config=AssetMatchingConfig(require_llm=True),
        text_generator=DatasetAssetMatcher(),
    )

    assets = stage.build(summaries, figures)

    assert assets[0].figures == []
    assert assets[1].figures[0].figure_id == "Figure 2"
    assert assets[1].figures[0].match_method == "llm_direct"
    assert stage.last_matching_method == "llm"


def test_asset_structurer_completes_high_confidence_assets_omitted_by_llm():
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method uses a processing framework."],
            source_text="Figure 1 illustrates the processing framework.",
        ),
        SectionSummary(
            title="Results",
            summary_sentences=["Experiments compare model performance and quality."],
            source_text="The results compare model performance and poster quality.",
        ),
    ]
    figures = [
        Figure(figure_id="Figure 1", caption="Architecture of the processing framework."),
        Figure(figure_id="Figure 3", caption="Experimental performance and poster quality comparison."),
    ]
    stage = AssetStructurer(text_generator=PartialAssetMatcher())

    assets = stage.build(summaries, figures)

    assert assets[0].figures[0].figure_id == "Figure 1"
    assert assets[1].figures[0].figure_id == "Figure 3"
    assert assets[1].figures[0].match_method == "rule_completion_semantic"
    assert stage.last_matching_method == "llm+rule_completion"


def test_asset_structurer_rewrites_assigned_captions_with_llm(tmp_path):
    summaries = [
        SectionSummary(
            title="Method",
            summary_sentences=["The method uses a multi-agent framework."],
            source_text="Figure 1 illustrates the proposed framework and its agents.",
        )
    ]
    figures = [
        Figure(
            figure_id="Figure 1",
            caption=(
                "Figure 1. The multi-agent architecture of the proposed paper-to-poster "
                "generation system with multiple checker modules and detailed workflow."
            ),
        )
    ]
    generator = CaptionRewriteGenerator()
    stage = AssetStructurer(
        config=AssetMatchingConfig(require_llm=True, rewrite_captions=True),
        text_generator=generator,
        report_dir=tmp_path,
    )

    assets = stage.build(summaries, figures, source_path=tmp_path / "paper.pdf")

    assert generator.calls == 2
    assert assets[0].figures[0].caption == "Figure 1: Multi-agent architecture of the poster generation system."
    assert stage.last_caption_rewrites[0]["method"] == "llm"
