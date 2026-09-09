import json

from postergen.config import ContentConfig
from postergen.models import PaperDocument, Section
from postergen.stages.content_understanding import ContentUnderstandingStage


class FakeTextGenerator:
    def generate(self, prompt: str) -> str:
        if "Select the paper sections" in prompt:
            return "[1]"
        assert "Section title: Method" in prompt
        return "- Uses a multi-agent framework.\n- Produces editable poster bullets."


class FakeSectionSelector:
    def generate(self, prompt: str) -> str:
        if "Select the paper sections" in prompt:
            assert "Related Work" not in prompt
            return "[1, 3]"
        return "- Selected by an LLM.\n- Summarized for poster use.\n- Keeps key details."


class FailingTextGenerator:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("temporary provider failure")


def test_content_stage_selects_relevant_sections_and_outputs_bullets(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(title="Conclusion", text="The system produces editable PPTX posters. The conclusion is concise."),
            Section(title="Method", text="We propose a framework for paper-to-poster generation. It has several stages."),
            Section(title="Related Work", text="Prior work is discussed here."),
            Section(title="Introduction", text="This task aims to generate posters from research papers. It is useful."),
        ],
    )

    summaries = ContentUnderstandingStage(ContentConfig(max_sections=3, summary_sentence_range=(1, 2))).select_and_summarize(paper)

    assert [summary.title for summary in summaries] == ["Conclusion", "Method", "Introduction"]
    assert all(summary.summary_sentences[0].startswith("- ") for summary in summaries)
    assert "propose a framework" in summaries[1].text


def test_content_stage_can_use_text_generator(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[Section(title="Method", text="The method section contains a long technical description.")],
    )

    summaries = ContentUnderstandingStage(
        ContentConfig(max_sections=1, summary_sentence_range=(2, 3)),
        text_generator=FakeTextGenerator(),
    ).select_and_summarize(paper)

    assert summaries[0].summary_sentences == [
        "- Uses a multi-agent framework.",
        "- The method section contains a long technical description.",
    ]


def test_content_stage_selects_by_category_but_keeps_original_title(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(
                title="Experiments and Analysis",
                text="The experiments evaluate poster quality and content fidelity.",
                category="Results",
            )
        ],
    )

    summaries = ContentUnderstandingStage(ContentConfig(max_sections=1)).select_and_summarize(paper)

    assert summaries[0].title == "Experiments and Analysis"


def test_content_stage_selects_relevant_sections_from_all_parsed_sections(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(title="Introduction", text="This paper proposes a system for poster generation.", category="Introduction"),
            Section(title="Related Work", text="Prior work is discussed here with many citations " * 20),
            Section(title="Methodology", text="The method introduces a framework with figure processing.", category="Method"),
            Section(title="Experiments and Analysis", text="Experiments evaluate poster quality and content fidelity.", category="Results"),
        ],
    )

    summaries = ContentUnderstandingStage(ContentConfig(max_sections=3)).select_and_summarize(paper)

    assert [summary.title for summary in summaries] == ["Introduction", "Methodology", "Experiments and Analysis"]


def test_content_stage_uses_llm_for_section_selection(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(title="Introduction", text="This paper proposes a system for poster generation.", category="Introduction"),
            Section(title="Related Work", text="Prior work is discussed here with many citations " * 20),
            Section(title="Methodology", text="The method introduces a framework with figure processing.", category="Method"),
        ],
    )

    summaries = ContentUnderstandingStage(
        ContentConfig(max_sections=3, summary_sentence_range=(2, 3)),
        text_generator=FakeSectionSelector(),
    ).select_and_summarize(paper)

    assert [summary.title for summary in summaries] == ["Introduction", "Methodology"]


def test_content_stage_falls_back_when_llm_fails_and_records_selection(tmp_path):
    paper = PaperDocument(
        title="Demo",
        source_path=tmp_path / "demo.pdf",
        sections=[
            Section(
                title="Introduction",
                text="This paper proposes an automatic poster generation system with editable output.",
                category="Introduction",
            ),
            Section(
                title="Methodology",
                text="The method parses papers, selects content, matches figures, and generates a layout.",
                category="Method",
            ),
        ],
    )
    report_dir = tmp_path / "assets"
    stage = ContentUnderstandingStage(
        ContentConfig(max_sections=2, summary_sentence_range=(1, 2)),
        text_generator=FailingTextGenerator(),
        selection_report_dir=report_dir,
    )

    summaries = stage.select_and_summarize(paper)

    assert [summary.title for summary in summaries] == ["Introduction", "Methodology"]
    report = json.loads((report_dir / "demo" / "section_selection.json").read_text())
    assert report["selection_method"] == "rule"
    assert report["selected_section_count"] == 2
    assert [item["title"] for item in report["selected_sections"]] == ["Introduction", "Methodology"]
    assert "temporary provider failure" in report["selection_error"]
    assert len(report["summary_errors"]) == 2


def test_content_stage_prompt_uses_configured_excerpt_and_summary_range():
    config = ContentConfig(summary_sentence_range=(3, 5), selection_excerpt_chars=1200)
    stage = ContentUnderstandingStage(config)
    section = Section(title="Method", text=("The Method uses GPT-4o on 30,460 examples. " + "x" * 1500), category="Method")

    selection_prompt = stage._selection_prompt([(0, section)])
    summary_prompt = stage._summary_prompt("Method", section.text, 3, 5)

    assert "x" * 1150 in selection_prompt
    assert "x" * 1201 not in selection_prompt
    assert "Write 3 to 5 concise bullet points." in summary_prompt
    assert "GPT-4o" in summary_prompt
    assert "30,460 examples" in summary_prompt


def test_content_stage_extracts_key_detail_hints():
    stage = ContentUnderstandingStage(ContentConfig())
    text = (
        "Table 2 reports AUROC of 0.986 on CIFAR-10. "
        "The DAB model compares against Deep Ensemble and VIB baselines."
    )

    details = stage._key_detail_hints(text)

    assert "Table 2" in details
    assert any("0.986" in detail for detail in details)
    assert "DAB" in details


def test_content_stage_extracts_structured_key_facts():
    stage = ContentUnderstandingStage(ContentConfig())
    text = (
        "The P2PINSTRUCT dataset contains 30,460 instruction-response pairs. "
        "The Figure Agent and Section Agent form the core framework. "
        "Table 2 reports ROUGE scores and shows Qwen3-P2P-8B performs best."
    )

    facts = stage._extract_key_facts(text)

    assert any("P2PINSTRUCT" in item for item in facts["datasets_or_benchmarks"])
    assert any("30,460 instruction-response pairs" in item for item in facts["numbers"])
    assert any("Figure Agent" in item for item in facts["methods_or_components"])
    assert any("ROUGE" in item for item in facts["metrics_or_results"])


def test_content_stage_accepts_section_titles_from_llm_response():
    sections = [
        Section(title="Introduction", text="Introduction content."),
        Section(title="Methodology", text="Method content."),
    ]
    stage = ContentUnderstandingStage(ContentConfig())

    selected = stage._parse_selected_section_indices("Introduction and Methodology", list(enumerate(sections)))

    assert selected == [0, 1]


def test_content_stage_accepts_structured_selection_with_roles_and_reasons():
    sections = [
        Section(title="Background", text="Background content."),
        Section(title="Proposed System", text="System content."),
    ]
    stage = ContentUnderstandingStage(ContentConfig())
    response = json.dumps(
        {
            "selected_sections": [
                {
                    "id": 2,
                    "poster_role": "core_contribution",
                    "reason": "Explains the principal system contribution.",
                }
            ]
        }
    )

    selected = stage._parse_selected_section_indices(response, list(enumerate(sections)))

    assert selected == [1]
    assert stage.last_selection_decisions[1] == {
        "poster_role": "core_contribution",
        "reason": "Explains the principal system contribution.",
    }


def test_content_stage_enforces_core_coverage_and_avoids_short_parent_section():
    long_text = "This section provides substantive research details and evidence for the poster. " * 10
    sections = [
        Section(title="Introduction", text=long_text, category="Introduction"),
        Section(title="Methodology", text="This section introduces the method and dataset."),
        Section(title="P2P: Multi-agent for Paper-to-Poster Generation", text=long_text, category="Other"),
        Section(title="P2PINSTRUCT Dataset", text=long_text, category="Dataset"),
        Section(title="Experimental Setup", text=long_text, category="Results"),
        Section(title="Results and Analysis", text=long_text, category="Results"),
        Section(title="Conclusion", text=long_text, category="Conclusion"),
    ]

    stage = ContentUnderstandingStage(ContentConfig(max_sections=5))
    candidates = stage._candidate_sections(sections)
    selected = stage._enforce_selection_constraints(
        sections,
        candidates,
        [sections[0], sections[3], sections[4], sections[6]],
    )

    assert [section.title for section in selected] == [
        "Introduction",
        "P2P: Multi-agent for Paper-to-Poster Generation",
        "P2PINSTRUCT Dataset",
        "Results and Analysis",
        "Conclusion",
    ]
    assert "Methodology" not in [section.title for section in selected]


def test_content_stage_rejects_unsupported_numeric_summary_claim():
    stage = ContentUnderstandingStage(ContentConfig(min_summary_grounding_ratio=0.2))
    source = "The method evaluates 12 models and reports an accuracy of 84 percent."

    grounded = stage._ground_bullets(
        "Results",
        ["- The method evaluates 99 models.", "- The reported accuracy is 84 percent."],
        source,
    )

    assert grounded == ["- The reported accuracy is 84 percent."]
    assert "unsupported numbers: 99" in stage.last_summary_grounding_warnings[0]


def test_content_stage_excludes_short_structural_parent_but_keeps_short_standalone_section():
    sections = [
        Section(title="Methods", text="This section introduces the following components.", level=1, category="Method"),
        Section(
            title="Architecture",
            text="The architecture contains several processing modules and validation components. " * 8,
            level=2,
            category="Method",
        ),
        Section(
            title="Conclusion",
            text="The system produces useful editable outputs and supports further manual refinement.",
            level=1,
            category="Conclusion",
        ),
    ]
    stage = ContentUnderstandingStage(
        ContentConfig(min_section_words=50, min_standalone_section_words=8)
    )

    candidates = [section.title for _, section in stage._candidate_sections(sections)]

    assert candidates == ["Architecture", "Conclusion"]
