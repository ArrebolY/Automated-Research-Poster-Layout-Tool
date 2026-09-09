import json

from postergen.config import ContentFittingConfig, LayoutConfig
from postergen.models import Panel, PosterLayout
from postergen.stages.content_fitting import ContentFittingStage


def _panel(title: str, body: str, *, w: float, h: float) -> Panel:
    return Panel(
        title=title,
        body=body,
        figures=[],
        tables=[],
        importance=1.0,
        x=0.0,
        y=0.12,
        w=w,
        h=h,
    )


def test_content_fitting_expands_sparse_panel_from_source():
    panel = _panel("Method", "- The system processes papers.", w=1.0, h=0.5)
    source = (
        "The system processes research papers automatically. "
        "A parser extracts structured text and visual assets. "
        "A selector identifies sections suitable for poster presentation. "
        "The layout module allocates space according to content demand. "
        "The exporter produces an editable PowerPoint poster. "
        "Each component can be evaluated independently during experiments."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Method": source})

    assert len(result.panels[0].body.splitlines()) > 1
    assert stage.last_decisions[0]["action"] == "expand"
    assert stage.last_decisions[0]["words_after"] > stage.last_decisions[0]["words_before"]


def test_content_fitting_compresses_dense_panel():
    bullets = [f"- Important result {index} " + " ".join(["detail"] * 28) + "." for index in range(6)]
    panel = _panel("Results", "\n".join(bullets), w=0.34, h=0.22)
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Results": "Original result text."})

    assert stage.last_decisions[0]["action"] == "compress"
    assert stage.last_decisions[0]["words_after"] < stage.last_decisions[0]["words_before"]
    assert len(result.panels[0].body.splitlines()) >= 2


class FittingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "Action: expand" in prompt
        return (
            "- The system processes research papers automatically.\n"
            "- A parser extracts structured text and visual assets.\n"
            "- The exporter produces an editable PowerPoint poster."
        )


class ForcedExpandGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompt = prompt
        assert "Action: expand" in prompt
        assert "unused space" in prompt
        return (
            "- The method frames uncertainty as a rate-distortion problem.\n"
            "- It learns compressed representations through a codebook of encoders.\n"
            "- The experiments show stronger out-of-distribution detection performance.\n"
            "- The approach also improves calibration for misclassification prediction."
        )


class FactAwareGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return (
            "- P2PINSTRUCT contains 30,460 instruction-response pairs.\n"
            "- Qwen3-P2P-8B achieves the strongest ROUGE scores."
        )


def test_content_fitting_uses_llm_only_for_panel_needing_adjustment(tmp_path):
    sparse = _panel("Method", "- The system processes papers.", w=1.0, h=0.5)
    source = (
        "The system processes research papers automatically. "
        "A parser extracts structured text and visual assets. "
        "The exporter produces an editable PowerPoint poster."
    )
    generator = FittingGenerator()
    stage = ContentFittingStage(
        ContentFittingConfig(),
        LayoutConfig(),
        generator,
        report_dir=tmp_path,
    )

    result = stage.fit(
        PosterLayout("Demo", [sparse]),
        {"Method": source},
        source_path=tmp_path / "paper.pdf",
    )

    report = json.loads((tmp_path / "paper" / "content_fitting.json").read_text())
    assert generator.calls == 2
    assert stage.last_decisions[0]["method"] in {"llm", "llm_trimmed"}
    assert stage.last_decisions[0]["iterations"] == 2
    assert result.panels[0].body.count("\n") == 2
    assert report["panels"][0]["action"] == "expand"


def test_content_fitting_passes_priority_facts_to_llm():
    panel = _panel("Results", "- P2P is evaluated.", w=1.0, h=0.5)
    source = (
        "P2PINSTRUCT contains 30,460 instruction-response pairs. "
        "Qwen3-P2P-8B achieves the strongest ROUGE scores."
    )
    facts = {
        "datasets_or_benchmarks": ["P2PINSTRUCT"],
        "numbers": ["30,460 instruction-response pairs"],
        "metrics_or_results": ["Qwen3-P2P-8B achieves the strongest ROUGE scores"],
    }
    generator = FactAwareGenerator()
    stage = ContentFittingStage(ContentFittingConfig(max_iterations=1), LayoutConfig(), generator)

    stage.fit(PosterLayout("Demo", [panel]), {"Results": source}, {"Results": facts})

    assert "Priority facts to preserve when possible:" in generator.prompt
    assert "30,460 instruction-response pairs" in generator.prompt
    assert stage.last_decisions[0]["priority_facts"] == facts


def test_content_fitting_can_force_expand_spacious_panel():
    panel = _panel("Conclusion", "- Short takeaway.", w=0.55, h=0.22)
    source = (
        "The method frames uncertainty as a rate-distortion problem. "
        "It learns compressed representations through a codebook of encoders. "
        "The experiments show stronger out-of-distribution detection performance. "
        "The approach also improves calibration for misclassification prediction."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig())

    result = stage.fit(
        PosterLayout("Demo", [panel]),
        {"Conclusion": source},
        forced_actions={"Conclusion": "expand"},
    )

    assert stage.last_decisions[0]["forced"] is True
    assert stage.last_decisions[0]["action"] == "expand"
    assert len(result.panels[0].body.splitlines()) > 1


def test_content_fitting_forced_expand_uses_llm_feedback_reason():
    panel = _panel(
        "Conclusion",
        "- The method frames uncertainty as a rate-distortion problem.",
        w=0.6,
        h=0.25,
    )
    source = (
        "The method frames uncertainty as a rate-distortion problem. "
        "It learns compressed representations through a codebook of encoders. "
        "The experiments show stronger out-of-distribution detection performance. "
        "The approach also improves calibration for misclassification prediction."
    )
    generator = ForcedExpandGenerator()
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig(), generator)

    result = stage.fit(
        PosterLayout("Demo", [panel]),
        {"Conclusion": source},
        forced_actions={"Conclusion": "expand"},
        forced_reasons={"Conclusion": "The quality check detected excessive unused space in this panel."},
    )

    assert generator.calls == 2
    assert stage.last_decisions[0]["method"] in {"llm", "llm_trimmed"}
    assert stage.last_decisions[0]["iterations"] == 2
    assert stage.last_decisions[0]["forced_reason"] == "The quality check detected excessive unused space in this panel."
    assert stage.last_decisions[0]["words_after"] > stage.last_decisions[0]["words_before"]
    assert result.panels[0].body.count("\n") == 3


class UngroundedGenerator:
    def generate(self, prompt: str) -> str:
        return "- The experiment improves accuracy by 99 percent."


class IncompleteSentenceGenerator:
    def generate(self, prompt: str) -> str:
        return (
            "- The orchestrate agent assembles HTML posters and.\n"
            "- The checker modules refine text coherence and layout aesthetics."
        )


class DanglingPhraseGenerator:
    def generate(self, prompt: str) -> str:
        return (
            "- We propose P2P, the first flexible framework for generating.\n"
            "- Poster Checker evaluates layout aesthetics, triggering iterative.\n"
            "- A special case at threshold M = N will be formally proven for.\n"
            "- The framework creates editable academic posters from research papers."
        )


class OverlongGenerator:
    def generate(self, prompt: str) -> str:
        return "\n".join(
            [
                "- The system processes research papers automatically with extensive details about every stage."
                for _ in range(8)
            ]
        )


class GroundedOverlongGenerator:
    def generate(self, prompt: str) -> str:
        return "\n".join(
            [
                "- The results compare poster quality across multiple models and evaluation settings.",
                "- The evaluation includes human preference and automatic metrics for poster quality.",
                "- The ablation confirms that reflection improves poster generation and evaluation.",
                "- The results compare poster quality across multiple models and evaluation settings.",
                "- The evaluation includes human preference and automatic metrics for poster quality.",
                "- The ablation confirms that reflection improves poster generation and evaluation.",
            ]
        )


class MethodGroundedOverlongGenerator:
    def generate(self, prompt: str) -> str:
        return "\n".join(
            [
                "- The system processes research papers automatically through a structured pipeline.",
                "- The parser extracts text and visual assets from the source paper.",
                "- The layout module allocates space before the exporter creates PowerPoint output.",
                "- The system processes research papers automatically through a structured pipeline.",
                "- The parser extracts text and visual assets from the source paper.",
                "- The layout module allocates space before the exporter creates PowerPoint output.",
            ]
        )


def test_content_fitting_rejects_overlong_llm_expansion():
    panel = _panel("Method", "- The system processes papers.", w=0.34, h=0.18)
    source = (
        "The system processes research papers automatically. "
        "The parser extracts text and visual assets. "
        "The layout module allocates space. "
        "The exporter creates PowerPoint output."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig(), OverlongGenerator())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Method": source})

    assert stage.last_decisions[0]["method"] == "llm_trimmed"
    assert stage.last_decisions[0]["words_after"] <= stage.last_decisions[0]["capacity_words"]
    assert result.panels[0].body.count("\n") < 7


def test_content_fitting_strict_llm_trims_overlong_output_without_rule_fallback():
    panel = _panel("Method", "- The system processes papers.", w=0.34, h=0.18)
    source = (
        "The system processes research papers automatically. "
        "The parser extracts text and visual assets. "
        "The layout module allocates space. "
        "The exporter creates PowerPoint output."
    )
    stage = ContentFittingStage(
        ContentFittingConfig(require_llm=True),
        LayoutConfig(),
        MethodGroundedOverlongGenerator(),
    )

    stage.fit(PosterLayout("Demo", [panel]), {"Method": source})

    assert stage.last_decisions[0]["method"] == "llm_trimmed"
    assert stage.last_decisions[0]["method"] != "rule"
    assert stage.last_decisions[0]["words_after"] <= stage.last_decisions[0]["capacity_words"]


def test_content_fitting_strict_llm_accepts_trimmed_text_even_when_expand_cannot_grow():
    panel = _panel("Results", "- " + " ".join(["result"] * 50), w=0.34, h=0.18)
    source = (
        "The results compare poster quality across multiple models. "
        "The evaluation includes human preference and automatic metrics. "
        "The ablation confirms that reflection improves poster generation."
    )
    stage = ContentFittingStage(
        ContentFittingConfig(require_llm=True),
        LayoutConfig(),
        GroundedOverlongGenerator(),
    )

    stage.fit(
        PosterLayout("Demo", [panel]),
        {"Results": source},
        forced_actions={"Results": "expand"},
    )

    assert stage.last_decisions[0]["method"] == "llm_trimmed"
    assert stage.last_decisions[0]["method"] != "rule"
    assert stage.last_decisions[0]["words_after"] <= stage.last_decisions[0]["capacity_words"]


def test_content_fitting_rejects_unsupported_llm_numbers():
    panel = _panel("Results", "- The experiment improves performance.", w=1.0, h=0.5)
    source = (
        "The experiment improves performance over the baseline. "
        "Human reviewers prefer the generated poster. "
        "The evaluation measures readability and visual quality."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig(), UngroundedGenerator())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Results": source})

    assert "99" not in result.panels[0].body
    assert stage.last_decisions[0]["method"] == "rule"


def test_content_fitting_filters_incomplete_llm_bullets():
    panel = _panel("Method", "- The system processes papers.", w=1.0, h=0.5)
    source = (
        "The orchestrate agent assembles HTML posters with modular CSS. "
        "The checker modules refine text coherence and layout aesthetics."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig(), IncompleteSentenceGenerator())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Method": source})

    assert "posters and." not in result.panels[0].body
    assert "checker modules" in result.panels[0].body.lower()


def test_content_fitting_filters_dangling_llm_phrases():
    panel = _panel("Method", "- The system processes papers.", w=1.0, h=0.5)
    source = (
        "We propose P2P, the first flexible framework for generating academic posters. "
        "Poster Checker evaluates layout aesthetics and triggers iterative refinement. "
        "The framework creates editable academic posters from research papers."
    )
    stage = ContentFittingStage(ContentFittingConfig(), LayoutConfig(), DanglingPhraseGenerator())

    result = stage.fit(PosterLayout("Demo", [panel]), {"Method": source})

    assert "for generating." not in result.panels[0].body
    assert "triggering iterative." not in result.panels[0].body
    assert "proven for." not in result.panels[0].body
    assert "editable academic posters" in result.panels[0].body
