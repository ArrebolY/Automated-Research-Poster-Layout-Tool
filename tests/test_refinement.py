from postergen.models import Panel, PosterLayout
from postergen.stages.refinement import PanelRefinementStage


def _panel(title: str, bullets: list[str]) -> Panel:
    return Panel(
        title=title,
        body="\n".join(f"- {bullet}" for bullet in bullets),
        figures=[],
        importance=1.0,
    )


def test_refinement_keeps_specific_content_and_removes_cross_panel_repetition():
    layout = PosterLayout(
        title="Demo",
        panels=[
            _panel(
                "Introduction",
                [
                    "Manual poster creation is time-consuming.",
                    "P2P uses a multi-agent framework for poster generation.",
                ],
            ),
            _panel(
                "P2P Multi-agent Framework",
                [
                    "P2P uses three specialized agents for poster generation.",
                    "Each agent has a checker for iterative refinement.",
                ],
            ),
            _panel(
                "Conclusion",
                [
                    "P2P uses a multi-agent framework for poster generation.",
                    "The system produces high-quality research posters.",
                ],
            ),
        ],
    )
    stage = PanelRefinementStage()

    refined = stage.refine(layout)

    assert "P2P uses three specialized agents" in refined.panels[1].body
    assert "P2P uses a multi-agent framework" not in refined.panels[0].body
    assert len(refined.panels[2].body.splitlines()) == 2
    assert "high-quality research posters" in refined.panels[2].body
    assert stage.last_removed_bullets


def test_refinement_restores_outcome_before_method_detail_in_conclusion():
    layout = PosterLayout(
        title="Demo",
        panels=[
            _panel(
                "Method",
                [
                    "The framework uses three agents for iterative refinement.",
                    "The pipeline generates academic posters from papers.",
                ],
            ),
            _panel(
                "Results",
                [
                    "The system achieves higher poster quality than the baseline.",
                    "Human evaluation demonstrates improved readability.",
                ],
            ),
            _panel(
                "Conclusion",
                [
                    "The framework uses three agents for iterative refinement.",
                    "The system achieves higher poster quality than the baseline.",
                    "Human evaluation demonstrates improved readability.",
                ],
            ),
        ],
    )

    refined = PanelRefinementStage().refine(layout)

    conclusion = refined.panels[-1].body
    assert len(conclusion.splitlines()) == 2
    assert "higher poster quality" in conclusion
    assert "improved readability" in conclusion
