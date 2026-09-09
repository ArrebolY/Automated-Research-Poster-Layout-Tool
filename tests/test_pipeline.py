from postergen.pipeline import PosterGenerationPipeline


def test_forced_content_feedback_prioritizes_overflow_before_blank_space():
    pipeline = PosterGenerationPipeline.__new__(PosterGenerationPipeline)
    report = {
        "panels": [
            {
                "title": "Motivation",
                "issues": [
                    {"type": "text_overflow_risk", "action": "compress_text"},
                    {"type": "pure_text_too_spacious", "action": "expand_text_and_decrease_panel_weight"},
                ],
            }
        ]
    }

    actions, reasons = pipeline._forced_content_feedback(report)

    assert actions == {"Motivation": "compress"}
    assert "overflow" in reasons["Motivation"]
