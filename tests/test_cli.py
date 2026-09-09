from postergen.cli import build_parser


def test_cli_accepts_llm_configuration():
    args = build_parser().parse_args(
        [
            "paper.pdf",
            "--output",
            "poster.pptx",
            "--llm-provider",
            "deepseek",
            "--llm-model",
            "deepseek-v4-flash",
            "--llm-api-key-env",
            "DEEPSEEK_API_KEY",
        ]
    )

    assert args.input_pdf.name == "paper.pdf"
    assert args.output.name == "poster.pptx"
    assert args.llm_provider == "deepseek"
    assert args.llm_model == "deepseek-v4-flash"
    assert args.llm_api_key_env == "DEEPSEEK_API_KEY"


def test_cli_accepts_experiment_mode():
    args = build_parser().parse_args(
        [
            "paper.pdf",
            "--experiment-mode",
            "without_panel_refinement",
        ]
    )

    assert args.experiment_mode == "without_panel_refinement"
