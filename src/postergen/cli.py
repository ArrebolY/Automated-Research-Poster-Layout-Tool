from __future__ import annotations

import argparse
from pathlib import Path

from .config import LLMConfig, PipelineConfig
from .pipeline import PosterGenerationPipeline

EXPERIMENT_MODES = (
    "proposed",
    "direct_llm",
    "template_based",
    "without_semantic_figure_matching",
    "without_llm_figure_description",
    "without_panel_refinement",
    "template_layout_only",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a poster PPTX from a research paper PDF.")
    parser.add_argument("input_pdf", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/poster.pptx"),
        help="Output PPTX path. In direct_llm mode, use a .html path to save the direct HTML poster.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("rule", "openai", "deepseek", "openrouter", "gemini"),
        default="rule",
        help="Text model provider for section summarization. Default: rule",
    )
    parser.add_argument("--llm-model", default="", help="Model name for the selected provider")
    parser.add_argument("--llm-api-key-env", default="", help="Environment variable containing the provider API key")
    parser.add_argument("--llm-base-url", default="", help="Override base URL for OpenAI-compatible providers")
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default="proposed",
        help="Baseline, ablation, or proposed-system mode to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        input_pdf=args.input_pdf,
        output_pptx=args.output,
        experiment_mode=args.experiment_mode,
        llm=LLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            base_url=args.llm_base_url,
        ),
    )
    layout = PosterGenerationPipeline(config).run()
    print(f"Generated poster with {len(layout.panels)} panels at {args.output}")


if __name__ == "__main__":
    main()
