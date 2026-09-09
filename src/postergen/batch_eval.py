from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from .config import LLMConfig, PipelineConfig
from .evaluation import evaluate_report_dir, write_evaluation
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
    parser = argparse.ArgumentParser(description="Run automatic module evaluation for generated poster reports.")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=None,
        help="Directory containing per-paper report folders.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Optional directory of PDFs or Paper2Poster-style sample folders.",
    )
    parser.add_argument(
        "--paper-list",
        type=Path,
        default=None,
        help="Optional text file containing one PDF path per line for a fixed evaluation set.",
    )
    parser.add_argument("--run-pipeline", action="store_true", help="Run poster generation before evaluation.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/evaluation_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/evaluation_summary.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of papers to process. 0 means all.")
    parser.add_argument("--llm-provider", default="rule", choices=("rule", "openai", "deepseek", "openrouter", "gemini"))
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key-env", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "Optional suffix for keeping outputs separate, e.g. gpt4o_mini writes "
            "to outputs/batch_posters/<mode>_gpt4o_mini and outputs/assets_gpt4o_mini."
        ),
    )
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default="proposed",
        help="Baseline, ablation, or proposed-system mode to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_suffix = _slug(args.run_name, max_length=40) if args.run_name else ""
    output_mode = f"{args.experiment_mode}_{run_suffix}" if run_suffix else args.experiment_mode
    reports_root = args.reports_root or (
        Path("outputs") / f"assets_{run_suffix}" if run_suffix else Path("outputs/assets")
    )
    processed_report_dirs: list[Path] = []
    if args.run_pipeline:
        if args.paper_list is not None:
            pdfs = read_paper_list(args.paper_list)
            root = args.paper_list.parent
        elif args.paper_dir is not None:
            pdfs = discover_pdfs(args.paper_dir)
            root = args.paper_dir
        else:
            raise SystemExit("--paper-dir or --paper-list is required when --run-pipeline is used.")
        if args.limit:
            pdfs = pdfs[: args.limit]
        for pdf_path in pdfs:
            run_pdf = prepare_batch_input(pdf_path, root, args.experiment_mode)
            output = Path("outputs") / "batch_posters" / output_mode / f"{run_pdf.stem}.pptx"
            config = PipelineConfig(
                input_pdf=run_pdf,
                output_pptx=output,
                report_root=reports_root,
                experiment_mode=args.experiment_mode,
                llm=LLMConfig(
                    provider=args.llm_provider,
                    model=args.llm_model,
                    api_key_env=args.llm_api_key_env,
                    base_url=args.llm_base_url,
                ),
            )
            PosterGenerationPipeline(config).run()
            processed_report_dirs.append(reports_root / run_pdf.stem)

    if processed_report_dirs:
        evaluations = [write_evaluation(path) for path in processed_report_dirs if path.is_dir()]
    else:
        evaluations = evaluate_all(reports_root)
    if args.limit and not processed_report_dirs:
        evaluations = evaluations[: args.limit]
    write_summary(evaluations, args.output_json, args.output_csv)
    print(f"Evaluated {len(evaluations)} papers.")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


def discover_pdfs(root: Path) -> list[Path]:
    direct = sorted(root.glob("*.pdf"))
    nested = sorted(path for path in root.glob("**/paper.pdf") if path.is_file())
    nested += sorted(path for path in root.glob("**/*.pdf") if path.is_file() and path.name != "poster.pdf")
    seen: set[Path] = set()
    result = []
    for path in [*direct, *nested]:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def read_paper_list(path: Path) -> list[Path]:
    pdfs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pdf_path = Path(line).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = path.parent / pdf_path
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Paper listed in {path} does not exist: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"Paper listed in {path} is empty and cannot be parsed: {pdf_path}")
        pdfs.append(pdf_path)
    return pdfs


def prepare_batch_input(pdf_path: Path, root: Path, experiment_mode: str = "proposed") -> Path:
    """Create a stable unique input path so generic paper.pdf files do not overwrite reports."""
    stem = _batch_stem(pdf_path, root)
    if experiment_mode != "proposed":
        stem = f"{stem}_{_slug(experiment_mode, max_length=40)}"
    if stem == pdf_path.stem:
        return pdf_path

    input_dir = Path("outputs") / "batch_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / f"{stem}.pdf"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(pdf_path.resolve())
    except OSError:
        shutil.copy2(pdf_path, target)
    return target


def _batch_stem(pdf_path: Path, root: Path) -> str:
    try:
        relative = pdf_path.relative_to(root)
    except ValueError:
        relative = pdf_path.name
    if pdf_path.stem.lower() != "paper":
        return _slug(pdf_path.stem)
    parent_parts = relative.parts[:-1] if isinstance(relative, Path) else pdf_path.parent.parts[-1:]
    parent_name = parent_parts[-1] if parent_parts else pdf_path.parent.name
    return _slug(parent_name or pdf_path.stem)


def _slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return (slug or "paper")[:max_length]


def evaluate_all(reports_root: Path) -> list[dict]:
    if not reports_root.is_dir():
        return []
    evaluations = []
    for report_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        if not (report_dir / "parser_quality.json").is_file():
            continue
        evaluations.append(write_evaluation(report_dir))
    return evaluations


def write_summary(evaluations: list[dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_evaluations(evaluations)
    output_json.write_text(
        json.dumps({"paper_count": len(evaluations), "aggregate": aggregate, "papers": evaluations}, indent=2),
        encoding="utf-8",
    )
    rows = [_flatten_evaluation(item) for item in evaluations]
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_evaluations(evaluations: list[dict]) -> dict:
    if not evaluations:
        return {}
    rows = [_flatten_evaluation(item) for item in evaluations]
    numeric_keys = [
        key for key in rows[0] if all(isinstance(row.get(key), int | float | bool) for row in rows)
    ]
    aggregate = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        aggregate[f"avg_{key}"] = round(sum(values) / len(values), 3)
    return aggregate


def _flatten_evaluation(evaluation: dict) -> dict:
    row = {
        "paper_id": evaluation.get("paper_id", ""),
        "source_pdf": evaluation.get("source_pdf", ""),
        "experiment_mode": evaluation.get("experiment_mode", ""),
    }
    for module_name in ("pdf_parsing", "section_planning", "asset_planning", "layout_refinement", "overall"):
        for key, value in (evaluation.get(module_name, {}) or {}).items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    row[f"{module_name}.{key}.{child_key}"] = child_value
            elif isinstance(value, list):
                row[f"{module_name}.{key}"] = "|".join(str(item) for item in value)
            else:
                row[f"{module_name}.{key}"] = value
    return row


if __name__ == "__main__":
    main()
