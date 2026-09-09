from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import PaperDocument
from .stages.parser import PaperParser


@dataclass(slots=True)
class DocumentValidationResult:
    file: str
    title_correct: bool
    section_precision: float
    section_recall: float
    section_f1: float
    section_level_accuracy: float
    required_body_pass_rate: float
    figure_recall: float
    table_recall: float
    caption_completeness: float


def validate_document(document: PaperDocument, expected: dict) -> DocumentValidationResult:
    expected_sections = {
        _normalize(item["title"]): item
        for item in expected.get("sections", [])
    }
    predicted_sections = {_normalize(section.title): section for section in document.sections}
    matched_sections = expected_sections.keys() & predicted_sections.keys()

    section_precision = _ratio(len(matched_sections), len(predicted_sections))
    section_recall = _ratio(len(matched_sections), len(expected_sections))
    section_f1 = _f1(section_precision, section_recall)
    level_matches = sum(
        predicted_sections[key].level == int(expected_sections[key].get("level", 1))
        for key in matched_sections
    )
    section_level_accuracy = _ratio(level_matches, len(matched_sections))

    required_bodies = [
        key for key, item in expected_sections.items() if item.get("body_required", True)
    ]
    body_passes = sum(
        key in predicted_sections
        and len(predicted_sections[key].text.split()) >= int(expected_sections[key].get("min_body_words", 10))
        for key in required_bodies
    )

    predicted_figure_ids = {_normalize(figure.figure_id) for figure in document.figures}
    expected_figure_ids = {_normalize(item) for item in expected.get("figure_ids", [])}
    predicted_table_ids = {_normalize(table.table_id) for table in document.tables}
    expected_table_ids = {_normalize(item) for item in expected.get("table_ids", [])}
    captions = [figure.caption for figure in document.figures] + [table.caption for table in document.tables]

    return DocumentValidationResult(
        file=expected["file"],
        title_correct=_normalize(document.title) == _normalize(expected["title"]),
        section_precision=section_precision,
        section_recall=section_recall,
        section_f1=section_f1,
        section_level_accuracy=section_level_accuracy,
        required_body_pass_rate=_ratio(body_passes, len(required_bodies)),
        figure_recall=_set_recall(predicted_figure_ids, expected_figure_ids),
        table_recall=_set_recall(predicted_table_ids, expected_table_ids),
        caption_completeness=_ratio(sum(bool(caption.strip()) for caption in captions), len(captions)),
    )


def run_validation(gold_path: Path, project_root: Path | None = None) -> list[DocumentValidationResult]:
    project_root = project_root or Path.cwd()
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    parser = PaperParser()
    results = []
    for expected in gold["documents"]:
        pdf_path = project_root / expected["file"]
        results.append(validate_document(parser.parse(pdf_path), expected))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PDF parsing on a small representative paper set.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("validation/parser_gold.json"),
        help="Path to the parser validation checklist",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_validation(args.gold)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        print(
            f"{Path(result.file).name}: "
            f"title={'PASS' if result.title_correct else 'FAIL'}, "
            f"section F1={result.section_f1:.3f}, "
            f"levels={result.section_level_accuracy:.3f}, "
            f"body={result.required_body_pass_rate:.3f}, "
            f"figures={result.figure_recall:.3f}, "
            f"tables={result.table_recall:.3f}, "
            f"captions={result.caption_completeness:.3f}"
        )

    print(
        "Average: "
        f"title={sum(result.title_correct for result in results) / len(results):.3f}, "
        f"section F1={_mean(result.section_f1 for result in results):.3f}, "
        f"body={_mean(result.required_body_pass_rate for result in results):.3f}, "
        f"figures={_mean(result.figure_recall for result in results):.3f}, "
        f"tables={_mean(result.table_recall for result in results):.3f}"
    )


def _normalize(text: str) -> str:
    text = text.lower().replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", "", text)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _set_recall(predicted: set[str], expected: set[str]) -> float:
    return _ratio(len(predicted & expected), len(expected))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
