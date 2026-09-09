from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import PaperDocument


@dataclass(slots=True)
class ParserQualityReport:
    backend: str
    fallback_reason: str
    title: str
    title_present: bool
    abstract_length: int
    abstract_present: bool
    section_count: int
    suspicious_section_titles: list[str]
    figure_count: int
    figure_paths: list[str]
    figures_without_captions: list[str]
    table_count: int
    table_paths: list[str]
    tables_without_captions: list[str]
    suspicious_captions: list[str]
    extraction_warnings: list[str]
    missing_files: list[str]


def build_parser_quality_report(
    paper: PaperDocument,
    backend: str,
    fallback_reason: str = "",
    extraction_warnings: list[str] | None = None,
) -> ParserQualityReport:
    figure_paths = [str(figure.path) for figure in paper.figures if figure.path is not None]
    table_paths = [str(table.path) for table in paper.tables if table.path is not None]
    missing_files = []

    for figure in paper.figures:
        if figure.path is None:
            missing_files.append(f"{figure.figure_id}: no image path")
        elif not figure.path.is_file():
            missing_files.append(str(figure.path))
    for table in paper.tables:
        if table.path is None:
            missing_files.append(f"{table.table_id}: no image path")
        elif not table.path.is_file():
            missing_files.append(str(table.path))

    return ParserQualityReport(
        backend=backend,
        fallback_reason=fallback_reason,
        title=paper.title,
        title_present=bool(paper.title.strip()),
        abstract_length=len(paper.abstract.split()),
        abstract_present=bool(paper.abstract.strip()),
        section_count=len(paper.sections),
        suspicious_section_titles=[
            section.title for section in paper.sections if _is_suspicious_title(section.title)
        ],
        figure_count=len(paper.figures),
        figure_paths=figure_paths,
        figures_without_captions=[figure.figure_id for figure in paper.figures if not figure.caption.strip()],
        table_count=len(paper.tables),
        table_paths=table_paths,
        tables_without_captions=[table.table_id for table in paper.tables if not table.caption.strip()],
        suspicious_captions=[
            *[
                f"{figure.figure_id}: {figure.caption}"
                for figure in paper.figures
                if re.match(r"^table\s*\d+", figure.caption.strip(), re.IGNORECASE)
            ],
            *[
                f"{table.table_id}: {table.caption}"
                for table in paper.tables
                if re.match(r"^fig(?:ure)?\.?\s*\d+", table.caption.strip(), re.IGNORECASE)
            ],
        ],
        extraction_warnings=list(extraction_warnings or []),
        missing_files=missing_files,
    )


def write_parser_quality_report(
    paper: PaperDocument,
    output_path: Path,
    backend: str,
    fallback_reason: str = "",
    extraction_warnings: list[str] | None = None,
) -> ParserQualityReport:
    report = build_parser_quality_report(
        paper,
        backend=backend,
        fallback_reason=fallback_reason,
        extraction_warnings=extraction_warnings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def _is_suspicious_title(title: str) -> bool:
    normalized = title.strip()
    if not normalized or len(normalized) > 120:
        return True
    if re.match(r"^(?:fig(?:ure)?|table)\s*\d+", normalized, re.IGNORECASE):
        return True
    if "..." in normalized or ". . ." in normalized:
        return True
    alphanumeric = re.sub(r"[^A-Za-z0-9]", "", normalized)
    return bool(alphanumeric) and sum(character.isdigit() for character in alphanumeric) / len(alphanumeric) > 0.5
