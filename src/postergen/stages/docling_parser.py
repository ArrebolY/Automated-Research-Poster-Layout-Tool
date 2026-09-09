from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

from ..models import Figure, PaperDocument, Section, Table
from ..parser_quality import write_parser_quality_report
from .parser import PaperParser


class DoclingPaperParser:
    """Parse with Docling and fall back to the legacy PyMuPDF parser."""

    _STOP_HEADINGS = {"acknowledgements", "acknowledgments", "appendix", "bibliography", "references"}
    _CATEGORY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Introduction", ("introduction", "background")),
        ("Related Work", ("related work", "literature review")),
        ("Method", ("method", "methods", "methodology", "approach", "framework", "system")),
        ("Dataset", ("dataset", "data curation", "benchmark construction")),
        ("Evaluation", ("evaluation", "metrics", "benchmark")),
        ("Results", ("result", "results", "experiment", "experiments", "ablation")),
        ("Discussion", ("discussion",)),
        ("Conclusion", ("conclusion", "conclusions", "future work")),
    )

    def __init__(
        self,
        asset_dir: Path | None = None,
        fallback: PaperParser | None = None,
        images_scale: float = 2.0,
    ) -> None:
        self.asset_dir = asset_dir or Path("outputs/assets")
        self.fallback = fallback or PaperParser(asset_dir=self.asset_dir)
        self.images_scale = images_scale
        self.last_backend = ""
        self.last_fallback_reason = ""
        self.last_extraction_warnings: list[str] = []

    def parse(self, pdf_path: Path) -> PaperDocument:
        self.last_extraction_warnings = []
        try:
            paper = self._parse_with_docling(pdf_path)
            self.last_backend = "docling"
            self.last_fallback_reason = ""
        except Exception as exc:
            self.last_backend = "legacy"
            self.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"Docling parsing failed for {pdf_path.name}; using the legacy parser. "
                f"Reason: {self.last_fallback_reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            paper = self.fallback.parse(pdf_path)

        report_path = self.asset_dir / pdf_path.stem / "parser_quality.json"
        write_parser_quality_report(
            paper,
            report_path,
            backend=self.last_backend,
            fallback_reason=self.last_fallback_reason,
            extraction_warnings=self.last_extraction_warnings,
        )
        return paper

    def _parse_with_docling(self, pdf_path: Path) -> PaperDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem, TableItem

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = self.images_scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(pdf_path)
        return self._paper_from_docling_document(
            pdf_path,
            result.document,
            picture_type=PictureItem,
            table_type=TableItem,
        )

    def _paper_from_docling_document(
        self,
        pdf_path: Path,
        document: Any,
        picture_type: type,
        table_type: type,
    ) -> PaperDocument:
        output_dir = self.asset_dir / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("figure_*.png", "table_*.png", "uncaptioned_*.png"):
            for stale_path in output_dir.glob(pattern):
                stale_path.unlink()

        markdown = document.export_to_markdown()
        (output_dir / "paper.md").write_text(markdown, encoding="utf-8")
        title, abstract, sections = self._parse_markdown(markdown, pdf_path.stem)
        figures, tables = self._extract_assets(document, output_dir, picture_type, table_type)

        return PaperDocument(
            title=title,
            source_path=pdf_path,
            abstract=abstract,
            sections=sections,
            figures=figures,
            tables=tables,
        )

    def _parse_markdown(self, markdown: str, fallback_title: str) -> tuple[str, str, list[Section]]:
        lines = markdown.splitlines()
        headings: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
            if match:
                headings.append((index, len(match.group(1)), self._clean_heading(match.group(2))))

        abstract_heading_position = next(
            (
                position
                for position, heading in enumerate(headings)
                if self._normalized_heading(heading[2]) in {"abstract", "summary"}
            ),
            len(headings),
        )
        title_heading = next(
            (
                heading
                for heading in headings
                if heading[1] == 1 and self._normalized_heading(heading[2]) not in {"abstract", "summary"}
            ),
            None,
        )
        if title_heading is None:
            title_heading = next(
                (
                    heading
                    for heading in headings[:abstract_heading_position]
                    if len(re.findall(r"[A-Za-z]", heading[2])) >= 8
                ),
                None,
            )
        title = title_heading[2] if title_heading else fallback_title.replace("_", " ").replace("-", " ")
        abstract = self._extract_markdown_abstract(lines, headings)

        body_start = next(
            (
                index
                for index, heading in enumerate(headings)
                if self._infer_category(heading[2]) == "Introduction"
            ),
            0,
        )
        section_base_level = headings[body_start][1] if headings else 1
        sections: list[Section] = []
        for position in range(body_start, len(headings)):
            line_index, markdown_level, raw_title = headings[position]
            normalized = self._normalized_heading(raw_title)
            if normalized in self._STOP_HEADINGS:
                break
            if normalized in {"abstract", "summary"} or (title_heading and line_index == title_heading[0]):
                continue

            next_line_index = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            body = self._clean_markdown_body(lines[line_index + 1 : next_line_index])
            sections.append(
                Section(
                    title=self._strip_section_number(raw_title),
                    text=body,
                    level=self._section_level(raw_title, markdown_level, section_base_level),
                    category=self._infer_category(raw_title),
                )
            )

        if not sections:
            full_text = self._clean_markdown_body(lines)
            sections = [Section(title="Full Paper", text=full_text, category="Full Paper")]
        return self._clean_text(title), abstract, sections

    def _extract_markdown_abstract(
        self,
        lines: list[str],
        headings: list[tuple[int, int, str]],
    ) -> str:
        for position, (line_index, _level, title) in enumerate(headings):
            if self._normalized_heading(title) not in {"abstract", "summary"}:
                continue
            next_line_index = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            return self._clean_markdown_body(lines[line_index + 1 : next_line_index])
        return ""

    def _extract_assets(
        self,
        document: Any,
        output_dir: Path,
        picture_type: type,
        table_type: type,
    ) -> tuple[list[Figure], list[Table]]:
        figures: list[Figure] = []
        tables: list[Table] = []
        used_figure_ids: set[str] = set()
        used_table_ids: set[str] = set()
        uncaptioned_picture_count = 0
        uncaptioned_table_count = 0

        for element, _level in document.iterate_items():
            if isinstance(element, picture_type):
                caption = self._clean_text(element.caption_text(document))
                if not caption:
                    uncaptioned_picture_count += 1
                    path = self._save_element_image(
                        element,
                        document,
                        output_dir / f"uncaptioned_picture_{uncaptioned_picture_count}.png",
                    )
                    self.last_extraction_warnings.append(
                        f"Excluded uncaptioned picture: {path or 'image unavailable'}"
                    )
                    continue
                counter = len(figures) + 1
                figure_id = self._unique_asset_id(
                    self._asset_id_from_caption(caption, "Figure", counter),
                    used_figure_ids,
                )
                path = self._save_element_image(element, document, output_dir / f"figure_{counter}.png")
                figures.append(Figure(figure_id=figure_id, caption=caption, path=path))

            elif isinstance(element, table_type):
                caption = self._clean_text(element.caption_text(document))
                if not caption:
                    uncaptioned_table_count += 1
                    path = self._save_element_image(
                        element,
                        document,
                        output_dir / f"uncaptioned_table_{uncaptioned_table_count}.png",
                    )
                    self.last_extraction_warnings.append(
                        f"Excluded uncaptioned table: {path or 'image unavailable'}"
                    )
                    continue
                counter = len(tables) + 1
                table_id = self._unique_asset_id(
                    self._asset_id_from_caption(caption, "Table", counter),
                    used_table_ids,
                )
                path = self._save_element_image(element, document, output_dir / f"table_{counter}.png")
                tables.append(
                    Table(
                        table_id=table_id,
                        caption=caption,
                        cells=self._table_cells(element),
                        path=path,
                    )
                )
        return figures, tables

    def _save_element_image(self, element: Any, document: Any, path: Path) -> Path | None:
        image = element.get_image(document)
        if image is None:
            return None
        image.save(path, "PNG")
        return path

    def _table_cells(self, element: Any) -> list[list[str]]:
        grid = getattr(getattr(element, "data", None), "grid", None) or []
        return [[self._clean_text(getattr(cell, "text", "")) for cell in row] for row in grid]

    def _asset_id_from_caption(self, caption: str, kind: str, fallback_number: int) -> str:
        label = r"(?:fig(?:ure)?\.?)" if kind == "Figure" else r"table"
        match = re.search(rf"\b{label}\s*(\d+[a-z]?)\b", caption, re.IGNORECASE)
        return f"{kind} {match.group(1)}" if match else f"{kind} {fallback_number}"

    def _unique_asset_id(self, candidate: str, used: set[str]) -> str:
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix = 2
        while f"{candidate}.{suffix}" in used:
            suffix += 1
        unique = f"{candidate}.{suffix}"
        used.add(unique)
        return unique

    def _section_level(
        self,
        title: str,
        markdown_level: int,
        section_base_level: int,
    ) -> int:
        number = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", title)
        if number:
            return number.group(1).count(".") + 1
        return max(1, markdown_level - section_base_level + 1)

    def _infer_category(self, title: str) -> str:
        normalized = self._normalized_heading(self._strip_section_number(title))
        for category, aliases in self._CATEGORY_ALIASES:
            if normalized in aliases or any(alias in normalized for alias in aliases if len(alias) >= 6):
                return category
        return "Other"

    def _strip_section_number(self, title: str) -> str:
        return re.sub(r"^(?:\d+(?:\.\d+)*|[A-Z])\.?\s+", "", title).strip(" .:-")

    def _normalized_heading(self, title: str) -> str:
        return re.sub(r"\s+", " ", self._strip_section_number(title).lower()).strip(" .:-")

    def _clean_heading(self, text: str) -> str:
        text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
        text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
        return text.strip(" *_`")

    def _clean_markdown_body(self, lines: list[str]) -> str:
        content = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("<!--") or stripped.startswith("![]("):
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                continue
            content.append(stripped)
        return self._clean_text(" ".join(content))

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"-\s+", "", text)
        return re.sub(r"\s+", " ", text).strip()
