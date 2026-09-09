from __future__ import annotations

import re
from pathlib import Path

from ..models import Figure, PaperDocument, Section, Table


class PaperParser:
    """Parse a research paper PDF into text sections and figure captions."""

    _SECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Introduction", ("introduction", "background")),
        (
            "Method",
            ("method", "methods", "methodology", "approach", "model", "proposed method", "system overview"),
        ),
        ("Evaluation", ("evaluation", "benchmark", "poster evaluation framework")),
        ("Results", ("result", "results", "experiment", "experiments", "experiments and analysis")),
        ("Discussion", ("discussion",)),
        ("Conclusion", ("conclusion", "conclusions", "concluding remarks")),
    )
    _FIGURE_PATTERN = re.compile(
        r"^\s*(?P<label>fig(?:ure)?\.?\s*\d+[a-z]?)\s*[:.\-]\s*(?P<caption>.+?)(?=\n\s*(?:fig(?:ure)?\.?\s*\d+[a-z]?\s*[:.\-]|table\s*\d+\s*[:.\-]|[A-Z][A-Za-z ]{2,40}\n)|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    _TABLE_PATTERN = re.compile(
        r"^\s*(?P<label>table\s*\d+[a-z]?)\s*[:.\-]\s*(?P<caption>.+?)(?=\n\s*(?:fig(?:ure)?\.?\s*\d+[a-z]?\s*[:.\-]|table\s*\d+[a-z]?\s*[:.\-]|[A-Z][A-Za-z ]{2,40}\n)|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    _STOP_HEADINGS = {"references", "bibliography", "acknowledgments", "acknowledgements", "appendix"}
    _MIN_IMAGE_WIDTH = 160
    _MIN_IMAGE_HEIGHT = 120

    def __init__(self, asset_dir: Path | None = None) -> None:
        self.asset_dir = asset_dir #保存图片的路径

    def parse(self, pdf_path: Path) -> PaperDocument:
        try:
            import fitz #PyMuPDF
        except ModuleNotFoundError as exc:
            raise RuntimeError("PDF parsing requires PyMuPDF. Install dependencies with `pip install -e .`.") from exc

        document = fitz.open(pdf_path)
        try:
            text = self._extract_text(document)
            title = self._extract_title(document, text, pdf_path)
            image_paths_by_id = self._extract_figure_region_assets(document, pdf_path, fitz)
        finally:
            document.close()

        sections = self._split_sections(text)
        if not sections and text.strip():
            sections = [Section(title="Full Paper", text=text.strip(), category="Full Paper")]

        paper = PaperDocument(
            title=title,
            source_path=pdf_path,
            abstract=self._extract_abstract(text),
            sections=sections,
            figures=self._extract_figures(text, image_paths_by_id),
            tables=self._extract_tables(text),
        )
        self._attach_in_text_references(paper)
        return paper

    def _extract_text(self, document) -> str:
        pages = [self._clean_page_text(page.get_text("text")) for page in document]
        return "\n\n".join(page for page in pages if page)

    def _extract_title(self, document, text: str, pdf_path: Path) -> str:
        # 1. 先看 PDF metadata title
        # 2. 再从文本前几行找一个像标题的候选
        # 3. 如果文本标题比 metadata 更可信，就用文本标题
        # 4. 否则用 metadata
        # 5. 再不行就用文件名
        metadata_title = (document.metadata or {}).get("title", "").strip()
        text_title = ""
        for line in text.splitlines():
            candidate = line.strip()
            if 8 <= len(candidate) <= 180 and not self._match_section_title(candidate):
                text_title = candidate
                break

        if text_title and (not metadata_title or len(text_title) >= len(metadata_title)):
            return text_title
        if metadata_title and metadata_title.lower() not in {"untitled", "unknown"}:
            return metadata_title
        return pdf_path.stem.replace("-", " ").replace("_", " ").title()

    def _extract_abstract(self, text: str) -> str:
        lines = text.splitlines()
        collecting = False
        abstract_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"(abstract|summary)", stripped, re.IGNORECASE):
                collecting = True
                continue
            if collecting and self._match_section_title(stripped):
                break
            if collecting:
                abstract_lines.append(stripped)

        return self._clean_text(" ".join(abstract_lines))

    def _split_sections(self, text: str) -> list[Section]:#按section切分
        text = self._body_text_for_sections(text)
        sections: list[Section] = []
        current_title: str | None = None
        current_category: str = ""
        current_lines: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            matched_section = self._match_section_title(line)
            if self._is_stop_heading(line):
                if current_title and current_lines:
                    sections.append(
                        Section(
                            title=current_title,
                            text=self._clean_text(" ".join(current_lines)),
                            category=current_category,
                        )
                    )
                current_title = None
                current_category = ""
                current_lines = []
                break

            if matched_section:
                if current_title and current_lines:
                    sections.append(
                        Section(
                            title=current_title,
                            text=self._clean_text(" ".join(current_lines)),
                            category=current_category,
                        )
                    )
                current_category, current_title = matched_section
                current_lines = []
                continue

            if current_title:
                current_lines.append(line)

        if current_title and current_lines:
            sections.append(
                Section(title=current_title, text=self._clean_text(" ".join(current_lines)), category=current_category)
            )

        return self._merge_duplicate_sections(sections)

    def _match_section_title(self, line: str) -> tuple[str, str] | None:
        raw_line = line.strip()
        has_number_prefix = bool(re.match(r"^\d+(?:\.\d+)*\.?\s+", raw_line))
        display_title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", raw_line).strip(" .:-")
        normalized = display_title.lower()
        if "···" in normalized or "..." in normalized:
            return None
        if len(normalized) > 90 or normalized.startswith("section "):
            return None

        for canonical_title, aliases in self._SECTION_ALIASES:
            if normalized in aliases:
                return canonical_title, display_title
            if (has_number_prefix or ":" in display_title) and any(
                alias in normalized for alias in aliases if len(alias) >= 8
            ):
                return canonical_title, display_title
        return None

    def _is_stop_heading(self, line: str) -> bool:
        normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", line.strip().lower()).strip(" .:-")
        return normalized in self._STOP_HEADINGS

    def _body_text_for_sections(self, text: str) -> str:#去掉前言部分，直接从正文开始
        contents_index = text.lower().find("\ncontents\n")
        if contents_index == -1:
            return text

        after_contents = text[contents_index + len("\ncontents\n") :]
        match = re.search(r"(?m)^\s*1\.?\s+Introduction\s*$", after_contents, re.IGNORECASE)
        return after_contents[match.start() :] if match else text

    def _merge_duplicate_sections(self, sections: list[Section]) -> list[Section]:#合并重复的section
        merged: dict[str, list[str]] = {}
        titles: dict[str, str] = {}
        categories: dict[str, str] = {}
        order: list[str] = []

        for section in sections:
            key = section.category or section.title
            if key not in merged:
                merged[key] = []
                titles[key] = section.title
                categories[key] = section.category
                order.append(key)
            merged[key].append(section.text)

        return [
            Section(title=titles[key], text=self._clean_text(" ".join(merged[key])), category=categories[key])
            for key in order
        ]

    def _extract_figures(self, text: str, image_paths_by_id: dict[str, Path]) -> list[Figure]:#用正则提取图表
        figures: list[Figure] = []
        seen: set[str] = set()

        for match in self._FIGURE_PATTERN.finditer(text):
            figure_id = self._normalize_figure_id(match.group("label"))
            caption = self._clean_text(match.group("caption"))
            if not caption or figure_id in seen:
                continue
            seen.add(figure_id)
            image_path = image_paths_by_id.get(figure_id)
            figures.append(Figure(figure_id=figure_id, caption=caption, path=image_path))

        return figures

    def _extract_tables(self, text: str) -> list[Table]:
        tables: list[Table] = []
        seen: set[str] = set()

        for match in self._TABLE_PATTERN.finditer(text):
            table_id = self._normalize_table_id(match.group("label"))
            caption = self._clean_text(match.group("caption"))
            if not caption or table_id in seen:
                continue
            seen.add(table_id)
            tables.append(Table(table_id=table_id, caption=caption))

        return tables

    def _clean_table_cells(self, table: list[list[str | None]]) -> list[list[str]]:
        cleaned_rows = []
        for row in table:
            cleaned_row = [self._clean_text(cell or "") for cell in row]
            if any(cleaned_row):
                cleaned_rows.append(cleaned_row)
        return cleaned_rows if len(cleaned_rows) >= 2 else []

    def _attach_in_text_references(self, paper: PaperDocument) -> None:
        for figure in paper.figures:
            number = re.search(r"\d+[a-z]?", figure.figure_id, re.IGNORECASE)
            if number is None:
                continue
            pattern = re.compile(rf"\b(?:fig\.?|figure)\s*{re.escape(number.group(0))}\b", re.IGNORECASE)
            figure.referenced_in = [
                section.title
                for section in paper.sections
                if pattern.search(self._remove_caption_reference(section.text, "figure", number.group(0)))
            ]

        for table in paper.tables:
            number = re.search(r"\d+[a-z]?", table.table_id, re.IGNORECASE)
            if number is None:
                continue
            pattern = re.compile(rf"\btable\s*{re.escape(number.group(0))}\b", re.IGNORECASE)
            table.referenced_in = [
                section.title
                for section in paper.sections
                if pattern.search(self._remove_caption_reference(section.text, "table", number.group(0)))
            ]

    def _remove_caption_reference(self, text: str, kind: str, number: str) -> str:
        label = r"(?:fig\.?|figure)" if kind == "figure" else "table"
        return re.sub(
            rf"\b{label}\s*{re.escape(number)}\s*[:\-][^.]*\.?",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    def _extract_figure_region_assets(self, document, pdf_path: Path, fitz) -> dict[str, Path]:
        if self.asset_dir is None:
            return {}

        output_dir = self.asset_dir / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths: dict[str, Path] = {}

        for page_index, page in enumerate(document):
            page_text = page.get_text("text")
            labels = self._figure_labels_on_page(page_text)
            for figure_id, label in labels:
                if figure_id in image_paths:
                    continue
                caption_rects = page.search_for(label)
                if not caption_rects:
                    continue

                caption_rect = caption_rects[0]
                page_rect = page.rect
                top = max(page_rect.y0, caption_rect.y0 - page_rect.height * 0.38)
                bottom = max(top + 80, caption_rect.y0 - 8)
                clip = fitz.Rect(page_rect.x0 + 28, top, page_rect.x1 - 28, min(bottom, page_rect.y1))
                if clip.width < self._MIN_IMAGE_WIDTH or clip.height < self._MIN_IMAGE_HEIGHT:
                    continue

                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                image_path = output_dir / f"{figure_id.lower().replace(' ', '_')}.png"
                pixmap.save(image_path)
                self._trim_image_whitespace(image_path)
                image_paths[figure_id] = image_path

        return image_paths

    def _trim_image_whitespace(self, image_path: Path) -> None:
        try:
            from PIL import Image, ImageChops
        except ModuleNotFoundError:
            return

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            background = Image.new("RGB", rgb_image.size, (255, 255, 255))
            diff = ImageChops.difference(rgb_image, background)
            diff = ImageChops.add(diff, diff, 2.0, -18)
            bbox = diff.getbbox()
            if bbox is None:
                return

            left, top, right, bottom = bbox
            pad = 18
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(rgb_image.width, right + pad)
            bottom = min(rgb_image.height, bottom + pad)
            if right - left < self._MIN_IMAGE_WIDTH or bottom - top < self._MIN_IMAGE_HEIGHT:
                return
            rgb_image.crop((left, top, right, bottom)).save(image_path)

    def _figure_labels_on_page(self, page_text: str) -> list[tuple[str, str]]:
        labels: list[tuple[str, str]] = []
        for line in page_text.splitlines():
            match = re.match(r"\s*(Fig(?:ure)?\.?\s*(\d+[a-z]?))\s*[:.\-]", line, re.IGNORECASE)
            if match:
                figure_id = f"Figure {match.group(2)}"
                labels.append((figure_id, match.group(1)))
        return labels

    def _normalize_figure_id(self, label: str) -> str:
        number = re.search(r"\d+[a-z]?", label, re.IGNORECASE)
        return f"Figure {number.group(0)}" if number else label.strip()

    def _normalize_table_id(self, label: str) -> str:
        number = re.search(r"\d+[a-z]?", label, re.IGNORECASE)
        return f"Table {number.group(0)}" if number else label.strip()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_page_text(self, text: str) -> str:
        text = re.sub(r"-\s*\n\s*", "", text)
        cleaned_lines = []
        for line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line or re.fullmatch(r"\d+", line):
                continue
            if ". . ." in line:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
