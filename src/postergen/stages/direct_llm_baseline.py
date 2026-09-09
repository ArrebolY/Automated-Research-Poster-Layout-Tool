from __future__ import annotations

import html
import json
import re
from pathlib import Path

from ..llm import TextGenerator
from ..models import PaperDocument, Panel, PosterLayout


class DirectLLMBaselineStage:
    """Generate a complete poster page directly as HTML/CSS."""

    def __init__(self, text_generator: TextGenerator, max_sections: int = 5, max_chars: int = 12000) -> None:
        self.text_generator = text_generator
        self.max_sections = max_sections
        self.max_chars = max_chars
        self.last_response = ""
        self.last_html = ""

    def generate_html(self, paper: PaperDocument) -> str:
        response = self.text_generator.generate(self._html_prompt(paper))
        self.last_response = response
        document = self._extract_html(response)
        self.last_html = document
        return document

    def save_html(self, paper: PaperDocument, output_path: Path | None = None) -> Path:
        html_path = output_path if output_path and output_path.suffix.lower() == ".html" else self.default_html_path(paper)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_html = self._rewrite_local_image_sources(self.last_html, html_path)
        html_path.write_text(self.last_html, encoding="utf-8")
        return html_path

    def default_html_path(self, paper: PaperDocument) -> Path:
        return Path("outputs") / "direct_llm" / paper.source_path.stem / "poster.html"

    def proxy_layout(self, paper: PaperDocument) -> PosterLayout:
        panels = self._panels_from_html(self.last_html)
        if not panels:
            panels = [
                Panel(
                    title="Direct LLM Poster",
                    body="The direct LLM baseline generated a complete HTML poster without modular planning.",
                    figures=[],
                    tables=[],
                    importance=1.0,
                )
            ]
        return PosterLayout(title=paper.title, panels=panels[: self.max_sections])

    def write_report(self, paper: PaperDocument, layout: PosterLayout, report_dir: Path, html_path: Path) -> None:
        target = report_dir / paper.source_path.stem
        target.mkdir(parents=True, exist_ok=True)
        sections = [
            {
                "title": panel.title,
                "effective_category": self._category(panel.title),
                "source_excerpt": "",
                "summary_bullets": [line for line in panel.body.splitlines() if line.strip()],
            }
            for panel in layout.panels
        ]
        (target / "direct_llm_poster.json").write_text(
            json.dumps(
                {
                    "source_pdf": str(paper.source_path),
                    "html_path": str(html_path),
                    "method": "direct_llm_html_css_poster",
                    "response": self.last_response,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "section_selection.json").write_text(
            json.dumps(
                {
                    "source_pdf": str(paper.source_path),
                    "selection_method": "direct_llm_html_generation",
                    "selection_response": self.last_response,
                    "parsed_section_count": len(paper.sections),
                    "candidate_section_count": len(paper.sections),
                    "selected_section_count": len(layout.panels),
                    "selected_sections": sections,
                    "summary_errors": [],
                    "summary_grounding_warnings": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "asset_selection.json").write_text(
            json.dumps(
                {
                    "source_pdf": str(paper.source_path),
                    "selection_method": "direct_llm_html_generation",
                    "candidate_count": len(paper.figures) + len(paper.tables),
                    "selected_figure_count": 0,
                    "selected_table_count": 0,
                    "assets": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "asset_matching.json").write_text(
            json.dumps(
                {
                    "source_pdf": str(paper.source_path),
                    "matching_method": "direct_llm_html_generation_no_modular_matching",
                    "assignments": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _html_prompt(self, paper: PaperDocument) -> str:
        sections = []
        for section in paper.sections:
            excerpt = re.sub(r"\s+", " ", section.text).strip()[:1000]
            sections.append(f"## {section.title}\n{excerpt}")
        assets = [
            {
                "type": "figure",
                "id": figure.figure_id,
                "caption": figure.caption,
                "path": str(figure.path) if figure.path else "",
            }
            for figure in paper.figures[:8]
        ] + [
            {
                "type": "table",
                "id": table.table_id,
                "caption": table.caption,
                "path": str(table.path) if table.path else "",
            }
            for table in paper.tables[:5]
        ]
        content = (
            f"Title: {paper.title}\n\n"
            f"Abstract:\n{paper.abstract}\n\n"
            f"Available visual assets as JSON:\n{json.dumps(assets, ensure_ascii=True)}\n\n"
            "Paper sections:\n"
            + "\n\n".join(sections)
        )
        content = content[: self.max_chars]
        return (
            "You are a baseline system that directly generates one complete academic research poster as HTML/CSS.\n"
            "Generate the whole poster yourself in a single HTML document. Do not output JSON, Markdown, or explanations.\n"
            "The output format must be a one-page 16:9 landscape HTML/CSS poster.\n"
            "Return a complete HTML document starting with <!doctype html>.\n\n"
            f"{content}"
        )

    def _extract_html(self, response: str) -> str:
        fenced = re.search(r"```html\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if fenced:
            response = fenced.group(1)
        start = response.lower().find("<!doctype html")
        if start == -1:
            start = response.lower().find("<html")
        if start == -1:
            body_start = response.lower().find("<body")
            if body_start != -1:
                document = response[body_start:].strip()
            elif "<section" in response.lower() or "<div" in response.lower():
                document = response.strip()
            else:
                raise ValueError("Direct LLM response did not contain an HTML document.")
        else:
            document = response[start:].strip()
        end = document.lower().rfind("</html>")
        if end != -1:
            document = document[: end + len("</html>")]
        return self._ensure_complete_html(document)

    def _ensure_complete_html(self, document: str) -> str:
        lowered = document.lower()
        if "<body" not in lowered:
            document = f"<body>\n{document}\n</body>"
            lowered = document.lower()
        if "<style" not in lowered:
            style = self._default_style()
            if "<head" in lowered:
                document = re.sub(r"</head>", f"{style}</head>", document, count=1, flags=re.IGNORECASE)
            else:
                document = f"<head>{style}</head>\n{document}"
            lowered = document.lower()
        if "<html" not in lowered:
            document = f"<!doctype html>\n<html lang=\"en\">\n{document}\n</html>"
        elif "<!doctype html" not in lowered:
            document = "<!doctype html>\n" + document
        return document

    def _rewrite_local_image_sources(self, document: str, html_path: Path) -> str:
        def replace_src(match: re.Match) -> str:
            quote = match.group(1)
            src = html.unescape(match.group(2).strip())
            if self._is_external_src(src):
                return match.group(0)
            resolved = self._resolve_image_src(src, html_path)
            if resolved is None:
                return match.group(0)
            return f"src={quote}{resolved.as_uri()}{quote}"

        return re.sub(r"src=(['\"])(.*?)\1", replace_src, document, flags=re.IGNORECASE)

    def _is_external_src(self, src: str) -> bool:
        lowered = src.lower()
        return lowered.startswith(("http://", "https://", "data:", "file:"))

    def _resolve_image_src(self, src: str, html_path: Path) -> Path | None:
        candidate = Path(src).expanduser()
        candidates = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(Path.cwd() / candidate)
            candidates.append(html_path.parent / candidate)
        for path in candidates:
            if path.is_file():
                return path.resolve()
        return None

    def _default_style(self) -> str:
        return """
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: #fff; color: #111; }
  .poster, main { width: 1600px; height: 900px; padding: 42px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; overflow: hidden; }
  h1 { grid-column: 1 / -1; margin: 0 0 8px; text-align: center; font-size: 42px; line-height: 1.1; }
  section { border-top: 6px solid #0b4f75; background: #f8fafc; padding: 14px 16px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,.2); }
  h2, h3 { margin: 0 0 8px; color: #0b4f75; font-size: 22px; line-height: 1.1; }
  p, li { font-size: 16px; line-height: 1.25; margin: 0 0 6px; }
  img { max-width: 100%; max-height: 260px; display: block; margin: 8px auto; object-fit: contain; }
</style>
"""

    def _panels_from_html(self, document: str) -> list[Panel]:
        titles = re.findall(r"<h[23][^>]*>(.*?)</h[23]>", document, flags=re.DOTALL | re.IGNORECASE)
        blocks = re.split(r"<h[23][^>]*>.*?</h[23]>", document, flags=re.DOTALL | re.IGNORECASE)[1:]
        panels = []
        for title, block in zip(titles, blocks, strict=False):
            clean_title = self._strip_tags(title)
            body_parts = re.findall(r"<(?:p|li)[^>]*>(.*?)</(?:p|li)>", block, flags=re.DOTALL | re.IGNORECASE)
            body = "\n".join(self._strip_tags(part) for part in body_parts if self._strip_tags(part))
            if clean_title and body:
                panels.append(
                    Panel(
                        title=clean_title,
                        body=body,
                        figures=[],
                        tables=[],
                        importance=max(1.0, len(body.split())),
                    )
                )
        return panels

    def _strip_tags(self, value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value)
        return html.unescape(re.sub(r"\s+", " ", text).strip())

    def _category(self, title: str) -> str:
        lowered = title.lower()
        if "intro" in lowered or "motivation" in lowered:
            return "introduction"
        if any(term in lowered for term in ("method", "model", "approach", "framework")):
            return "method"
        if any(term in lowered for term in ("result", "experiment", "evaluation", "analysis")):
            return "results"
        if "conclusion" in lowered or "discussion" in lowered:
            return "conclusion"
        return "supporting_content"
