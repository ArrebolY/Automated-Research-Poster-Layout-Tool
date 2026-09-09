from __future__ import annotations

import re
from dataclasses import replace

from ..models import Panel, PosterLayout


class PanelRefinementStage:
    _GENERIC_TITLE_TERMS = {
        "analysis",
        "conclusion",
        "dataset",
        "evaluation",
        "generation",
        "introduction",
        "method",
        "methodology",
        "paper",
        "poster",
        "result",
        "results",
        "system",
    }

    def __init__(self) -> None:
        self.last_removed_bullets: list[dict[str, str]] = []

    def refine(self, layout: PosterLayout) -> PosterLayout:
        refined_panels = [self._refine_panel(panel) for panel in layout.panels]
        deduplicated = self._deduplicate_panels(refined_panels)
        return PosterLayout(title=layout.title, panels=deduplicated)

    def _refine_panel(self, panel: Panel) -> Panel:
        body = panel.body.strip()
        bullet_lines = [line.strip() for line in body.splitlines() if line.strip()]
        has_visual_asset = bool(panel.figures or panel.tables)

        if bullet_lines and all(line.startswith("- ") for line in bullet_lines):
            max_bullets = 3 if has_visual_asset else 5
            max_words = 18 if has_visual_asset else 24
            body = "\n".join(self._trim_bullet(line, max_words) for line in bullet_lines[:max_bullets])
        else:
            words = body.split()
            max_words = 38 if has_visual_asset else 70
            if len(words) > max_words:
                body = self._trim_text(body, max_words)
        return replace(panel, body=body)

    def _trim_bullet(self, line: str, max_words: int) -> str:
        words = line.split()
        if len(words) <= max_words:
            return line
        return self._trim_text(line, max_words)

    def _trim_text(self, text: str, max_words: int) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        prefix = "- " if clean.startswith("- ") else ""
        clean = clean.removeprefix("- ").strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", clean)
            if sentence.strip()
        ]
        result: list[str] = []
        for sentence in sentences:
            candidate = " ".join([*result, sentence]).strip()
            if len(candidate.split()) > max_words:
                break
            result.append(sentence)
        if result:
            return prefix + " ".join(result)
        words = clean.split()
        return prefix + " ".join(words[:max_words]).rstrip(" .,;:") + "."

    def _deduplicate_panels(self, panels: list[Panel]) -> list[Panel]:
        self.last_removed_bullets = []
        if not panels:
            return panels

        specialized_title_terms = {
            term
            for panel in panels
            if not self._is_context_or_conclusion(panel.title)
            for term in self._terms(panel.title)
            if term not in self._GENERIC_TITLE_TERMS
        }
        processing_order = sorted(
            range(len(panels)),
            key=lambda index: (self._is_context_or_conclusion(panels[index].title), index),
        )
        seen_terms: list[set[str]] = []
        updated = list(panels)
        for index in processing_order:
            panel = panels[index]
            lines = [line.strip() for line in panel.body.splitlines() if line.strip()]
            if not lines or not all(line.startswith("- ") for line in lines):
                continue

            generic_panel = self._is_context_or_conclusion(panel.title)
            kept: list[str] = []
            rejected: list[tuple[float, str]] = []
            for line in lines:
                terms = self._terms(line)
                similarity = max((self._overlap_ratio(terms, prior) for prior in seen_terms), default=0.0)
                repeats_specialized_topic = generic_panel and bool(terms & specialized_title_terms)
                if similarity >= 0.62 or repeats_specialized_topic:
                    rejected.append((similarity, line))
                    self.last_removed_bullets.append({"panel": panel.title, "bullet": line})
                else:
                    kept.append(line)
                    if terms:
                        seen_terms.append(terms)

            minimum = min(2, len(lines)) if self._is_conclusion(panel.title) or not generic_panel else 1
            for _similarity, line in sorted(
                rejected,
                key=lambda item: (-self._retention_priority(panel.title, item[1]), item[0]),
            ):
                if len(kept) >= minimum:
                    break
                kept.append(line)
                terms = self._terms(line)
                if terms:
                    seen_terms.append(terms)
            updated[index] = replace(panel, body="\n".join(kept))
        return updated

    def _retention_priority(self, title: str, line: str) -> int:
        """Prefer role-specific information when duplicate bullets must be restored."""
        lowered_title = title.lower()
        lowered_line = line.lower()
        if "conclusion" in lowered_title:
            outcome_terms = (
                "achieve",
                "contribution",
                "demonstrate",
                "effective",
                "improve",
                "outperform",
                "performance",
                "quality",
                "result",
                "significant",
            )
            method_terms = ("agent", "architecture", "framework", "method", "pipeline", "uses")
            return 2 * sum(term in lowered_line for term in outcome_terms) - sum(
                term in lowered_line for term in method_terms
            )
        if any(term in lowered_title for term in ("introduction", "background", "motivation")):
            context_terms = ("challenge", "limitation", "motivation", "problem", "time-consuming")
            return sum(term in lowered_line for term in context_terms)
        return 0

    def _is_context_or_conclusion(self, title: str) -> bool:
        lowered = title.lower()
        return any(term in lowered for term in ("introduction", "background", "motivation", "conclusion"))

    def _is_conclusion(self, title: str) -> bool:
        return "conclusion" in title.lower()

    def _terms(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
        stopwords = {"and", "are", "for", "from", "the", "that", "this", "uses", "with"}
        return {word for word in words if word not in stopwords}

    def _overlap_ratio(self, first: set[str], second: set[str]) -> float:
        if not first or not second:
            return 0.0
        return len(first & second) / min(len(first), len(second))
