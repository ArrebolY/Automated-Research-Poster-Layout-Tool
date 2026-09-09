from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

from ..config import ContentFittingConfig, LayoutConfig
from ..llm import TextGenerator
from ..models import Panel, PosterLayout
from ..prompt_loader import load_prompt


class ContentFittingStage:
    """Adjust panel text after layout using an estimated rendered capacity."""

    _STOPWORDS = {
        "about",
        "also",
        "and",
        "are",
        "for",
        "from",
        "have",
        "into",
        "that",
        "the",
        "their",
        "this",
        "using",
        "were",
        "which",
        "with",
    }
    _INCOMPLETE_ENDINGS = {
        "and",
        "as",
        "at",
        "because",
        "by",
        "for",
        "from",
        "generating",
        "in",
        "including",
        "into",
        "iterative",
        "of",
        "or",
        "proven",
        "supporting",
        "than",
        "that",
        "the",
        "through",
        "to",
        "triggering",
        "under",
        "using",
        "via",
        "while",
        "with",
    }

    def __init__(
        self,
        config: ContentFittingConfig,
        layout_config: LayoutConfig,
        text_generator: TextGenerator | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.layout_config = layout_config
        self.text_generator = text_generator
        self.report_dir = report_dir
        self.last_decisions: list[dict] = []

    def fit(
        self,
        layout: PosterLayout,
        source_text_by_title: dict[str, str],
        key_facts_by_title: dict[str, dict[str, list[str]]] | None = None,
        source_path: Path | None = None,
        forced_actions: dict[str, str] | None = None,
        forced_reasons: dict[str, str] | None = None,
    ) -> PosterLayout:
        self.last_decisions = []
        if not self.config.enabled:
            return layout

        fitted: list[Panel] = []
        for panel in layout.panels:
            source_text = source_text_by_title.get(panel.title, "")
            key_facts = (key_facts_by_title or {}).get(panel.title, {})
            capacity = self._estimated_word_capacity(panel)
            before_words = self._word_count(panel.body)
            occupancy = before_words / capacity if capacity else 1.0
            forced_action = (forced_actions or {}).get(panel.title)
            forced_reason = (forced_reasons or {}).get(panel.title, "")
            method = "unchanged"
            error = ""
            body = panel.body
            action = self._resolve_action(panel, occupancy, source_text, forced_action)
            target_words = self._target_words(panel, capacity, action, forced=forced_action is not None)
            iterations = 0

            other_panel_text = "\n".join(other.body for other in layout.panels if other is not panel)
            while action != "keep" and iterations < max(1, self.config.max_iterations):
                iterations += 1
                current_words = self._word_count(body)
                working_panel = replace(panel, body=body)
                try:
                    body, method = self._fit_once(
                        working_panel,
                        source_text,
                        key_facts,
                        other_panel_text,
                        action,
                        target_words,
                        forced_reason if iterations == 1 else "Follow-up layout-aware fitting pass.",
                        current_words,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if self.config.require_llm:
                        raise RuntimeError(
                            f"LLM content fitting failed for panel '{panel.title}' and rule fallback is disabled: {error}"
                        ) from exc
                    warnings.warn(
                        f"LLM content fitting failed for panel '{panel.title}'; using the rule-based fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    method = "rule"
                    body = self._fit_with_rules(
                        body,
                        source_text,
                        other_panel_text,
                        action,
                        target_words,
                    )

                after_iteration_words = self._word_count(body)
                after_occupancy = after_iteration_words / capacity if capacity else 1.0
                next_action = self._resolve_action(
                    replace(panel, body=body),
                    after_occupancy,
                    source_text,
                    None,
                )
                if next_action == "keep" or after_iteration_words == current_words:
                    break
                if action == "compress" and next_action == "expand":
                    break
                action = next_action
                target_words = self._target_words(panel, capacity, action)

            after_words = self._word_count(body)
            fitted.append(replace(panel, body=body))
            self.last_decisions.append(
                {
                    "title": panel.title,
                    "capacity_words": capacity,
                    "words_before": before_words,
                    "words_after": after_words,
                    "occupancy_before": round(occupancy, 3),
                    "occupancy_after": round(after_words / capacity, 3) if capacity else 1.0,
                    "action": action,
                    "forced": forced_action is not None,
                    "forced_reason": forced_reason,
                    "method": method,
                    "iterations": iterations,
                    "priority_facts": key_facts,
                    "error": error,
                }
            )

        result = PosterLayout(title=layout.title, panels=fitted)
        self._write_report(source_path)
        return result

    def _action(self, occupancy: float, source_text: str, panel: Panel) -> str:
        sparse_ratio = 0.7 if self._has_renderable_visual(panel) else self.config.sparse_ratio
        if occupancy < sparse_ratio and source_text.strip():
            return "expand"
        if occupancy > self.config.dense_ratio:
            return "compress"
        return "keep"

    def _resolve_action(
        self,
        panel: Panel,
        occupancy: float,
        source_text: str,
        forced_action: str | None,
    ) -> str:
        natural_action = self._action(occupancy, source_text, panel)
        if forced_action is None:
            if self.config.target_min_ratio <= occupancy <= self.config.target_max_ratio:
                return "keep"
            return natural_action
        if forced_action == "compress" and occupancy < self.config.target_min_ratio and source_text.strip():
            return "expand"
        if forced_action == "expand" and occupancy > self.config.target_max_ratio:
            return "compress"
        return forced_action

    def _target_words(self, panel: Panel, capacity: int, action: str, forced: bool = False) -> int:
        if action == "expand":
            if self._has_renderable_visual(panel):
                ratio = 0.78 if forced else 0.74
            else:
                ratio = 0.82 if forced else self.config.target_ratio
        elif action == "compress":
            ratio = min(self.config.target_ratio, 0.68 if self._has_renderable_visual(panel) else 0.78)
        else:
            ratio = self.config.target_ratio
        return max(self.config.min_bullets * 6, round(capacity * ratio))

    def _estimated_word_capacity(self, panel: Panel) -> int:
        margin = self.layout_config.margin
        panel_width = max(0.6, (panel.w - 2 * margin) * self.layout_config.page_width)
        panel_height = max(0.8, (panel.h - 2 * margin) * self.layout_config.page_height)
        text_width = max(0.25, panel_width - 0.32)
        has_visual = self._has_renderable_visual(panel)
        font_size = self._body_font_size(panel, has_visual)

        if has_visual:
            word_count = self._word_count(panel.body)
            ratio = 0.27 if word_count <= 45 else 0.34 if word_count <= 75 else 0.42
            text_height = max(0.62, min(panel_height * ratio, panel_height * 0.48))
        else:
            text_height = max(0.3, panel_height - 0.2)

        heading_height = (font_size + 3) * 1.3 / 72
        body_height = max(0.15, text_height - heading_height)
        chars_per_line = text_width * 72 / (font_size * 0.55)
        line_count = body_height * 72 / (font_size * 1.25)
        capacity = int(chars_per_line * line_count / 6.3)
        return max(12, min(capacity, 160))

    def _body_font_size(self, panel: Panel, has_visual: bool) -> int:
        if panel.w * panel.h < 0.18:
            return 8
        if has_visual:
            return 9
        return 11

    def _has_renderable_visual(self, panel: Panel) -> bool:
        if any(figure.path is not None for figure in panel.figures):
            return True
        return any(
            table.path is not None and (not table.cells or len(table.cells) <= 20)
            for table in panel.tables
        )

    def _fit_with_llm(
        self,
        panel: Panel,
        source_text: str,
        key_facts: dict[str, list[str]],
        other_panel_text: str,
        action: str,
        target_words: int,
        action_reason: str = "",
    ) -> str:
        prompt = load_prompt(
            "content_fitting.txt",
            ACTION=action,
            ACTION_REASON=action_reason or "Automatic layout-aware text fitting.",
            SECTION_TITLE=panel.title,
            TARGET_WORDS=target_words,
            MIN_BULLETS=self.config.min_bullets,
            MAX_BULLETS=self.config.max_bullets,
            CURRENT_TEXT=panel.body,
            OTHER_PANEL_TEXT=other_panel_text[:2500],
            KEY_FACTS=self._format_key_facts(key_facts),
            SOURCE_TEXT=source_text[: self.config.max_source_chars],
        )
        response = self.text_generator.generate(prompt)
        bullets = self._parse_bullets(response)
        grounded = self._ground_bullets(bullets, source_text)
        if not grounded:
            raise ValueError("The fitted bullets were not grounded in the source section.")
        return "\n".join(grounded[: self.config.max_bullets])

    def _fit_once(
        self,
        panel: Panel,
        source_text: str,
        key_facts: dict[str, list[str]],
        other_panel_text: str,
        action: str,
        target_words: int,
        action_reason: str,
        before_words: int,
    ) -> tuple[str, str]:
        if self.config.require_llm and self.text_generator is None:
            raise RuntimeError(f"Content fitting for panel '{panel.title}' requires an LLM, but no text generator is configured.")
        if self.text_generator is None:
            return (
                self._fit_with_rules(panel.body, source_text, other_panel_text, action, target_words),
                "rule",
            )

        candidate = self._fit_with_llm(
            panel,
            source_text,
            key_facts,
            other_panel_text,
            action,
            target_words,
            action_reason,
        )
        if self._is_effective(action, before_words, candidate, target_words):
            return candidate, "llm"
        trimmed = self._limit_text_to_target(candidate, target_words)
        if self._is_usable_llm_text(trimmed, target_words):
            return trimmed, "llm_trimmed"
        if self.config.require_llm:
            raise ValueError(f"The LLM output was not usable for action '{action}' within the target length.")
        return (
            self._fit_with_rules(panel.body, source_text, other_panel_text, action, target_words),
            "rule",
        )

    def _format_key_facts(self, key_facts: dict[str, list[str]]) -> str:
        if not key_facts:
            return "- None detected."
        labels = {
            "datasets_or_benchmarks": "Datasets or benchmarks",
            "methods_or_components": "Methods or components",
            "metrics_or_results": "Metrics or results",
            "numbers": "Numbers",
            "figures_or_tables": "Figures or tables",
            "claims_or_conclusions": "Claims or conclusions",
        }
        lines: list[str] = []
        for category, values in key_facts.items():
            cleaned = [value.strip() for value in values if value.strip()]
            if not cleaned:
                continue
            lines.append(f"{labels.get(category, category.replace('_', ' ').title())}:")
            lines.extend(f"- {value}" for value in cleaned[:5])
        return "\n".join(lines) or "- None detected."

    def _fit_with_rules(
        self,
        current_text: str,
        source_text: str,
        other_panel_text: str,
        action: str,
        target_words: int,
    ) -> str:
        current = self._parse_bullets(current_text) or [self._to_bullet(current_text)]
        if action == "compress":
            return self._limit_text_to_target("\n".join(self._compress_bullets(current, target_words)), target_words)
        return self._limit_text_to_target(
            "\n".join(self._expand_bullets(current, source_text, other_panel_text, target_words)),
            target_words,
        )

    def _compress_bullets(self, bullets: list[str], target_words: int) -> list[str]:
        minimum = min(self.config.min_bullets, len(bullets))
        result: list[str] = []
        for bullet in bullets:
            remaining = max(8, target_words - self._word_count("\n".join(result)))
            trimmed = self._trim_bullet(bullet, min(24, remaining))
            if not trimmed:
                continue
            result.append(trimmed)
            if len(result) >= minimum and self._word_count("\n".join(result)) >= target_words * 0.8:
                break
            if len(result) >= self.config.max_bullets:
                break
        return result[: max(minimum, 1)] if target_words < 12 else result

    def _expand_bullets(
        self,
        bullets: list[str],
        source_text: str,
        other_panel_text: str,
        target_words: int,
    ) -> list[str]:
        result = list(bullets[: self.config.max_bullets])
        existing_terms = [self._terms(bullet) for bullet in result]
        other_terms = self._terms(other_panel_text)
        for sentence in self._sentences(source_text):
            if len(result) >= self.config.max_bullets or self._word_count("\n".join(result)) >= target_words:
                break
            candidate = self._to_bullet(self._trim_words(sentence, 24))
            if not candidate:
                continue
            terms = self._terms(candidate)
            if not terms:
                continue
            duplicate = max((self._overlap(terms, prior) for prior in existing_terms), default=0.0)
            repeats_other_panel = self._overlap(terms, other_terms) >= 0.8
            if duplicate >= 0.65 or repeats_other_panel:
                continue
            result.append(candidate)
            existing_terms.append(terms)
        if self._word_count("\n".join(result)) < target_words:
            for sentence in self._sentences(source_text):
                if len(result) >= self.config.max_bullets or self._word_count("\n".join(result)) >= target_words:
                    break
                candidate = self._to_bullet(self._trim_words(sentence, 28))
                if not candidate:
                    continue
                if candidate in result:
                    continue
                terms = self._terms(candidate)
                duplicate = max((self._overlap(terms, prior) for prior in existing_terms), default=0.0)
                if duplicate >= 0.9:
                    continue
                result.append(candidate)
                existing_terms.append(terms)
        return result

    def _parse_bullets(self, text: str) -> list[str]:
        bullets = []
        for line in text.splitlines():
            clean = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if clean:
                bullet = self._to_bullet(clean)
                if bullet:
                    bullets.append(bullet)
        return bullets

    def _ground_bullets(self, bullets: list[str], source_text: str) -> list[str]:
        source_terms = self._terms(source_text)
        source_numbers = self._numbers(source_text)
        grounded = []
        for bullet in bullets:
            terms = self._terms(bullet)
            ratio = len(terms & source_terms) / len(terms) if terms else 1.0
            if self._numbers(bullet) - source_numbers:
                continue
            if len(terms) >= 4 and ratio < self.config.min_grounding_ratio:
                continue
            if self._ends_incomplete(bullet):
                continue
            grounded.append(bullet)
        return grounded

    def _is_effective(self, action: str, before_words: int, candidate: str, target_words: int) -> bool:
        after_words = self._word_count(candidate)
        if action == "expand":
            return before_words < after_words <= max(before_words + 8, round(target_words * 1.08))
        return 0 < after_words < before_words and after_words <= round(target_words * 1.08)

    def _is_usable_llm_text(self, candidate: str, target_words: int) -> bool:
        words = self._word_count(candidate)
        return 0 < words <= max(8, round(target_words * 1.08))

    def _sentences(self, text: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(sentence.split()) >= 5
        ]

    def _trim_bullet(self, bullet: str, max_words: int) -> str:
        clean = bullet[2:].strip() if bullet.startswith("- ") else bullet.strip()
        return self._to_bullet(self._trim_words(clean, max_words))

    def _trim_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return self._complete_text_within_limit(text, max_words)

    def _limit_text_to_target(self, text: str, target_words: int) -> str:
        if self._word_count(text) <= target_words:
            return text
        bullets = self._parse_bullets(text)
        if not bullets:
            return self._trim_words(text, target_words)
        result: list[str] = []
        for bullet in bullets:
            remaining = target_words - self._word_count("\n".join(result))
            if remaining <= 4:
                break
            trimmed = self._trim_bullet(bullet, min(remaining, 24))
            if trimmed:
                result.append(trimmed)
        fallback = next((self._trim_bullet(bullet, target_words) for bullet in bullets if self._trim_bullet(bullet, target_words)), "")
        return "\n".join(result or ([fallback] if fallback else []))

    def _to_bullet(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip().removeprefix("- ").strip()
        clean = re.sub(r"\s*\.\.\.\s*$", "", clean).strip()
        if not clean or self._ends_incomplete(clean):
            return ""
        if not clean.endswith((".", "!", "?")):
            clean += "."
        return f"- {clean}"

    def _ends_incomplete(self, text: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if not words:
            return True
        last = words[-1].strip("'").lower()
        return last in self._INCOMPLETE_ENDINGS

    def _complete_text_within_limit(self, text: str, max_words: int) -> str:
        clean = re.sub(r"\s+", " ", text).strip().removeprefix("- ").strip()
        sentences = self._sentences(clean)
        result: list[str] = []
        for sentence in sentences:
            candidate = " ".join([*result, sentence]).strip()
            if self._word_count(candidate) > max_words:
                break
            result.append(sentence)
        if result:
            return " ".join(result)
        words = clean.split()
        if len(words) <= max_words:
            return clean
        if max_words < 6:
            return " ".join(words[:max_words]).rstrip(" .,;:") + "."
        return " ".join(words[:max_words]).rstrip(" .,;:") + "."

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", text))

    def _terms(self, text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
            if word not in self._STOPWORDS
        }

    def _numbers(self, text: str) -> set[str]:
        return set(re.findall(r"\b\d[\d,.]*\b", text))

    def _overlap(self, first: set[str], second: set[str]) -> float:
        if not first or not second:
            return 0.0
        return len(first & second) / min(len(first), len(second))

    def _write_report(self, source_path: Path | None) -> None:
        if self.report_dir is None or source_path is None:
            return
        report_path = self.report_dir / source_path.stem / "content_fitting.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"source_pdf": str(source_path), "panels": self.last_decisions}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.warn(f"Could not write content fitting report: {exc}", RuntimeWarning, stacklevel=2)
