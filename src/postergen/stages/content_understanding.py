from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

from ..config import ContentConfig
from ..llm import TextGenerator
from ..models import PaperDocument, Section, SectionSummary
from ..prompt_loader import load_prompt


class ContentUnderstandingStage:
    _SECTION_PRIORITY = {
        "introduction": 0,
        "method": 1,
        "evaluation": 2,
        "results": 3,
        "discussion": 4,
        "conclusion": 5,
    }
    _KEY_TERMS = {
        "aim",
        "approach",
        "benchmark",
        "contribution",
        "dataset",
        "evaluate",
        "evaluation",
        "experiment",
        "framework",
        "generate",
        "method",
        "model",
        "propose",
        "result",
        "show",
        "system",
        "task",
    }
    _SUMMARY_STOPWORDS = {
        "about",
        "after",
        "also",
        "among",
        "and",
        "are",
        "based",
        "been",
        "being",
        "for",
        "from",
        "have",
        "into",
        "its",
        "that",
        "the",
        "their",
        "this",
        "through",
        "using",
        "was",
        "were",
        "which",
        "with",
    }

    def __init__(
        self,
        config: ContentConfig,
        text_generator: TextGenerator | None = None,
        selection_report_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.selection_report_dir = selection_report_dir
        self.last_selection_method = ""
        self.last_selection_error = ""
        self.last_selection_response = ""
        self.last_selection_decisions: dict[int, dict[str, str]] = {}
        self.last_candidate_count = 0
        self.last_selection_adjustments: list[str] = []
        self.last_summary_errors: list[str] = []
        self.last_summary_grounding_warnings: list[str] = []

    def select_and_summarize(self, paper: PaperDocument) -> list[SectionSummary]:
        self.last_selection_error = ""
        self.last_selection_response = ""
        self.last_selection_decisions = {}
        self.last_selection_adjustments = []
        self.last_summary_errors = []
        self.last_summary_grounding_warnings = []
        selected = self._select_sections(paper.sections)
        summaries = [self._summarize_section(section.title, section.text) for section in selected]
        self._write_selection_report(paper, selected, summaries)
        return summaries

    def _select_sections(self, sections: list[Section]) -> list[Section]:
        if not sections:
            self.last_selection_method = "none"
            self.last_candidate_count = 0
            return []
        candidates = self._candidate_sections(sections)
        self.last_candidate_count = len(candidates)
        if self.config.require_llm and self.text_generator is None:
            raise RuntimeError("Content section selection requires an LLM, but no text generator is configured.")
        if self.text_generator is not None:
            try:
                selected = self._select_sections_with_llm(sections, candidates)
            except Exception as exc:
                self.last_selection_error = f"{type(exc).__name__}: {exc}"
                if self.config.require_llm:
                    raise RuntimeError(
                        f"LLM section selection failed and rule fallback is disabled: {self.last_selection_error}"
                    ) from exc
                warnings.warn(
                    "LLM section selection failed; using the rule-based fallback. "
                    f"Reason: {self.last_selection_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                selected = []
            if selected:
                self.last_selection_method = "llm"
                return self._enforce_selection_constraints(sections, candidates, selected)
            if not self.last_selection_error:
                self.last_selection_error = "The LLM returned no valid section numbers."
                if self.config.require_llm:
                    raise RuntimeError("LLM section selection returned no valid sections and rule fallback is disabled.")
                warnings.warn(
                    "LLM section selection returned no valid sections; using the rule-based fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        scored = [
            (index, section, self._section_relevance(section))
            for index, section in candidates
        ]

        self.last_selection_method = "rule"
        chosen = sorted(scored, key=lambda item: (-item[2], item[0]))[: self.config.max_sections]
        selected = [section for _, section, _ in sorted(chosen, key=lambda item: item[0])]
        return self._enforce_selection_constraints(sections, candidates, selected)

    def _select_sections_with_llm(
        self,
        sections: list[Section],
        candidates: list[tuple[int, Section]],
    ) -> list[Section]:
        prompt = self._selection_prompt(candidates)
        response = self.text_generator.generate(prompt)
        self.last_selection_response = response
        selected_indices = self._parse_selected_section_indices(response, candidates)
        if not selected_indices:
            return []

        candidate_indexes = {index for index, _ in candidates}
        selected = [
            sections[index]
            for index in sorted(selected_indices)
            if index in candidate_indexes and 0 <= index < len(sections)
        ]
        return selected[: self.config.max_sections]

    def _selection_prompt(self, candidates: list[tuple[int, Section]]) -> str:
        section_descriptions = []
        for index, section in candidates:
            excerpt = self._clean_sentence(section.text[: self.config.selection_excerpt_chars])
            section_descriptions.append(
                f"{index + 1}. Title: {section.title}\n"
                f"Category hint: {self._effective_category(section).title()}\n"
                f"Source word count: {len(section.text.split())}\n"
                f"Excerpt: {excerpt}"
            )

        return load_prompt(
            "section_selection.txt",
            MIN_SECTIONS=min(3, self.config.max_sections),
            MAX_SECTIONS=self.config.max_sections,
            CANDIDATES="\n\n".join(section_descriptions),
        )

    def _parse_selected_section_indices(
        self,
        response: str,
        candidates: list[tuple[int, Section]] | None = None,
    ) -> list[int]:
        response = response.strip()
        object_match = re.search(r"\{.*\}", response, re.DOTALL)
        if object_match:
            try:
                data = json.loads(object_match.group(0))
                items = data.get("selected_sections", [])
                if isinstance(items, list):
                    indexes = []
                    for item in items:
                        if isinstance(item, dict) and str(item.get("id", "")).isdigit():
                            index = int(item["id"]) - 1
                            indexes.append(index)
                            self.last_selection_decisions[index] = {
                                "poster_role": str(item.get("poster_role", "supporting_content")),
                                "reason": str(item.get("reason", "Selected by the LLM.")),
                            }
                        elif str(item).isdigit():
                            index = int(item) - 1
                            indexes.append(index)
                            self.last_selection_decisions[index] = {
                                "poster_role": "supporting_content",
                                "reason": "Selected by the LLM.",
                            }
                    if indexes:
                        return indexes
            except json.JSONDecodeError:
                pass

        json_match = re.search(r"\[[^\]]*\]", response, re.DOTALL)
        if json_match:
            try:
                values = json.loads(json_match.group(0))
                indexes = [int(value) - 1 for value in values if str(value).strip().isdigit()]
                for index in indexes:
                    self.last_selection_decisions[index] = {
                        "poster_role": "supporting_content",
                        "reason": "Selected by the LLM.",
                    }
                return indexes
            except json.JSONDecodeError:
                pass

        numbers = re.findall(r"\b\d+\b", response)
        if numbers:
            indexes = [int(number) - 1 for number in numbers]
            for index in indexes:
                self.last_selection_decisions[index] = {
                    "poster_role": "supporting_content",
                    "reason": "Selected by the LLM.",
                }
            return indexes

        normalized_response = re.sub(r"\s+", " ", response.lower())
        indexes = [
            index
            for index, section in candidates or []
            if len(section.title.strip()) >= 4 and section.title.lower() in normalized_response
        ]
        candidate_map = dict(candidates or [])
        for index in indexes:
            self.last_selection_decisions[index] = {
                "poster_role": self._poster_role(candidate_map[index]),
                "reason": "Selected by section title in the LLM response.",
            }
        return indexes

    def _candidate_sections(self, sections: list[Section]) -> list[tuple[int, Section]]:
        non_excluded = [
            (index, section)
            for index, section in enumerate(sections)
            if not self._is_excluded_section(section)
        ]
        substantive = [
            (index, section)
            for index, section in non_excluded
            if len(section.text.split()) >= self.config.min_section_words
            or (
                len(section.text.split()) >= self.config.min_standalone_section_words
                and not self._has_descendants(index, sections)
            )
        ]
        return substantive or non_excluded or list(enumerate(sections))

    def _has_descendants(self, index: int, sections: list[Section]) -> bool:
        parent_level = sections[index].level
        for following in sections[index + 1 :]:
            if following.level <= parent_level:
                return False
            if following.level > parent_level:
                return True
        return False

    def _is_excluded_section(self, section: Section) -> bool:
        category = self._effective_category(section)
        title = section.title.lower()
        if category in {"references", "appendix", "related work"}:
            return True
        if "related work" in title or "references" in title:
            return True
        return False

    def _is_candidate_section(self, section: Section) -> bool:
        return not self._is_excluded_section(section) and len(section.text.split()) >= self.config.min_section_words

    def _effective_category(self, section: Section) -> str:
        category = (section.category or "").lower().strip()
        if category and category != "other":
            return category
        title = section.title.lower()
        if any(term in title for term in ("introduction", "background", "motivation")):
            return "introduction"
        if any(term in title for term in ("result", "finding", "analysis")):
            return "results"
        if any(term in title for term in ("conclusion", "future work")):
            return "conclusion"
        if any(term in title for term in ("dataset", "data set", "instruct")):
            return "dataset"
        if any(term in title for term in ("evaluation", "benchmark", "metric")):
            return "evaluation"
        if any(term in title for term in ("method", "approach", "framework", "architecture", "agent", "system")):
            return "method"
        return category or "other"

    def _enforce_selection_constraints(
        self,
        sections: list[Section],
        candidates: list[tuple[int, Section]],
        selected: list[Section],
    ) -> list[Section]:
        index_by_id = {id(section): index for index, section in enumerate(sections)}
        candidate_by_index = {index: section for index, section in candidates}
        selected_indexes = [
            index_by_id[id(section)]
            for section in selected
            if id(section) in index_by_id and index_by_id[id(section)] in candidate_by_index
        ]
        selected_indexes = list(dict.fromkeys(selected_indexes))

        required_indexes: list[int] = []
        for coverage_role, role_candidates in self._coverage_targets(candidates):
            best_index, best_section = max(
                role_candidates,
                key=lambda item: (
                    self._coverage_score(item[1], self._effective_category(item[1])),
                    len(item[1].text.split()),
                ),
            )
            required_indexes.append(best_index)
            if best_index not in selected_indexes:
                self.last_selection_adjustments.append(
                    f"Added {best_section.title} to cover the {coverage_role} content."
                )

        final_indexes = list(dict.fromkeys(required_indexes))
        covered_categories = {self._effective_category(candidate_by_index[index]) for index in final_indexes}

        for index in selected_indexes:
            if index in final_indexes:
                continue
            category = self._effective_category(candidate_by_index[index])
            if category not in covered_categories and len(final_indexes) < self.config.max_sections:
                final_indexes.append(index)
                covered_categories.add(category)

        for index in selected_indexes:
            if index not in final_indexes and len(final_indexes) < self.config.max_sections:
                final_indexes.append(index)

        if len(final_indexes) < self.config.max_sections:
            remaining = sorted(
                (
                    (index, section)
                    for index, section in candidates
                    if index not in final_indexes
                ),
                key=lambda item: (-self._section_relevance(item[1]), item[0]),
            )
            for index, _section in remaining:
                if len(final_indexes) >= self.config.max_sections:
                    break
                final_indexes.append(index)

        final_indexes = sorted(final_indexes[: self.config.max_sections])
        removed = [
            sections[index].title
            for index in selected_indexes
            if index not in final_indexes
        ]
        if removed:
            self.last_selection_adjustments.append(
                "Removed redundant or lower-priority selections: " + ", ".join(removed)
            )
        for index in final_indexes:
            if index not in self.last_selection_decisions:
                self.last_selection_decisions[index] = {
                    "poster_role": self._poster_role(sections[index]),
                    "reason": "Added or retained by deterministic coverage validation.",
                }
        return [sections[index] for index in final_indexes]

    def _poster_role(self, section: Section) -> str:
        category = self._effective_category(section)
        if category == "introduction":
            return "research_context"
        if category in {"method", "dataset"}:
            return "core_contribution"
        if category in {"results", "evaluation"}:
            return "evidence_or_evaluation"
        if category == "conclusion":
            return "conclusion"
        return "supporting_content"

    def _coverage_targets(
        self,
        candidates: list[tuple[int, Section]],
    ) -> list[tuple[str, list[tuple[int, Section]]]]:
        by_category: dict[str, list[tuple[int, Section]]] = {}
        for index, section in candidates:
            by_category.setdefault(self._effective_category(section), []).append((index, section))

        targets: list[tuple[str, list[tuple[int, Section]]]] = []
        if by_category.get("introduction"):
            targets.append(("research context", by_category["introduction"]))

        contribution = [
            *by_category.get("method", []),
            *by_category.get("dataset", []),
        ]
        if contribution:
            targets.append(("core contribution", contribution))

        evidence = [
            *by_category.get("results", []),
            *by_category.get("evaluation", []),
        ]
        if evidence:
            targets.append(("evidence or evaluation", evidence))

        if by_category.get("conclusion"):
            targets.append(("conclusion", by_category["conclusion"]))
        return targets[: self.config.max_sections]

    def _coverage_score(self, section: Section, category: str) -> float:
        score = self._section_relevance(section)
        title = section.title.lower()
        if category == "results":
            if any(term in title for term in ("result", "finding", "main result")):
                score += 5.0
            if any(term in title for term in ("setup", "metric")):
                score -= 4.0
        if category in {"results", "evaluation"} and any(
            term in title for term in ("setup", "setting", "implementation detail")
        ):
            score -= 4.0
        if category in {"method", "dataset"} and any(
            term in title
            for term in ("agent", "architecture", "framework", "approach", "system", "dataset", "data set")
        ):
            score += 4.0
        return score

    def _section_relevance(self, section: Section) -> float:
        category = self._effective_category(section)
        title = section.title.lower()
        text = section.text.lower()
        words = re.findall(r"[a-zA-Z]{3,}", text)

        score = 0.0
        score += {
            "introduction": 4.0,
            "method": 5.0,
            "dataset": 4.5,
            "evaluation": 4.5,
            "results": 5.0,
            "discussion": 2.5,
            "conclusion": 3.5,
        }.get(category, 1.5)
        score += min(len(words) / 180.0, 3.0)
        score += min(len(re.findall(r"\b(?:fig\.?|figure)\s*\d+", text, re.IGNORECASE)), 3) * 0.8
        score += len(set(words) & self._KEY_TERMS) * 0.25
        if any(term in title for term in ("result", "experiment", "evaluation", "benchmark", "method", "approach")):
            score += 1.0
        return score

    def _summarize_section(self, title: str, text: str) -> SectionSummary:
        if self.config.require_llm and self.text_generator is None:
            raise RuntimeError(f"Section summarization for '{title}' requires an LLM, but no text generator is configured.")
        if self.text_generator is not None:
            try:
                return self._summarize_with_llm(title, text)
            except Exception as exc:
                error = f"{title}: {type(exc).__name__}: {exc}"
                self.last_summary_errors.append(error)
                if self.config.require_llm:
                    raise RuntimeError(f"LLM summarization failed and rule fallback is disabled: {error}") from exc
                warnings.warn(
                    f"LLM summarization failed for section '{title}'; using the rule-based fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        target_min, target_max = self.config.summary_sentence_range
        sentences = self._sentences(text)
        scored = sorted(
            enumerate(sentences),
            key=lambda item: (-self._score_sentence(title, item[1], item[0]), item[0]),
        )
        chosen_indexes = sorted(index for index, _ in scored[:target_max])
        chosen = [self._to_bullet(sentences[index]) for index in chosen_indexes]

        if len(chosen) < target_min:
            fallback = [self._to_bullet(sentence) for sentence in sentences[:target_min]]
            chosen = fallback or [self._to_bullet(text)]

        return SectionSummary(
            title=title,
            summary_sentences=chosen[:target_max],
            source_text=text,
            key_facts=self._extract_key_facts(text),
        )

    def _summarize_with_llm(self, title: str, text: str) -> SectionSummary:
        target_min, target_max = self.config.summary_sentence_range
        key_facts = self._extract_key_facts(text)
        prompt = self._summary_prompt(title, text[: self.config.max_section_chars], target_min, target_max, key_facts)
        response = self.text_generator.generate(prompt)
        bullets = self._ground_bullets(title, self._parse_bullets(response), text)
        if len(bullets) < target_min and not self.config.require_llm:
            bullets = self._unique_bullets(
                bullets + self._rule_based_bullets(title, text, target_max)
            )
        if not bullets and not self.config.require_llm:
            bullets = [self._to_bullet(response)] if response.strip() else self._rule_based_bullets(title, text, 1)
        if self.config.require_llm and len(bullets) < target_min:
            raise ValueError("The LLM summary did not contain enough grounded bullet points.")
        return SectionSummary(title=title, summary_sentences=bullets[:target_max], source_text=text, key_facts=key_facts)

    def _summary_prompt(
        self,
        title: str,
        text: str,
        target_min: int,
        target_max: int,
        key_facts: dict[str, list[str]] | None = None,
    ) -> str:
        return load_prompt(
            "section_summary.txt",
            SECTION_TITLE=title,
            MIN_BULLETS=target_min,
            MAX_BULLETS=target_max,
            KEY_DETAILS=self._format_key_facts(key_facts or self._extract_key_facts(text)),
            SECTION_TEXT=text,
        )

    def _parse_bullets(self, response: str) -> list[str]:
        bullets: list[str] = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\.)]\s*", "", line)
            line = re.sub(r"^[•*-]\s*", "", line)
            if line:
                bullets.append(self._to_bullet(line))
        return bullets

    def _rule_based_bullets(self, title: str, text: str, limit: int) -> list[str]:
        sentences = self._sentences(text)
        scored = sorted(
            enumerate(sentences),
            key=lambda item: (-self._score_sentence(title, item[1], item[0]), item[0]),
        )
        chosen_indexes = sorted(index for index, _ in scored[:limit])
        return [self._to_bullet(sentences[index]) for index in chosen_indexes]

    def _key_detail_hints(self, text: str, limit: int = 10) -> list[str]:
        details: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            cleaned = self._clean_sentence(value)
            cleaned = cleaned.strip(" ,.;:")
            key = cleaned.lower()
            if cleaned and key not in seen and len(cleaned) <= 140:
                seen.add(key)
                details.append(cleaned)

        detail_patterns = [
            r"\b(?:Table|Figure|Fig\.|Algorithm|Theorem|Lemma|Corollary)\s*\d+(?:\.\d+)?\b",
            r"\b[A-Z][A-Za-z0-9-]*(?:Net|Former|GAN|BERT|GPT|LLM|VLM|CNN|RNN|SVM|PCA|KRR|RL|QA)(?:[-–]?[A-Za-z0-9]+)?\b",
            r"\b[A-Z]{2,}(?:[-–][A-Za-z0-9]+)*\b",
            r"\b\d[\d,.]*\s*(?:(?:[A-Za-z]+[-–]?){1,3}\s+)?(?:%|percent|tokens?|examples?|samples?|pairs?|models?|datasets?|parameters?|hours?|minutes?|seconds?|GB|MB|B|M|K)\b",
            r"\b(?:AUROC|AUPRC|accuracy|precision|recall|F1|ROUGE|BLEU|score|loss|error)\b[^.;]{0,70}",
        ]
        for pattern in detail_patterns:
            for match in re.finditer(pattern, text):
                add(match.group(0))
                if len(details) >= limit:
                    return details

        for sentence in self._sentences(text):
            if len(details) >= limit:
                break
            if self._has_detail_signal(sentence):
                add(sentence)
        return details[:limit]

    def _extract_key_facts(self, text: str, per_category_limit: int = 5) -> dict[str, list[str]]:
        facts = {
            "datasets_or_benchmarks": [],
            "methods_or_components": [],
            "metrics_or_results": [],
            "numbers": [],
            "figures_or_tables": [],
            "claims_or_conclusions": [],
        }
        seen: set[str] = set()

        def add(category: str, value: str) -> None:
            cleaned = self._clean_sentence(value).strip(" ,.;:")
            cleaned = re.sub(r"\s+", " ", cleaned)
            key = f"{category}:{cleaned.lower()}"
            if not cleaned or key in seen or len(cleaned) > 180:
                return
            if len(facts[category]) >= per_category_limit:
                return
            seen.add(key)
            facts[category].append(cleaned)

        for match in re.finditer(r"\b(?:Table|Figure|Fig\.|Algorithm|Theorem|Lemma|Corollary)\s*\d+(?:\.\d+)?\b", text):
            add("figures_or_tables", match.group(0))

        for match in re.finditer(
            r"\b\d[\d,.]*\s*(?:(?:[A-Za-z]+[-–]?){1,3}\s+)?(?:%|percent|tokens?|examples?|samples?|pairs?|models?|datasets?|parameters?|hours?|minutes?|seconds?|GB|MB|B|M|K)\b",
            text,
            flags=re.IGNORECASE,
        ):
            add("numbers", match.group(0))

        for sentence in self._sentences(text):
            lowered = sentence.lower()
            if re.search(r"\b(dataset|benchmark|corpus|data set)\b", lowered):
                add("datasets_or_benchmarks", sentence)
            if re.search(r"\b(method|model|framework|agent|module|algorithm|architecture|component|pipeline)\b", lowered):
                add("methods_or_components", sentence)
            if re.search(r"\b(result|outperform|improve|achieve|accuracy|auroc|auprc|f1|rouge|bleu|score|error|loss)\b", lowered):
                add("metrics_or_results", sentence)
            if re.search(r"\b(show|demonstrate|indicate|suggest|conclude|therefore|we find|we observe)\b", lowered):
                add("claims_or_conclusions", sentence)

        # Preserve compact named entities and acronyms that are often answer-bearing.
        for match in re.finditer(
            r"\b[A-Z][A-Za-z0-9-]*(?:Net|Former|GAN|BERT|GPT|LLM|VLM|CNN|RNN|SVM|PCA|KRR|RL|QA)(?:[-–]?[A-Za-z0-9]+)?\b|\b[A-Z]{2,}(?:[-–][A-Za-z0-9]+)*\b",
            text,
        ):
            token = match.group(0)
            category = "metrics_or_results" if re.search(r"AUROC|AUPRC|F1|ROUGE|BLEU", token, re.I) else "methods_or_components"
            add(category, token)

        return {category: values for category, values in facts.items() if values}

    def _format_key_facts(self, key_facts: dict[str, list[str]]) -> str:
        lines: list[str] = []
        labels = {
            "datasets_or_benchmarks": "Datasets or benchmarks",
            "methods_or_components": "Methods or components",
            "metrics_or_results": "Metrics or results",
            "numbers": "Numbers",
            "figures_or_tables": "Figures or tables",
            "claims_or_conclusions": "Claims or conclusions",
        }
        for category, label in labels.items():
            values = key_facts.get(category, [])
            if values:
                lines.append(f"{label}:")
                lines.extend(f"- {value}" for value in values)
        return "\n".join(lines) or "- None detected."

    def _has_detail_signal(self, sentence: str) -> bool:
        return bool(
            re.search(r"\d", sentence)
            or re.search(r"\b[A-Z]{2,}(?:[-–][A-Za-z0-9]+)*\b", sentence)
            or re.search(r"\b(?:dataset|benchmark|metric|table|figure|model|theorem|algorithm)\b", sentence, re.I)
        )

    def _unique_bullets(self, bullets: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for bullet in bullets:
            key = re.sub(r"\W+", "", bullet).lower()
            if bullet and key not in seen:
                seen.add(key)
                unique.append(bullet)
        return unique

    def _ground_bullets(self, title: str, bullets: list[str], source_text: str) -> list[str]:
        grounded: list[str] = []
        source_numbers = self._number_tokens(source_text)
        source_terms = self._summary_terms(source_text)
        for bullet in bullets:
            unsupported_numbers = sorted(self._number_tokens(bullet) - source_numbers)
            bullet_terms = self._summary_terms(bullet)
            grounding_ratio = len(bullet_terms & source_terms) / len(bullet_terms) if bullet_terms else 1.0
            reasons = []
            if unsupported_numbers:
                reasons.append("unsupported numbers: " + ", ".join(unsupported_numbers))
            if len(bullet_terms) >= 4 and grounding_ratio < self.config.min_summary_grounding_ratio:
                reasons.append(f"low source-term support ({grounding_ratio:.2f})")
            if reasons:
                self.last_summary_grounding_warnings.append(
                    f"{title}: rejected '{bullet}' ({'; '.join(reasons)})"
                )
                continue
            grounded.append(bullet)
        return grounded

    def _number_tokens(self, text: str) -> set[str]:
        return {
            re.sub(r"[^0-9]", "", token)
            for token in re.findall(r"\b\d[\d,.]*\b", text)
            if re.sub(r"[^0-9]", "", token)
        }

    def _summary_terms(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
        return {word.strip("-") for word in words if word not in self._SUMMARY_STOPWORDS}

    def _write_selection_report(
        self,
        paper: PaperDocument,
        selected: list[Section],
        summaries: list[SectionSummary],
    ) -> None:
        if self.selection_report_dir is None:
            return

        section_indexes = {id(section): index + 1 for index, section in enumerate(paper.sections)}
        report = {
            "source_pdf": str(paper.source_path),
            "selection_method": self.last_selection_method,
            "selection_error": self.last_selection_error,
            "selection_response": self.last_selection_response[:4000],
            "parsed_section_count": len(paper.sections),
            "candidate_section_count": self.last_candidate_count,
            "selected_section_count": len(selected),
            "selected_sections": [
                {
                    "source_index": section_indexes[id(section)],
                    "title": section.title,
                    "level": section.level,
                    "category": section.category,
                    "effective_category": self._effective_category(section),
                    "source_word_count": len(section.text.split()),
                    "source_excerpt": section.text[:500],
                    "key_detail_hints": self._key_detail_hints(section.text),
                    "key_facts": summary.key_facts,
                    "summary_bullet_count": len(summary.summary_sentences),
                    "summary_bullets": summary.summary_sentences,
                    "poster_role": self.last_selection_decisions.get(
                        section_indexes[id(section)] - 1,
                        {},
                    ).get("poster_role", self._poster_role(section)),
                    "selection_reason": self.last_selection_decisions.get(
                        section_indexes[id(section)] - 1,
                        {},
                    ).get("reason", ""),
                }
                for section, summary in zip(selected, summaries)
            ],
            "selection_adjustments": self.last_selection_adjustments,
            "summary_errors": self.last_summary_errors,
            "summary_grounding_warnings": self.last_summary_grounding_warnings,
        }
        report_path = self.selection_report_dir / paper.source_path.stem / "section_selection.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(
                f"Could not write section selection report: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _sort_sections(self, sections: list[Section]) -> list[Section]:
        return sorted(
            sections,
            key=lambda section: self._SECTION_PRIORITY.get((section.category or section.title).lower(), 99),
        )

    def _sentences(self, text: str) -> list[str]:
        candidates = re.split(r"(?<=[.!?])\s+", text.strip())
        return [self._clean_sentence(sentence) for sentence in candidates if len(sentence.split()) >= 5]

    def _score_sentence(self, title: str, sentence: str, index: int) -> float:
        words = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
        score = len(words & self._KEY_TERMS)
        if self._has_detail_signal(sentence):
            score += 1.2
        score += min(len(re.findall(r"\b\d[\d,.]*\b", sentence)), 3) * 0.35
        if title.lower() in {"results", "conclusion"} and words & {"show", "result", "evaluate", "evaluation"}:
            score += 2
        if title.lower() == "method" and words & {"method", "framework", "model", "approach"}:
            score += 2
        score += max(0.0, 1.5 - index * 0.15)
        if len(sentence.split()) > 45:
            score -= 1
        return score

    def _to_bullet(self, sentence: str) -> str:
        sentence = self._clean_sentence(sentence)
        if not sentence:
            return ""
        sentence = sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."
        return f"- {sentence}"

    def _clean_sentence(self, sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip(" ;")
