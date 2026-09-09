from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from ..config import AssetSelectionConfig
from ..llm import TextGenerator
from ..models import Figure, PaperDocument, SectionSummary, Table
from ..prompt_loader import load_prompt


@dataclass(slots=True)
class AssetDecision:
    kind: str
    asset_id: str
    reason: str
    score: float | None = None
    role: str = "generic"
    suitability: str = "medium"
    evidence: list[str] | None = None
    target_section: str = ""


class AssetSelectionStage:
    """Select poster-worthy figures and tables after PDF parsing."""

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "we",
        "with",
    }
    _POSTER_TERMS = {
        "ablation",
        "architecture",
        "benchmark",
        "calibration",
        "chart",
        "comparison",
        "dataset",
        "diagram",
        "evaluation",
        "experiment",
        "framework",
        "method",
        "overview",
        "performance",
        "pipeline",
        "proposed",
        "result",
        "results",
        "roc",
        "system",
    }
    _LOW_VALUE_TERMS = {
        "acknowledgement",
        "acknowledgment",
        "appendix",
        "icon",
        "logo",
        "supplementary",
    }

    def __init__(
        self,
        config: AssetSelectionConfig,
        text_generator: TextGenerator | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.report_dir = report_dir
        self.last_selection_method = ""
        self.last_selection_error = ""
        self.last_selection_response = ""
        self.last_decisions: list[AssetDecision] = []
        self.last_evaluations: list[AssetDecision] = []
        self.last_description_error = ""
        self.last_descriptions: list[dict] = []

    def select(
        self,
        paper: PaperDocument,
        sections: list[SectionSummary],
    ) -> tuple[list[Figure], list[Table]]:
        self.last_selection_error = ""
        self.last_selection_response = ""
        self.last_decisions = []
        self.last_evaluations = []
        self.last_description_error = ""
        self.last_descriptions = []
        candidate_figures, candidate_tables = self._prepare_asset_representations(paper, sections)
        candidates: list[Figure | Table] = [*candidate_figures, *candidate_tables]
        if not candidates or self.config.max_total_assets <= 0:
            self.last_selection_method = "none"
            self._write_report(paper, candidates, [], [])
            return [], []

        decisions: list[AssetDecision] = []
        if self.config.require_llm and self.text_generator is None:
            raise RuntimeError("Asset selection requires an LLM, but no text generator is configured.")
        if self.text_generator is not None:
            try:
                response = self.text_generator.generate(self._selection_prompt(sections, candidate_figures, candidate_tables))
                self.last_selection_response = response
                decisions = self._parse_llm_decisions(response, candidate_figures, candidate_tables)
                evaluations = self._score_candidates(sections, candidate_figures, candidate_tables)
                self.last_evaluations = evaluations
                decisions = self._enrich_decisions(decisions, evaluations)
                self.last_selection_method = "llm"
            except Exception as exc:
                self.last_selection_error = f"{type(exc).__name__}: {exc}"
                if self.config.require_llm:
                    raise RuntimeError(
                        f"LLM asset selection failed and rule fallback is disabled: {self.last_selection_error}"
                    ) from exc
                warnings.warn(
                    "LLM asset selection failed; using the rule-based fallback. "
                    f"Reason: {self.last_selection_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.text_generator is None or not decisions:
            if self.text_generator is not None and not self.last_selection_error:
                self.last_selection_error = "The LLM returned no valid asset IDs."
                if self.config.require_llm:
                    raise RuntimeError("LLM asset selection returned no valid assets and rule fallback is disabled.")
                warnings.warn(
                    "LLM asset selection returned no valid assets; using the rule-based fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            decisions = self._rule_based_decisions(sections, candidate_figures, candidate_tables)
            self.last_selection_method = "rule"

        decisions = self._apply_limits(decisions)
        self.last_decisions = decisions
        selected_keys = {(item.kind, item.asset_id.lower()) for item in decisions}
        figures = [
            figure
            for figure in candidate_figures
            if ("figure", figure.figure_id.lower()) in selected_keys
        ]
        tables = [
            table
            for table in candidate_tables
            if ("table", table.table_id.lower()) in selected_keys
        ]
        self._write_report(paper, candidates, figures, tables)
        return figures, tables

    def _prepare_asset_representations(
        self,
        paper: PaperDocument,
        sections: list[SectionSummary],
    ) -> tuple[list[Figure], list[Table]]:
        figures = list(paper.figures)
        tables = list(paper.tables)
        if not self.config.describe_assets_with_llm:
            return figures, tables
        if self.text_generator is None:
            if self.config.require_llm:
                raise RuntimeError("LLM figure/table description requires an LLM, but no text generator is configured.")
            return figures, tables

        try:
            described_figures = [self._describe_asset("figure", figure, sections) for figure in figures]
            described_tables = [self._describe_asset("table", table, sections) for table in tables]
            return described_figures, described_tables
        except Exception as exc:
            self.last_description_error = f"{type(exc).__name__}: {exc}"
            if self.config.require_llm:
                raise RuntimeError(
                    f"LLM figure/table description failed and fallback is disabled: {self.last_description_error}"
                ) from exc
            warnings.warn(
                "LLM figure/table description failed; continuing with original captions. "
                f"Reason: {self.last_description_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return figures, tables

    def _describe_asset(
        self,
        kind: str,
        asset: Figure | Table,
        sections: list[SectionSummary],
    ) -> Figure | Table:
        asset_id = self._asset_id(asset)
        context = self._asset_context(asset, kind, sections)
        prompt = load_prompt(
            "asset_description.txt",
            ASSET_TYPE=kind,
            ASSET_ID=asset_id,
            CAPTION=asset.caption[:1400],
            CONTEXT=context[:3000],
        )
        response = self.text_generator.generate(prompt)
        description = self._sanitize_description(response)
        self.last_descriptions.append(
            {
                "type": kind,
                "id": asset_id,
                "caption": asset.caption,
                "description": description,
                "context_sections": [
                    section.title
                    for section in sections
                    if self._contains_reference(section.source_text, asset, kind)
                ],
                "method": "llm",
            }
        )
        return replace(asset, summary=description)

    def _asset_context(
        self,
        asset: Figure | Table,
        kind: str,
        sections: list[SectionSummary],
    ) -> str:
        referenced = [
            section
            for section in sections
            if self._contains_reference(section.source_text, asset, kind)
        ]
        candidates = referenced or self._top_context_sections(asset, sections)
        chunks = []
        for section in candidates[:3]:
            chunks.append(
                f"Section: {section.title}\n"
                f"Poster summary: {section.text}\n"
                f"Source excerpt: {section.source_text[: self.config.section_excerpt_chars]}"
            )
        return "\n\n".join(chunks)

    def _top_context_sections(
        self,
        asset: Figure | Table,
        sections: list[SectionSummary],
    ) -> list[SectionSummary]:
        asset_terms = self._terms(asset.caption)
        ranked = sorted(
            (
                (len(asset_terms & self._terms(f"{section.title} {section.text} {section.source_text}")), index, section)
                for index, section in enumerate(sections)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [section for score, _index, section in ranked if score > 0] or sections[:2]

    def _sanitize_description(self, response: str) -> str:
        line = next((part.strip() for part in response.splitlines() if part.strip()), "")
        line = re.sub(r"^```(?:text)?\s*", "", line, flags=re.IGNORECASE).strip()
        line = line.strip("`\" ")
        line = re.sub(r"^\s*(?:description|summary)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+", " ", line)
        line = line.replace("…", "")
        line = re.sub(r"\s*\.\.\.\s*", " ", line).strip()
        words = line.split()
        if len(words) > 32:
            line = self._complete_phrase(words, 32)
        if len(line.split()) < 6:
            raise ValueError("The generated asset description is too short.")
        if not line.endswith((".", "!", "?")):
            line += "."
        return line

    def _complete_phrase(self, words: list[str], max_words: int) -> str:
        stop_endings = {"and", "or", "with", "by", "to", "from", "for", "of", "in", "as", "than", "while", "under", "over"}
        chosen = words[:max_words]
        while len(chosen) > 6 and chosen[-1].strip(".,;:").lower() in stop_endings:
            chosen = chosen[:-1]
        return " ".join(chosen).rstrip(" .,;:")

    def _selection_prompt(
        self,
        sections: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table],
    ) -> str:
        section_data = [
            {
                "title": section.title,
                "summary": section.summary_sentences,
                "source_excerpt": section.source_text[: self.config.section_excerpt_chars],
            }
            for section in sections
        ]
        figure_data = [self._asset_metadata("figure", figure) for figure in figures]
        table_data = [self._asset_metadata("table", table) for table in tables]
        return load_prompt(
            "asset_selection.txt",
            MAX_TOTAL=self.config.max_total_assets,
            MAX_FIGURES=self.config.max_figures,
            MAX_TABLES=self.config.max_tables,
            SECTIONS=json.dumps(section_data, ensure_ascii=True),
            FIGURES=json.dumps(figure_data, ensure_ascii=True),
            TABLES=json.dumps(table_data, ensure_ascii=True),
        )

    def _parse_llm_decisions(
        self,
        response: str,
        figures: list[Figure],
        tables: list[Table],
    ) -> list[AssetDecision]:
        valid_ids = {
            **{("figure", figure.figure_id.lower()): figure.figure_id for figure in figures},
            **{("table", table.table_id.lower()): table.table_id for table in tables},
        }
        raw_items: list = []
        object_match = re.search(r"\{.*\}", response, re.DOTALL)
        array_match = re.search(r"\[[\s\S]*\]", response)
        if object_match is not None:
            data = json.loads(object_match.group(0))
            if isinstance(data.get("selected_assets"), list):
                raw_items = data["selected_assets"]
            else:
                raw_items.extend(self._items_from_type_list(data.get("figures"), "figure"))
                raw_items.extend(self._items_from_type_list(data.get("tables"), "table"))
                raw_items.extend(self._items_from_legacy_dict(data.get("image_information"), "figure"))
                raw_items.extend(self._items_from_legacy_dict(data.get("table_information"), "table"))
        elif array_match is not None:
            raw_items = json.loads(array_match.group(0))

        decisions: list[AssetDecision] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_items:
            if isinstance(item, str):
                kind, requested_id, reason = self._kind_and_id_from_text(item, valid_ids)
            elif isinstance(item, dict):
                kind = str(item.get("type", "")).lower().strip()
                requested_id = str(item.get("id", "")).lower().strip()
                reason = str(item.get("reason", "Selected by the LLM.")).strip()
                if item.get("role"):
                    reason = f"{reason} Role suggested by LLM: {item.get('role')}."
            else:
                continue
            key = self._resolve_asset_key(kind, requested_id, valid_ids)
            if key not in valid_ids or key in seen:
                continue
            seen.add(key)
            decisions.append(
                AssetDecision(
                    kind=kind,
                    asset_id=valid_ids[key],
                    reason=reason,
                )
            )

        if not decisions:
            decisions = self._decisions_from_plain_text(response, valid_ids)
        if response.strip() and not decisions:
            raise ValueError("The LLM selected only unknown or invalid asset IDs.")
        return decisions

    def _items_from_type_list(self, value, kind: str) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [
            {"type": kind, "id": item.get("id", ""), "reason": item.get("reason", "Selected by the LLM.")}
            if isinstance(item, dict)
            else {"type": kind, "id": item, "reason": "Selected by the LLM."}
            for item in value
        ]

    def _items_from_legacy_dict(self, value, kind: str) -> list[dict]:
        if not isinstance(value, dict):
            return []
        return [
            {"type": kind, "id": key, "reason": "Selected by the LLM."}
            for key in value
        ]

    def _kind_and_id_from_text(
        self,
        value: str,
        valid_ids: dict[tuple[str, str], str],
    ) -> tuple[str, str, str]:
        lowered = value.lower().strip()
        for kind, asset_id in valid_ids:
            if asset_id in lowered:
                return kind, asset_id, "Selected by the LLM."
        return "", lowered, "Selected by the LLM."

    def _resolve_asset_key(
        self,
        kind: str,
        requested_id: str,
        valid_ids: dict[tuple[str, str], str],
    ) -> tuple[str, str]:
        direct = (kind, requested_id)
        if direct in valid_ids:
            return direct
        number = re.search(r"\d+[a-z]?", requested_id, re.IGNORECASE)
        if number is not None:
            for key in valid_ids:
                if key[0] == kind and re.search(r"\d+[a-z]?", key[1], re.IGNORECASE).group(0) == number.group(0):
                    return key
        return direct

    def _decisions_from_plain_text(
        self,
        response: str,
        valid_ids: dict[tuple[str, str], str],
    ) -> list[AssetDecision]:
        lowered = response.lower()
        matches = sorted(
            (
                (lowered.find(asset_id), kind, asset_id, canonical)
                for (kind, asset_id), canonical in valid_ids.items()
                if lowered.find(asset_id) >= 0
            ),
            key=lambda item: item[0],
        )
        return [
            AssetDecision(kind=kind, asset_id=canonical, reason="Selected from the LLM text response.")
            for _, kind, _asset_id, canonical in matches
        ]

    def _rule_based_decisions(
        self,
        sections: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table],
    ) -> list[AssetDecision]:
        evaluations = self._score_candidates(sections, figures, tables)
        self.last_evaluations = evaluations
        relevant = [decision for decision in evaluations if decision.score is not None and decision.score > 0]
        return sorted(relevant, key=lambda item: (-float(item.score or 0), item.kind, item.asset_id))

    def _score_candidates(
        self,
        sections: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table],
    ) -> list[AssetDecision]:
        return [
            self._score_asset("figure", figure, sections)
            for figure in figures
        ] + [
            self._score_asset("table", table, sections)
            for table in tables
        ]

    def _enrich_decisions(
        self,
        decisions: list[AssetDecision],
        evaluations: list[AssetDecision],
    ) -> list[AssetDecision]:
        evaluation_map = {
            (decision.kind, decision.asset_id.lower()): decision for decision in evaluations
        }
        enriched: list[AssetDecision] = []
        for decision in decisions:
            evaluated = evaluation_map.get((decision.kind, decision.asset_id.lower()))
            if evaluated is None:
                enriched.append(decision)
                continue
            if evaluated.score is not None and evaluated.score <= 0:
                continue
            enriched.append(
                AssetDecision(
                    kind=decision.kind,
                    asset_id=decision.asset_id,
                    reason=decision.reason,
                    score=evaluated.score,
                    role=evaluated.role,
                    suitability=evaluated.suitability,
                    evidence=evaluated.evidence,
                    target_section=evaluated.target_section,
                )
            )
        return enriched

    def _score_asset(
        self,
        kind: str,
        asset: Figure | Table,
        sections: list[SectionSummary],
    ) -> AssetDecision:
        caption_terms = self._terms(asset.representation)
        best_overlap = 0
        best_section = ""
        best_section_signal = -1.0
        explicit_sections: list[str] = []
        for section in sections:
            section_text = f"{section.title} {section.text} {section.source_text}"
            overlap = len(caption_terms & self._terms(section_text))
            best_overlap = max(best_overlap, overlap)
            if self._contains_reference(section.source_text, asset, kind):
                explicit_sections.append(section.title)
                signal = overlap + 20.0
            else:
                signal = overlap
            if signal > best_section_signal:
                best_section_signal = signal
                best_section = section.title

        width, height = self._image_size(asset.path)
        role = self._asset_role(kind, asset, width, height)
        score = best_overlap * 1.5
        score += len(caption_terms & self._POSTER_TERMS) * 0.75
        if explicit_sections:
            score += 10.0
        if caption_terms & self._LOW_VALUE_TERMS:
            score -= 15.0
        if asset.path is None or not asset.path.is_file():
            score -= 5.0
        if isinstance(asset, Table) and asset.cells:
            score += 0.5
        role_bonus = {
            "method_diagram": 5.0,
            "main_result": 4.0,
            "comparison_table": 2.5,
            "dataset_overview": 3.0,
            "qualitative_example": 2.0,
            "motivation_figure": 2.0,
            "ablation_table": 0.5,
            "dense_table": -3.0,
            "low_priority": -12.0,
            "generic": 0.0,
        }
        score += role_bonus.get(role, 0.0)
        readability_delta, suitability, readability_reason = self._readability_score(kind, asset, width, height, role)
        score += readability_delta

        reason_parts = []
        if explicit_sections:
            reason_parts.append(f"explicitly referenced by {', '.join(explicit_sections)}")
        if best_overlap:
            reason_parts.append(f"caption shares {best_overlap} content terms with selected sections")
        if isinstance(asset, Table) and asset.cells:
            reason_parts.append("structured table cells are available")
        reason_parts.append(f"classified as {role}")
        reason_parts.append(readability_reason)
        reason = "; ".join(reason_parts) or "caption has limited relevance to the selected sections"
        return AssetDecision(
            kind=kind,
            asset_id=self._asset_id(asset),
            reason=reason,
            score=round(score, 3),
            role=role,
            suitability=suitability,
            evidence=reason_parts,
            target_section=explicit_sections[0] if explicit_sections else best_section,
        )

    def _apply_limits(self, decisions: list[AssetDecision]) -> list[AssetDecision]:
        selected: list[AssetDecision] = []
        figure_count = 0
        table_count = 0
        dense_table_count = 0
        seen: set[tuple[str, str]] = set()
        ordered = sorted(decisions, key=lambda item: (-(item.score or 0), item.kind, item.asset_id))
        for index, decision in enumerate(ordered):
            key = (decision.kind, decision.asset_id.lower())
            if key in seen or len(selected) >= self.config.max_total_assets:
                continue
            if decision.kind == "figure" and figure_count >= self.config.max_figures:
                continue
            if decision.kind == "table" and table_count >= self.config.max_tables:
                continue
            if decision.kind == "table" and table_count >= 1 and self._has_remaining_figure(ordered[index + 1 :], seen):
                continue
            if decision.role == "dense_table" and dense_table_count >= self.config.max_dense_tables:
                continue
            if decision.kind not in {"figure", "table"}:
                continue
            selected.append(decision)
            seen.add(key)
            figure_count += decision.kind == "figure"
            table_count += decision.kind == "table"
            dense_table_count += decision.role == "dense_table"
        return selected

    def _has_remaining_figure(
        self,
        decisions: list[AssetDecision],
        seen: set[tuple[str, str]],
    ) -> bool:
        return any(
            decision.kind == "figure"
            and decision.score is not None
            and decision.score > 0
            and (decision.kind, decision.asset_id.lower()) not in seen
            and decision.suitability != "low"
            for decision in decisions
        )

    def _asset_metadata(self, kind: str, asset: Figure | Table) -> dict:
        width, height = self._image_size(asset.path)
        role = self._asset_role(kind, asset, width, height)
        _, suitability, _ = self._readability_score(kind, asset, width, height, role)
        return {
            "type": kind,
            "id": self._asset_id(asset),
            "caption": asset.caption,
            "description": asset.summary,
            "representation": asset.representation,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3) if width and height else None,
            "table_rows": len(asset.cells) if isinstance(asset, Table) else None,
            "role": role,
            "suitability": suitability,
        }

    def _asset_role(
        self,
        kind: str,
        asset: Figure | Table,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        text = asset.representation.lower()
        terms = self._terms(text)
        if terms & self._LOW_VALUE_TERMS:
            return "low_priority"
        if isinstance(asset, Table):
            row_count = len(asset.cells)
            if row_count > 20:
                return "dense_table"
            if any(term in text for term in ("ablation", "sensitivity", "component")):
                return "ablation_table"
            if any(term in text for term in ("result", "performance", "comparison", "score", "accuracy", "auroc", "auc")):
                return "comparison_table"
            return "generic"
        if any(term in text for term in ("result", "performance", "comparison", "accuracy", "auroc", "auc", "score", "metric")):
            return "main_result"
        if any(term in text for term in ("evaluation", "benchmark", "scoring")):
            return "main_result"
        if any(term in text for term in ("architecture", "framework", "pipeline", "overview", "workflow", "system", "diagram")):
            return "method_diagram"
        if any(term in text for term in ("dataset", "data set", "benchmark", "example", "samples")):
            return "dataset_overview"
        if any(term in text for term in ("qualitative", "visualization", "case study", "examples")):
            return "qualitative_example"
        if any(term in text for term in ("motivation", "problem", "illustration")):
            return "motivation_figure"
        if width and height and width > height * 1.6:
            return "method_diagram"
        return "generic"

    def _readability_score(
        self,
        kind: str,
        asset: Figure | Table,
        width: int | None,
        height: int | None,
        role: str,
    ) -> tuple[float, str, str]:
        if asset.path is None or not asset.path.is_file():
            return -4.0, "low", "asset image file is missing"
        if isinstance(asset, Table):
            row_count = len(asset.cells)
            if row_count > 20:
                return -5.0, "low", f"table has {row_count} rows and may be too dense for a poster"
            if row_count > 12:
                return -1.5, "medium", f"table has {row_count} rows and should be used sparingly"
            if row_count:
                return 1.5, "high", f"table has {row_count} rows and is compact enough to display"
        if width is None or height is None:
            return 0.0, "medium", "image dimensions are unavailable"
        min_side = min(width, height)
        aspect = width / max(1, height)
        if min_side < 180:
            return -3.0, "low", "image is very small and may be unreadable"
        if aspect > 4.5 or aspect < 0.22:
            return -2.0, "low", "image has an extreme aspect ratio"
        if role in {"method_diagram", "main_result", "dataset_overview"}:
            return 2.0, "high", "visual form is suitable for a poster panel"
        return 0.75, "medium", "visual is displayable but not clearly a key poster asset"

    def _image_size(self, path: Path | None) -> tuple[int | None, int | None]:
        if path is None or not path.is_file():
            return None, None
        try:
            from PIL import Image

            with Image.open(path) as image:
                return image.width, image.height
        except (OSError, ModuleNotFoundError):
            return None, None

    def _contains_reference(self, text: str, asset: Figure | Table, kind: str) -> bool:
        number = re.search(r"\d+[a-z]?", self._asset_id(asset), re.IGNORECASE)
        if number is None:
            return False
        label = r"(?:fig\.?|figure)" if kind == "figure" else "table"
        return bool(re.search(rf"\b{label}\s*{re.escape(number.group(0))}\b", text, re.IGNORECASE))

    def _terms(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {word for word in words if word not in self._STOPWORDS}

    def _asset_id(self, asset: Figure | Table) -> str:
        return asset.figure_id if isinstance(asset, Figure) else asset.table_id

    def _write_report(
        self,
        paper: PaperDocument,
        candidates: list[Figure | Table],
        figures: list[Figure],
        tables: list[Table],
    ) -> None:
        if self.report_dir is None:
            return
        selected_keys = {
            *(('figure', figure.figure_id.lower()) for figure in figures),
            *(('table', table.table_id.lower()) for table in tables),
        }
        evaluation_map = {
            (decision.kind, decision.asset_id.lower()): decision for decision in self.last_evaluations
        }
        candidate_data = []
        for asset in candidates:
            kind = "figure" if isinstance(asset, Figure) else "table"
            key = (kind, self._asset_id(asset).lower())
            decision = evaluation_map.get(key)
            candidate_data.append(
                {
                    **self._asset_metadata(kind, asset),
                    "selected": key in selected_keys,
                    "score": decision.score if decision else None,
                    "role": decision.role if decision else self._asset_metadata(kind, asset)["role"],
                    "suitability": decision.suitability if decision else self._asset_metadata(kind, asset)["suitability"],
                    "evidence": decision.evidence if decision else [],
                    "target_section": decision.target_section if decision else "",
                    "reason": decision.reason if decision else "Not evaluated or not selected by the LLM.",
                }
            )

        report = {
            "source_pdf": str(paper.source_path),
            "selection_method": self.last_selection_method,
            "selection_error": self.last_selection_error,
            "selection_response": self.last_selection_response[:4000],
            "description_enabled": self.config.describe_assets_with_llm,
            "description_error": self.last_description_error,
            "asset_descriptions": self.last_descriptions,
            "candidate_count": len(candidates),
            "selected_figure_count": len(figures),
            "selected_table_count": len(tables),
            "assets": candidate_data,
        }
        report_path = self.report_dir / paper.source_path.stem / "asset_selection.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write asset selection report: {exc}", RuntimeWarning, stacklevel=2)
