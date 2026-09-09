from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

from ..config import AssetMatchingConfig
from ..llm import TextGenerator
from ..models import Figure, SectionSummary, StructuredAsset, Table
from ..prompt_loader import load_prompt


class AssetStructurer:
    _STOPWORDS = {
        "academic",
        "and",
        "are",
        "figure",
        "for",
        "from",
        "generated",
        "generation",
        "into",
        "paper",
        "poster",
        "table",
        "that",
        "the",
        "this",
        "through",
        "using",
        "with",
    }

    def __init__(
        self,
        config: AssetMatchingConfig | None = None,
        text_generator: TextGenerator | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self.config = config or AssetMatchingConfig()
        self.text_generator = text_generator
        self.report_dir = report_dir
        self.last_matching_method = ""
        self.last_matching_error = ""
        self.last_matching_response = ""
        self.last_assignments: list[dict] = []
        self.last_caption_rewrites: list[dict] = []

    def build(
        self,
        summaries: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table] | None = None,
        source_path: Path | None = None,
    ) -> list[StructuredAsset]:
        figure_assignments = {summary.title: [] for summary in summaries}
        table_assignments = {summary.title: [] for summary in summaries}
        self.last_matching_error = ""
        self.last_matching_response = ""
        self.last_assignments = []
        self.last_caption_rewrites = []

        if self.config.require_llm and self.config.use_llm and (figures or tables) and self.text_generator is None:
            raise RuntimeError("Asset matching requires an LLM, but no text generator is configured.")
        matched_with_llm = False
        if self.config.use_llm and self.text_generator is not None and summaries:
            try:
                self._assign_with_llm(
                    summaries,
                    figures,
                    tables or [],
                    figure_assignments,
                    table_assignments,
                )
                self.last_matching_method = "llm"
                matched_with_llm = True
                if not self.config.require_llm:
                    completed = self._complete_llm_assignments(
                        summaries,
                        figures,
                        tables or [],
                        figure_assignments,
                        table_assignments,
                    )
                    if completed:
                        self.last_matching_method = "llm+rule_completion"
            except Exception as exc:
                self.last_matching_error = f"{type(exc).__name__}: {exc}"
                if self.config.require_llm:
                    raise RuntimeError(
                        f"LLM asset matching failed and rule fallback is disabled: {self.last_matching_error}"
                    ) from exc
                warnings.warn(
                    "LLM asset matching failed; using the rule-based fallback. "
                    f"Reason: {self.last_matching_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.last_assignments = []
                figure_assignments = {summary.title: [] for summary in summaries}
                table_assignments = {summary.title: [] for summary in summaries}

        if not matched_with_llm:
            if self.config.require_llm:
                raise RuntimeError("LLM asset matching produced no valid assignments and rule fallback is disabled.")
            self.last_matching_method = "rule"
            self._assign_assets(
                figures,
                summaries,
                figure_assignments,
                labels=("fig", "figure"),
                kind="figure",
                limit=self.config.max_figures_per_section,
            )
            self._assign_assets(
                tables or [],
                summaries,
                table_assignments,
                labels=("table",),
                kind="table",
                limit=self.config.max_tables_per_section,
            )

        structured = [
            StructuredAsset(
                section=summary,
                figures=figure_assignments[summary.title],
                tables=table_assignments[summary.title],
            )
            for summary in summaries
        ]
        structured = self._rewrite_structured_captions(structured)
        self._write_report(source_path, figures, tables or [])
        return structured

    def _rewrite_structured_captions(self, structured: list[StructuredAsset]) -> list[StructuredAsset]:
        if not self.config.rewrite_captions or self.text_generator is None:
            return structured
        rewritten: list[StructuredAsset] = []
        for item in structured:
            figures = [
                self._rewrite_asset_caption(figure, item.section.title, item.section.text)
                for figure in item.figures
            ]
            tables = [
                self._rewrite_asset_caption(table, item.section.title, item.section.text)
                for table in item.tables
            ]
            rewritten.append(StructuredAsset(section=item.section, figures=figures, tables=tables))
        return rewritten

    def _rewrite_asset_caption(
        self,
        asset: Figure | Table,
        section_title: str,
        section_text: str,
    ) -> Figure | Table:
        asset_id = self._asset_id(asset)
        original = asset.caption
        try:
            prompt = load_prompt(
                "caption_rewrite.txt",
                ASSET_ID=asset_id,
                SECTION_TITLE=section_title,
                SECTION_TEXT=section_text[:1200],
                ORIGINAL_CAPTION=original[:1200],
            )
            response = self.text_generator.generate(prompt)
            caption = self._sanitize_rewritten_caption(asset_id, response)
            method = "llm"
            error = ""
        except Exception as exc:
            caption = self._fallback_short_caption(asset_id, original)
            method = "fallback"
            error = f"{type(exc).__name__}: {exc}"
            if self.config.require_llm:
                # Caption rewriting is a presentational refinement. Keep generation robust
                # even in strict LLM mode by falling back to a grounded short caption.
                method = "fallback_strict"
        self.last_caption_rewrites.append(
            {
                "asset_id": asset_id,
                "section": section_title,
                "original_caption": original,
                "poster_caption": caption,
                "method": method,
                "error": error,
            }
        )
        return replace(asset, caption=caption)

    def _sanitize_rewritten_caption(self, asset_id: str, response: str) -> str:
        line = next((part.strip() for part in response.splitlines() if part.strip()), "")
        line = re.sub(r"^```(?:text|caption)?\s*", "", line, flags=re.IGNORECASE).strip()
        line = line.strip("`\" ")
        line = re.sub(r"\s+", " ", line)
        line = line.replace("…", "")
        line = re.sub(r"\s*\.\.\.\s*", " ", line).strip()
        if ":" in line:
            maybe_id, rest = line.split(":", 1)
            if self._asset_number(maybe_id) == self._asset_number(asset_id):
                line = rest.strip()
        line = self._strip_caption_label(line)
        words = line.split()
        if len(words) > 16:
            line = self._complete_caption_phrase(words, 16)
        if len(line.split()) < 4:
            raise ValueError("The rewritten caption is too short.")
        if not line.endswith((".", "!", "?")):
            line += "."
        return f"{asset_id}: {line}"

    def _fallback_short_caption(self, asset_id: str, caption: str) -> str:
        clean = self._strip_caption_label(caption)
        words = clean.split()
        if len(words) > 16:
            clean = self._complete_caption_phrase(words, 16)
        clean = clean.strip(" .,;:")
        if not clean:
            clean = "Selected visual evidence."
        if not clean.endswith((".", "!", "?")):
            clean += "."
        return f"{asset_id}: {clean}"

    def _strip_caption_label(self, caption: str) -> str:
        return re.sub(
            r"^\s*(?:fig(?:ure)?\.?|table)\s*\d+[a-z]?\s*[.:-]\s*",
            "",
            caption,
            flags=re.IGNORECASE,
        ).strip()

    def _complete_caption_phrase(self, words: list[str], max_words: int) -> str:
        stop_endings = {"and", "or", "with", "by", "to", "from", "for", "of", "in", "as", "than", "while", "under", "over"}
        chosen = words[:max_words]
        while len(chosen) > 4 and chosen[-1].strip(".,;:").lower() in stop_endings:
            chosen = chosen[:-1]
        return " ".join(chosen).rstrip(" .,;:")

    def _assign_with_llm(
        self,
        summaries: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table],
        figure_assignments: dict[str, list[Figure]],
        table_assignments: dict[str, list[Table]],
    ) -> None:
        section_data = [
            {
                "id": index + 1,
                "title": summary.title,
                "summary": summary.summary_sentences,
                "source_excerpt": self._remove_captions(summary.source_text)[:700],
            }
            for index, summary in enumerate(summaries)
        ]
        asset_data = [
            {
                "type": "figure",
                "id": figure.figure_id,
                "caption": figure.caption,
                "description": figure.summary,
                "representation": figure.representation,
                "role": self._asset_role(figure),
            }
            for figure in figures
        ] + [
            {
                "type": "table",
                "id": table.table_id,
                "caption": table.caption,
                "description": table.summary,
                "representation": table.representation,
                "role": self._asset_role(table),
                "table_rows": len(table.cells),
            }
            for table in tables
        ]
        prompt = load_prompt(
            "asset_matching.txt",
            SECTIONS=json.dumps(section_data, ensure_ascii=True),
            ASSETS=json.dumps(asset_data, ensure_ascii=True),
        )
        response = self.text_generator.generate(prompt)
        self.last_matching_response = response
        object_match = re.search(r"\{.*\}", response, re.DOTALL)
        if object_match is None:
            raise ValueError("The LLM response did not contain a JSON object.")
        data = json.loads(object_match.group(0))
        raw_assignments = data.get("assignments")
        if not isinstance(raw_assignments, list):
            raise ValueError("The LLM response is missing the assignments list.")

        asset_lookup: dict[tuple[str, str], Figure | Table] = {
            **{("figure", figure.figure_id.lower()): figure for figure in figures},
            **{("table", table.table_id.lower()): table for table in tables},
        }
        seen_assets: set[tuple[str, str]] = set()
        valid_count = 0
        for item in raw_assignments:
            if not isinstance(item, dict) or not str(item.get("section_id", "")).isdigit():
                continue
            section_index = int(item["section_id"]) - 1
            kind = str(item.get("asset_type", "")).lower().strip().replace("image", "figure")
            asset_id = str(item.get("asset_id", "")).lower().strip()
            key = (kind, asset_id)
            if not 0 <= section_index < len(summaries) or key not in asset_lookup or key in seen_assets:
                continue

            summary = summaries[section_index]
            asset = asset_lookup[key]
            target = figure_assignments if kind == "figure" else table_assignments
            limit = self.config.max_figures_per_section if kind == "figure" else self.config.max_tables_per_section
            if len(target[summary.title]) >= limit:
                continue
            if self.config.require_llm:
                target[summary.title].append(
                    self._with_reference(
                        asset,
                        summary.title,
                        score=0.0,
                        method="llm_direct",
                    )
                )
                seen_assets.add(key)
                valid_count += 1
                self.last_assignments.append(
                    {
                        "type": kind,
                        "id": self._asset_id(asset),
                        "section": summary.title,
                        "method": "llm_direct",
                        "term_overlap": None,
                        "score": None,
                        "reason": str(item.get("reason", "Assigned by the LLM.")),
                    }
                )
                continue
            labels = ("fig", "figure") if kind == "figure" else ("table",)
            validated_match = next(
                (
                    (score, overlap, method)
                    for score, overlap, method, candidate in self._rank_matches(asset, summaries, labels)
                    if candidate is summary
                ),
                None,
            )
            if validated_match is None:
                continue
            score, overlap, validation_method = validated_match
            target[summary.title].append(
                self._with_reference(
                    asset,
                    summary.title,
                    score=score,
                    method=f"llm_validated_{validation_method}",
                )
            )
            seen_assets.add(key)
            valid_count += 1
            self.last_assignments.append(
                {
                    "type": kind,
                    "id": self._asset_id(asset),
                    "section": summary.title,
                    "method": f"llm_validated_{validation_method}",
                    "term_overlap": overlap,
                    "score": round(score, 3),
                    "reason": str(item.get("reason", "Assigned by the LLM.")),
                }
            )

        if raw_assignments and valid_count == 0:
            raise ValueError("The LLM returned assignments that failed deterministic validation.")

    def _complete_llm_assignments(
        self,
        summaries: list[SectionSummary],
        figures: list[Figure],
        tables: list[Table],
        figure_assignments: dict[str, list[Figure]],
        table_assignments: dict[str, list[Table]],
    ) -> bool:
        assigned_figure_ids = {
            figure.figure_id.lower()
            for assigned in figure_assignments.values()
            for figure in assigned
        }
        assigned_table_ids = {
            table.table_id.lower()
            for assigned in table_assignments.values()
            for table in assigned
        }
        remaining_figures = [
            figure for figure in figures if figure.figure_id.lower() not in assigned_figure_ids
        ]
        remaining_tables = [
            table for table in tables if table.table_id.lower() not in assigned_table_ids
        ]
        before = sum(len(items) for items in figure_assignments.values()) + sum(
            len(items) for items in table_assignments.values()
        )
        self._assign_assets(
            remaining_figures,
            summaries,
            figure_assignments,
            labels=("fig", "figure"),
            kind="figure",
            limit=self.config.max_figures_per_section,
            method_prefix="rule_completion_",
        )
        self._assign_assets(
            remaining_tables,
            summaries,
            table_assignments,
            labels=("table",),
            kind="table",
            limit=self.config.max_tables_per_section,
            method_prefix="rule_completion_",
        )
        after = sum(len(items) for items in figure_assignments.values()) + sum(
            len(items) for items in table_assignments.values()
        )
        return after > before

    def _assign_assets(
        self,
        assets: list[Figure] | list[Table],
        summaries: list[SectionSummary],
        assignments: dict[str, list],
        labels: tuple[str, ...],
        kind: str,
        limit: int,
        method_prefix: str = "",
    ) -> None:
        rankings = [self._rank_matches(asset, summaries, labels) for asset in assets]
        edges = [
            (score, asset_index, overlap, method, summary)
            for asset_index, ranked in enumerate(rankings)
            for score, overlap, method, summary in ranked
        ]
        assigned_assets: dict[int, tuple[float, int, str, SectionSummary]] = {}
        for score, asset_index, overlap, method, summary in sorted(
            edges,
            key=lambda item: (-item[0], item[1], summaries.index(item[4])),
        ):
            if asset_index in assigned_assets or len(assignments[summary.title]) >= limit:
                continue
            asset = assets[asset_index]
            assignment_method = f"{method_prefix}{method}"
            assignments[summary.title].append(
                self._with_reference(asset, summary.title, score=score, method=assignment_method)
            )
            assigned_assets[asset_index] = (score, overlap, assignment_method, summary)

        for asset_index, asset in enumerate(assets):
            assigned = assigned_assets.get(asset_index)
            if assigned is not None:
                score, overlap, method, summary = assigned
                self.last_assignments.append(
                    {
                        "type": kind,
                        "id": self._asset_id(asset),
                        "section": summary.title,
                        "method": method,
                        "term_overlap": overlap,
                        "score": round(score, 3),
                    }
                )
            else:
                ranked = rankings[asset_index]
                self.last_assignments.append(
                    {
                        "type": kind,
                        "id": self._asset_id(asset),
                        "section": None,
                        "method": "unassigned",
                        "term_overlap": ranked[0][1] if ranked else 0,
                        "score": round(ranked[0][0], 3) if ranked else 0.0,
                    }
                )

    def _rank_matches(
        self,
        asset: Figure | Table,
        summaries: list[SectionSummary],
        labels: tuple[str, ...],
    ) -> list[tuple[float, int, str, SectionSummary]]:
        if not summaries:
            return []
        asset_terms = self._terms(asset.representation)
        explicit_ids = {
            id(summary)
            for summary in summaries
            if self._has_body_reference(self._asset_id(asset), summary.source_text, labels)
        }
        ranked = []
        asset_role = self._asset_role(asset)
        for summary in summaries:
            section_terms = self._terms(f"{summary.title} {summary.text} {self._remove_captions(summary.source_text)}")
            overlap = len(asset_terms & section_terms)
            is_explicit = id(summary) in explicit_ids
            if not is_explicit and not self.config.semantic_matching_enabled:
                continue
            if not is_explicit and overlap < self.config.min_semantic_overlap:
                continue
            coverage = overlap / max(1, len(asset_terms))
            score = overlap + coverage * 4.0 + (100.0 if is_explicit else 0.0)
            section_role = self._section_role(summary)
            compatibility = self._role_compatibility(asset_role, section_role)
            if asset_role != "generic":
                score += compatibility
            if section_role == "conclusion" and not is_explicit:
                score -= 5.0
            if asset_role == "dense_table" and section_role not in {"results", "evaluation"} and not is_explicit:
                score -= 8.0
            if asset_role == "comparison_table" and isinstance(asset, Table) and len(asset.cells) > 12 and not is_explicit:
                score -= 2.0
            if not is_explicit and score < self.config.min_semantic_score:
                continue
            ranked.append((score, overlap, "explicit_reference" if is_explicit else "semantic", summary))
        return sorted(ranked, key=lambda item: (-item[0], summaries.index(item[3])))

    def _asset_role(self, asset: Figure | Table) -> str:
        text = asset.representation.lower()
        if any(term in text for term in ("logo", "icon", "acknowledgement", "acknowledgment", "supplementary", "appendix")):
            return "low_priority"
        if isinstance(asset, Table):
            if len(asset.cells) > 20:
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
        return self._content_role(text)

    def _content_role(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ("result", "performance", "comparison", "experiment", "finding")):
            return "results"
        if any(term in lowered for term in ("evaluation", "benchmark", "metric", "scoring")):
            return "evaluation"
        if any(term in lowered for term in ("dataset", "data set", "instruction-response", "instruct")):
            return "dataset"
        if any(term in lowered for term in ("architecture", "framework", "pipeline", "multi-agent", "method")):
            return "method"
        if "introduction" in lowered or "motivation" in lowered:
            return "introduction"
        if "conclusion" in lowered:
            return "conclusion"
        return "generic"

    def _roles_compatible(self, asset_role: str, section_role: str) -> bool:
        compatible = {
            "method": {"method", "introduction"},
            "dataset": {"dataset", "method"},
            "evaluation": {"evaluation", "results"},
            "results": {"results", "evaluation"},
        }
        return section_role in compatible.get(asset_role, {asset_role})

    def _role_compatibility(self, asset_role: str, section_role: str) -> float:
        compatible = {
            "method_diagram": {"method": 9.0, "introduction": 3.0},
            "main_result": {"results": 9.0, "evaluation": 6.0},
            "comparison_table": {"results": 7.0, "evaluation": 5.0},
            "dataset_overview": {"dataset": 8.0, "method": 4.0, "introduction": 3.0},
            "qualitative_example": {"results": 5.0, "dataset": 4.0, "method": 2.0},
            "motivation_figure": {"introduction": 7.0, "method": 2.0},
            "ablation_table": {"results": 4.0, "evaluation": 4.0},
            "dense_table": {"results": 1.5, "evaluation": 1.5},
            "low_priority": {"introduction": -20.0, "method": -20.0, "results": -20.0, "evaluation": -20.0, "dataset": -20.0, "conclusion": -20.0},
            "method": {"method": 8.0, "introduction": 2.0},
            "dataset": {"dataset": 7.0, "method": 3.0},
            "evaluation": {"evaluation": 7.0, "results": 5.0},
            "results": {"results": 8.0, "evaluation": 5.0},
        }
        role_scores = compatible.get(asset_role, {})
        if section_role in role_scores:
            return role_scores[section_role]
        if section_role == "conclusion":
            return -4.0
        if role_scores:
            return -5.0
        return 0.0

    def _section_role(self, summary: SectionSummary) -> str:
        title = summary.title.lower()
        if any(term in title for term in ("introduction", "background", "motivation")):
            return "introduction"
        if any(term in title for term in ("result", "analysis", "experiment", "finding")):
            return "results"
        if any(term in title for term in ("evaluation", "benchmark", "metric")):
            return "evaluation"
        if any(term in title for term in ("dataset", "data set", "instruct")):
            return "dataset"
        if any(term in title for term in ("method", "framework", "architecture", "agent", "system")):
            return "method"
        if "conclusion" in title:
            return "conclusion"
        return self._content_role(summary.text)

    def _has_body_reference(self, asset_id: str, source_text: str, labels: tuple[str, ...]) -> bool:
        number = self._asset_number(asset_id)
        if not number:
            return False
        label_pattern = "|".join(re.escape(label) + r"\.?" for label in labels)
        pattern = re.compile(rf"\b(?:{label_pattern})\s*{re.escape(number)}\b", re.IGNORECASE)
        return bool(pattern.search(self._remove_captions(source_text)))

    def _remove_captions(self, text: str) -> str:
        return re.sub(
            r"(?:^|(?<=[.!?])\s+)(?:fig(?:ure)?\.?|table)\s*\d+[a-z]?\s*[.:-]\s*.*?(?=(?:fig(?:ure)?\.?|table)\s*\d+[a-z]?\s*[.:-]|$)",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    def _with_reference(
        self,
        asset: Figure | Table,
        section_title: str,
        score: float = 0.0,
        method: str = "",
    ) -> Figure | Table:
        references = list(asset.referenced_in)
        if section_title not in references:
            references.append(section_title)
        return replace(asset, referenced_in=references, match_score=score, match_method=method)

    def _asset_number(self, asset_id: str) -> str | None:
        match = re.search(r"\d+[a-z]?", asset_id, re.IGNORECASE)
        return match.group(0) if match else None

    def _asset_id(self, asset: Figure | Table) -> str:
        return asset.figure_id if isinstance(asset, Figure) else asset.table_id

    def _terms(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {self._stem(word) for word in words if word not in self._STOPWORDS}

    def _stem(self, word: str) -> str:
        if len(word) > 6 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 5 and word.endswith("ed"):
            return word[:-2]
        if len(word) > 5 and word.endswith("es"):
            return word[:-2]
        if len(word) > 4 and word.endswith("s"):
            return word[:-1]
        return word

    def _write_report(
        self,
        source_path: Path | None,
        figures: list[Figure],
        tables: list[Table],
    ) -> None:
        if self.report_dir is None or source_path is None:
            return
        report = {
            "source_pdf": str(source_path),
            "matching_method": self.last_matching_method,
            "matching_error": self.last_matching_error,
            "matching_response": self.last_matching_response[:4000],
            "candidate_figure_count": len(figures),
            "candidate_table_count": len(tables),
            "candidate_assets": [
                {
                    "type": "figure",
                    "id": figure.figure_id,
                    "caption": figure.caption,
                    "description": figure.summary,
                    "representation": figure.representation,
                    "role": self._asset_role(figure),
                }
                for figure in figures
            ] + [
                {
                    "type": "table",
                    "id": table.table_id,
                    "caption": table.caption,
                    "description": table.summary,
                    "representation": table.representation,
                    "role": self._asset_role(table),
                    "table_rows": len(table.cells),
                }
                for table in tables
            ],
            "assignments": self.last_assignments,
            "caption_rewrites": self.last_caption_rewrites,
        }
        report_path = self.report_dir / source_path.stem / "asset_matching.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write asset matching report: {exc}", RuntimeWarning, stacklevel=2)
