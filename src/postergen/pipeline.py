from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path

from .config import PipelineConfig
from .llm import build_text_generator
from .models import PosterLayout
from .stages.asset_selection import AssetSelectionStage
from .stages.asset_structuring import AssetStructurer
from .stages.content_understanding import ContentUnderstandingStage
from .stages.content_fitting import ContentFittingStage
from .stages.direct_llm_baseline import DirectLLMBaselineStage
from .stages.docling_parser import DoclingPaperParser
from .stages.layout import LayoutGenerationStage
from .stages.panel_quality import PanelQualityStage
from .stages.refinement import PanelRefinementStage


class PosterGenerationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._configure_llm_mode()
        self._configure_experiment_mode()
        self.report_root = config.report_root
        self.parser = DoclingPaperParser(asset_dir=self.report_root)
        self.text_generator = build_text_generator(config.llm)
        self.content_stage = ContentUnderstandingStage(
            config.content,
            self.text_generator,
            selection_report_dir=self.report_root,
        )
        self.content_fitting_stage = ContentFittingStage(
            config.content_fitting,
            config.layout,
            self.text_generator,
            report_dir=self.report_root,
        )
        self.asset_selection_stage = AssetSelectionStage(
            config.asset_selection,
            self.text_generator,
            report_dir=self.report_root,
        )
        self.asset_stage = AssetStructurer(
            config.asset_matching,
            self.text_generator,
            report_dir=self.report_root,
        )
        self.layout_stage = LayoutGenerationStage(config.layout, report_dir=self.report_root)
        self.panel_quality_stage = PanelQualityStage(config.layout, report_dir=self.report_root)
        self.refinement_stage = PanelRefinementStage()
        self.exporter = None

    def _configure_llm_mode(self) -> None:
        if self.config.llm.provider.lower() == "rule":
            return
        self.config.content.require_llm = True
        self.config.content_fitting.require_llm = True
        self.config.asset_selection.describe_assets_with_llm = True
        self.config.asset_selection.require_llm = True
        self.config.asset_matching.require_llm = True
        self.config.asset_matching.rewrite_captions = True

    def _configure_experiment_mode(self) -> None:
        mode = self.config.experiment_mode
        valid_modes = {
            "proposed",
            "direct_llm",
            "template_based",
            "without_semantic_figure_matching",
            "without_llm_figure_description",
            "without_panel_refinement",
            "template_layout_only",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unsupported experiment mode: {mode}")
        if mode in {"direct_llm", "template_based", "template_layout_only"}:
            self.config.layout.strategy = "template"
        if mode == "direct_llm":
            self.config.llm.max_output_tokens = max(self.config.llm.max_output_tokens, 3500)
        if mode == "without_semantic_figure_matching":
            self.config.asset_matching.use_llm = False
            self.config.asset_matching.require_llm = False
            self.config.asset_matching.semantic_matching_enabled = False
        if mode == "without_llm_figure_description":
            self.config.asset_selection.describe_assets_with_llm = False

    def run(self) -> PosterLayout:
        paper = self.parser.parse(self.config.input_pdf)
        self._write_experiment_config(paper.source_path)
        if self.config.experiment_mode == "direct_llm":
            return self._run_direct_llm_baseline(paper)

        selected_sections = self.content_stage.select_and_summarize(paper)
        selected_figures, selected_tables = self.asset_selection_stage.select(paper, selected_sections)
        structured_assets = self.asset_stage.build(
            selected_sections,
            selected_figures,
            selected_tables,
            source_path=paper.source_path,
        )
        layout = self.layout_stage.generate(paper.title, structured_assets, source_path=paper.source_path)
        if self.config.experiment_mode == "without_panel_refinement":
            quality_report = self.panel_quality_stage.analyze(layout, source_path=paper.source_path)
            self._write_refinement_history(
                paper.source_path,
                [
                    self._refinement_snapshot(
                        "disabled",
                        quality_report,
                        {},
                        {},
                        accepted=True,
                        best_score=self._quality_score(quality_report),
                    )
                ],
            )
            self._write_no_fitting_report(paper.source_path)
            if self.exporter is None:
                from .stages.pptx_exporter import PptxExporter

                self.exporter = PptxExporter(self.config.export, self.config.layout)
            self.exporter.export(layout, self.config.output_pptx)
            return layout

        refined_layout = self.refinement_stage.refine(layout)
        refined_layout = self.layout_stage.relayout(refined_layout, source_path=paper.source_path)
        refined_layout = self.content_fitting_stage.fit(
            refined_layout,
            {section.title: section.source_text for section in selected_sections},
            {section.title: section.key_facts for section in selected_sections},
            source_path=paper.source_path,
        )
        refinement_history: list[dict] = []
        quality_report = self.panel_quality_stage.analyze(refined_layout, source_path=paper.source_path)
        best_layout = deepcopy(refined_layout)
        best_quality_report = quality_report
        best_fitting_decisions = deepcopy(self.content_fitting_stage.last_decisions)
        best_score = self._quality_score(quality_report)
        for iteration in range(5):
            forced_actions, forced_reasons = self._forced_content_feedback(quality_report)
            refinement_history.append(
                self._refinement_snapshot(
                    iteration,
                    quality_report,
                    forced_actions,
                    forced_reasons,
                    accepted=True,
                    best_score=best_score,
                )
            )
            if not self.panel_quality_stage.has_actionable_issues(quality_report):
                break
            candidate_layout = self.panel_quality_stage.refine(refined_layout, source_path=paper.source_path)
            candidate_layout = self.layout_stage.relayout(candidate_layout, source_path=paper.source_path)
            candidate_layout = self.content_fitting_stage.fit(
                candidate_layout,
                {section.title: section.source_text for section in selected_sections},
                {section.title: section.key_facts for section in selected_sections},
                source_path=paper.source_path,
                forced_actions=forced_actions,
                forced_reasons=forced_reasons,
            )
            candidate_quality_report = self.panel_quality_stage.analyze(candidate_layout, source_path=paper.source_path)
            candidate_score = self._quality_score(candidate_quality_report)
            if candidate_score < best_score:
                refined_layout = candidate_layout
                quality_report = candidate_quality_report
                best_layout = deepcopy(candidate_layout)
                best_quality_report = candidate_quality_report
                best_fitting_decisions = deepcopy(self.content_fitting_stage.last_decisions)
                best_score = candidate_score
            else:
                refinement_history.append(
                    self._refinement_snapshot(
                        f"{iteration}_rejected",
                        candidate_quality_report,
                        forced_actions,
                        forced_reasons,
                        accepted=False,
                        best_score=best_score,
                    )
                )
                break
        refined_layout = best_layout
        self.content_fitting_stage.last_decisions = best_fitting_decisions
        self.content_fitting_stage._write_report(paper.source_path)
        self.layout_stage._write_report(paper.source_path, refined_layout.title, refined_layout.panels)
        final_quality_report = self.panel_quality_stage.analyze(refined_layout, source_path=paper.source_path)
        if self.panel_quality_stage.has_actionable_issues(final_quality_report):
            forced_actions, forced_reasons = self._forced_content_feedback(final_quality_report)
            content_only_candidate = self.content_fitting_stage.fit(
                refined_layout,
                {section.title: section.source_text for section in selected_sections},
                {section.title: section.key_facts for section in selected_sections},
                source_path=paper.source_path,
                forced_actions=forced_actions,
                forced_reasons=forced_reasons,
            )
            content_only_quality = self.panel_quality_stage.analyze(content_only_candidate, source_path=paper.source_path)
            if self._quality_score(content_only_quality) <= self._quality_score(final_quality_report):
                refined_layout = content_only_candidate
                final_quality_report = content_only_quality
                best_fitting_decisions = deepcopy(self.content_fitting_stage.last_decisions)
                self.content_fitting_stage.last_decisions = best_fitting_decisions
                self.content_fitting_stage._write_report(paper.source_path)
                refinement_history.append(
                    self._refinement_snapshot(
                        "final_content_only",
                        final_quality_report,
                        forced_actions,
                        forced_reasons,
                        accepted=True,
                        best_score=self._quality_score(final_quality_report),
                    )
                )
        if not refinement_history or refinement_history[-1]["issue_count"] != final_quality_report["summary"]["issue_count"]:
            refinement_history.append(
                self._refinement_snapshot(
                    "final",
                    final_quality_report,
                    {},
                    {},
                    accepted=True,
                    best_score=self._quality_score(best_quality_report),
                )
            )
        self._write_refinement_history(paper.source_path, refinement_history)
        if self.exporter is None:
            from .stages.pptx_exporter import PptxExporter

            self.exporter = PptxExporter(self.config.export, self.config.layout)
        self.exporter.export(refined_layout, self.config.output_pptx)
        return refined_layout

    def _run_direct_llm_baseline(self, paper) -> PosterLayout:
        if self.text_generator is None:
            raise RuntimeError("Direct LLM generation requires an LLM provider, not provider='rule'.")
        stage = DirectLLMBaselineStage(self.text_generator, max_sections=self.config.content.max_sections)
        stage.generate_html(paper)
        html_path = stage.save_html(paper, self.config.output_pptx)
        layout = stage.proxy_layout(paper)
        positioned = self.layout_stage.relayout(layout, source_path=paper.source_path)
        stage.write_report(paper, positioned, self.report_root, html_path)
        quality_report = self.panel_quality_stage.analyze(positioned, source_path=paper.source_path)
        self._write_refinement_history(
            paper.source_path,
            [
                self._refinement_snapshot(
                    "direct_llm_baseline_no_refinement",
                    quality_report,
                    {},
                    {},
                    accepted=True,
                    best_score=self._quality_score(quality_report),
                )
            ],
        )
        self._write_no_fitting_report(paper.source_path)
        if self.config.output_pptx.suffix.lower() == ".html":
            return positioned
        if self.exporter is None:
            from .stages.pptx_exporter import PptxExporter

            self.exporter = PptxExporter(self.config.export, self.config.layout)
        self.exporter.export(positioned, self.config.output_pptx)
        return positioned

    def _forced_content_feedback(self, quality_report: dict) -> tuple[dict[str, str], dict[str, str]]:
        actions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for panel in quality_report.get("panels", []):
            issue_types = {issue.get("type") for issue in panel.get("issues", [])}
            issue_actions = {issue.get("action") for issue in panel.get("issues", [])}
            title = panel["title"]
            if "compress_text" in issue_actions:
                actions[title] = "compress"
                reasons[title] = "The quality check detected text overflow risk in this panel."
            elif {"excessive_blank_space", "pure_text_too_spacious"} & issue_types:
                actions[title] = "expand"
                reasons[title] = "The quality check detected excessive unused space in this panel."
            elif "expand_text" in issue_actions or "expand_text_and_decrease_panel_weight" in issue_actions:
                actions[title] = "expand"
                reasons[title] = "The quality check detected sparse panel content."
        return actions, reasons

    def _forced_content_actions(self, quality_report: dict) -> dict[str, str]:
        actions, _ = self._forced_content_feedback(quality_report)
        return actions

    def _refinement_snapshot(
        self,
        iteration: int | str,
        quality_report: dict,
        forced_actions: dict[str, str],
        forced_reasons: dict[str, str],
        accepted: bool = True,
        best_score: tuple[float, int, int, int, int] | None = None,
    ) -> dict:
        panels = quality_report.get("panels", [])
        occupancies = [panel.get("text_occupancy", 0.0) for panel in panels]
        visual_ratios = [panel.get("visual_area_ratio", 0.0) for panel in panels]
        return {
            "iteration": iteration,
            "issue_count": quality_report.get("summary", {}).get("issue_count", 0),
            "issue_counts": quality_report.get("summary", {}).get("issue_counts", {}),
            "average_text_occupancy": round(sum(occupancies) / len(occupancies), 3) if occupancies else 0.0,
            "average_visual_area_ratio": round(sum(visual_ratios) / len(visual_ratios), 3) if visual_ratios else 0.0,
            "forced_actions": forced_actions,
            "forced_reasons": forced_reasons,
            "accepted": accepted,
            "quality_score": list(self._quality_score(quality_report)),
            "best_quality_score": list(best_score) if best_score is not None else [],
        }

    def _write_refinement_history(self, source_path: Path, history: list[dict]) -> None:
        report_path = self.report_root / source_path.stem / "refinement_history.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"source_pdf": str(source_path), "iterations": history}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.warn(f"Could not write refinement history report: {exc}", RuntimeWarning, stacklevel=2)

    def _write_no_fitting_report(self, source_path: Path) -> None:
        report_path = self.report_root / source_path.stem / "content_fitting.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "source_pdf": str(source_path),
                        "enabled": False,
                        "reason": f"Disabled for experiment mode: {self.config.experiment_mode}",
                        "panels": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.warn(f"Could not write content fitting report: {exc}", RuntimeWarning, stacklevel=2)

    def _write_experiment_config(self, source_path: Path) -> None:
        report_path = self.report_root / source_path.stem / "experiment_config.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "source_pdf": str(source_path),
                        "experiment_mode": self.config.experiment_mode,
                        "llm_provider": self.config.llm.provider,
                        "llm_model": self.config.llm.model,
                        "layout_strategy": self.config.layout.strategy,
                        "asset_description_enabled": self.config.asset_selection.describe_assets_with_llm,
                        "asset_matching_use_llm": self.config.asset_matching.use_llm,
                        "semantic_matching_enabled": self.config.asset_matching.semantic_matching_enabled,
                        "caption_rewriting_enabled": self.config.asset_matching.rewrite_captions,
                        "content_fitting_enabled": self.config.content_fitting.enabled,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.warn(f"Could not write experiment config report: {exc}", RuntimeWarning, stacklevel=2)

    def _quality_score(self, quality_report: dict) -> tuple[float, int, int, int, int]:
        weights = {
            "out_of_bounds": 5.0,
            "text_overflow_risk": 3.0,
            "panel_too_small": 2.5,
            "image_too_small": 2.0,
            "table_too_small": 2.0,
            "excessive_blank_space": 1.0,
            "pure_text_too_spacious": 1.0,
            "visual_panel_unused_text_space": 0.8,
        }
        weighted = 0.0
        overflow = 0
        small_visual = 0
        blank = 0
        issue_count = 0
        for panel in quality_report.get("panels", []):
            for issue in panel.get("issues", []):
                issue_count += 1
                issue_type = issue.get("type", "")
                weighted += weights.get(issue_type, 1.0)
                if issue_type == "text_overflow_risk":
                    overflow += 1
                elif issue_type in {"image_too_small", "table_too_small"}:
                    small_visual += 1
                elif issue_type in {"excessive_blank_space", "pure_text_too_spacious", "visual_panel_unused_text_space"}:
                    blank += 1
        return (round(weighted, 3), issue_count, overflow, small_visual, blank)
