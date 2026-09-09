from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace
from pathlib import Path

from ..config import LayoutConfig
from ..models import Panel, PosterLayout


class PanelQualityStage:
    """Rule-based commenter for panel-level layout problems."""

    def __init__(self, layout_config: LayoutConfig, report_dir: Path | None = None) -> None:
        self.layout_config = layout_config
        self.report_dir = report_dir
        self.last_report: dict = {}

    def analyze(self, layout: PosterLayout, source_path: Path | None = None) -> dict:
        panel_reports = [self._panel_report(panel) for panel in layout.panels]
        report = {
            "source_pdf": str(source_path) if source_path else "",
            "title": layout.title,
            "method": "rule_based_panel_commenter",
            "summary": self._summary(panel_reports),
            "panels": panel_reports,
        }
        self.last_report = report
        self._write_report(source_path, report)
        return report

    def refine(self, layout: PosterLayout, source_path: Path | None = None) -> PosterLayout:
        report = self.analyze(layout, source_path)
        refined = []
        for panel, panel_report in zip(layout.panels, report["panels"], strict=False):
            issues = {issue["type"] for issue in panel_report["issues"]}
            importance = panel.importance
            if "table_too_small" in issues:
                importance += 1.45
            if "image_too_small" in issues:
                importance += 1.3
            if "visual_panel_unused_text_space" in issues:
                importance += 0.45
            if "text_overflow_risk" in issues:
                importance += 0.7
            if (
                {"excessive_blank_space", "pure_text_too_spacious"} & issues
                and not (panel.figures or panel.tables)
                and "text_overflow_risk" not in issues
            ):
                importance = max(0.02, importance * 0.42)
            refined.append(replace(panel, importance=importance))
        return PosterLayout(title=layout.title, panels=refined)

    def has_actionable_issues(self, report: dict | None = None) -> bool:
        current = report or self.last_report
        return any(
            issue.get("action")
            in {
                "increase_panel_weight",
                "decrease_panel_weight",
                "compress_text",
                "expand_text",
                "expand_text_and_decrease_panel_weight",
            }
            for panel in current.get("panels", [])
            for issue in panel.get("issues", [])
        )

    def _panel_report(self, panel: Panel) -> dict:
        panel_w = max(0.01, panel.w * self.layout_config.page_width)
        panel_h = max(0.01, panel.h * self.layout_config.page_height)
        panel_area = panel_w * panel_h
        words = len(panel.body.split())
        capacity = self._estimated_word_capacity(panel, panel_w, panel_h)
        text_occupancy = words / capacity if capacity else 1.0
        visual = self._primary_visual(panel)
        visual_box = self._estimated_visual_box(panel, panel_w, panel_h, visual) if visual else None
        visual_area_ratio = (
            (visual_box["width"] * visual_box["height"]) / panel_area
            if visual_box and panel_area
            else 0.0
        )
        issues = self._issues(panel, panel_w, panel_h, text_occupancy, visual_box, visual_area_ratio)
        return {
            "title": panel.title,
            "x": round(panel.x, 4),
            "y": round(panel.y, 4),
            "w": round(panel.w, 4),
            "h": round(panel.h, 4),
            "physical_width_in": round(panel_w, 3),
            "physical_height_in": round(panel_h, 3),
            "word_count": words,
            "estimated_word_capacity": capacity,
            "text_occupancy": round(text_occupancy, 3),
            "visual_count": len(panel.figures) + len(panel.tables),
            "visual_box": visual_box,
            "visual_area_ratio": round(visual_area_ratio, 3),
            "issues": issues,
        }

    def _issues(
        self,
        panel: Panel,
        panel_w: float,
        panel_h: float,
        text_occupancy: float,
        visual_box: dict | None,
        visual_area_ratio: float,
    ) -> list[dict]:
        issues: list[dict] = []
        if panel.x < 0 or panel.y < 0 or panel.x + panel.w > 1 or panel.y + panel.h > 1:
            issues.append(self._issue("out_of_bounds", "high", "regenerate_layout"))
        if panel_w < 1.25 or panel_h < 1.0:
            issues.append(self._issue("panel_too_small", "medium", "increase_panel_weight"))
        if text_occupancy > 1.12:
            issues.append(self._issue("text_overflow_risk", "high", "compress_text"))
        blank_threshold = 0.68 if visual_box else 0.78
        if text_occupancy < blank_threshold and visual_area_ratio < 0.18:
            issues.append(self._issue("excessive_blank_space", "medium", "expand_text"))
        if visual_box and visual_area_ratio >= 0.28 and text_occupancy < 0.74 and panel_h >= 2.2:
            issues.append(self._issue("visual_panel_unused_text_space", "medium", "expand_text"))
        panel_area = panel_w * panel_h
        text_density = len(panel.body.split()) / max(0.01, panel_area)
        if not visual_box and panel_area > 9.0 and text_density < 9.5 and text_occupancy < 0.92:
            issues.append(self._issue("pure_text_too_spacious", "medium", "expand_text_and_decrease_panel_weight"))
        if visual_box:
            issue_type = "table_too_small" if panel.tables and not panel.figures else "image_too_small"
            if visual_box["width"] < 1.55 or visual_box["height"] < 0.78 or visual_area_ratio < 0.18:
                issues.append(self._issue(issue_type, "high", "increase_panel_weight"))
            elif visual_area_ratio < 0.24:
                issues.append(self._issue(issue_type, "medium", "increase_panel_weight"))
        return issues

    def _issue(self, issue_type: str, severity: str, action: str) -> dict:
        return {"type": issue_type, "severity": severity, "action": action}

    def _summary(self, panel_reports: list[dict]) -> dict:
        issue_counts: dict[str, int] = {}
        for panel in panel_reports:
            for issue in panel["issues"]:
                issue_counts[issue["type"]] = issue_counts.get(issue["type"], 0) + 1
        return {
            "panel_count": len(panel_reports),
            "issue_count": sum(issue_counts.values()),
            "issue_counts": issue_counts,
        }

    def _estimated_word_capacity(self, panel: Panel, panel_w: float, panel_h: float) -> int:
        has_visual = bool(self._primary_visual(panel))
        font_size = 9 if has_visual else 11
        text_w = max(0.3, panel_w - 0.32)
        if has_visual:
            ratio = 0.25 if len(panel.body.split()) <= 45 else 0.34
            text_h = max(0.42, min(panel_h * ratio, panel_h * 0.46))
        else:
            text_h = max(0.35, panel_h - 0.2)
        heading_h = (font_size + 3) * 1.3 / 72
        body_h = max(0.12, text_h - heading_h)
        chars_per_line = text_w * 72 / (font_size * 0.55)
        line_count = body_h * 72 / (font_size * 1.25)
        return max(12, min(int(chars_per_line * line_count / 6.3), 180))

    def _estimated_visual_box(self, panel: Panel, panel_w: float, panel_h: float, visual) -> dict:
        kind, aspect = visual
        text_words = len(panel.body.split())
        caption_h = 0.16 if panel_w >= 3.0 else 0.22
        if kind != "table" and self._side_by_side(panel, panel_w, panel_h, aspect):
            text_ratio = 0.42 if text_words <= 45 else 0.48 if text_words <= 75 else 0.54
            max_w = max(0.4, panel_w * (1 - text_ratio) - 0.34)
            max_h = max(0.45, panel_h - 0.22 - caption_h)
        else:
            text_ratio = 0.25 if text_words <= 45 else 0.34 if text_words <= 75 else 0.42
            text_h = max(0.42, min(panel_h * text_ratio, panel_h * 0.46))
            max_w = max(0.4, panel_w - 0.24)
            max_h = max(0.45, panel_h - 0.1 - text_h - 0.16 - 0.08 - caption_h)
        fitted_w, fitted_h = self._fit_box(max_w, max_h, aspect)
        return {"kind": kind, "width": round(fitted_w, 3), "height": round(fitted_h, 3)}

    def _side_by_side(self, panel: Panel, panel_w: float, panel_h: float, aspect: float) -> bool:
        panel_aspect = panel_w / max(0.01, panel_h)
        title_lines = max(1, math.ceil(len(panel.title) / 34))
        text_lines = title_lines + max(1, math.ceil(len(panel.body.split()) / max(10, int(panel_w * 3.6))))
        return panel_aspect >= 2.15 and panel_h >= 1.75 and aspect <= 3.2 and len(panel.body.split()) <= 75 and text_lines <= 7

    def _fit_box(self, max_w: float, max_h: float, aspect: float) -> tuple[float, float]:
        width = max_w
        height = width / max(0.1, aspect)
        if height > max_h:
            height = max_h
            width = height * aspect
        return width, height

    def _primary_visual(self, panel: Panel) -> tuple[str, float] | None:
        if panel.figures and panel.figures[0].path:
            return "figure", self._asset_aspect(panel.figures[0].path) or 1.4
        renderable_tables = [table for table in panel.tables if table.path and (not table.cells or len(table.cells) <= 20)]
        if renderable_tables:
            return "table", self._asset_aspect(renderable_tables[0].path) or 1.8
        return None

    def _asset_aspect(self, path: Path | None) -> float | None:
        if path is None or not path.is_file():
            return None
        try:
            from PIL import Image

            with Image.open(path) as image:
                return image.width / max(1, image.height)
        except (OSError, ModuleNotFoundError):
            return None

    def _write_report(self, source_path: Path | None, report: dict) -> None:
        if self.report_dir is None or source_path is None:
            return
        report_path = self.report_dir / source_path.stem / "panel_quality.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write panel quality report: {exc}", RuntimeWarning, stacklevel=2)
