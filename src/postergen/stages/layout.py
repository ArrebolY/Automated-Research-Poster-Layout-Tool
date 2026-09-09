from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace
from pathlib import Path

from ..config import LayoutConfig
from ..models import Panel, PosterLayout, StructuredAsset


class LayoutGenerationStage:
    _CONTENT_Y = 0.12
    _CONTENT_H = 0.88
    _MIN_WIDTH = 0.34
    _MIN_HEIGHT = 0.16

    def __init__(self, config: LayoutConfig, report_dir: Path | None = None) -> None:
        self.config = config
        self.report_dir = report_dir

    def generate(
        self,
        title: str,
        assets: list[StructuredAsset],
        source_path: Path | None = None,
    ) -> PosterLayout:
        if not assets:
            return PosterLayout(title=title, panels=[])

        panels = [self._panel_from_asset(asset, assets) for asset in assets]
        positioned = self._position_panels(panels)
        self._write_report(source_path, title, positioned)
        return PosterLayout(title=title, panels=positioned)

    def relayout(self, layout: PosterLayout, source_path: Path | None = None) -> PosterLayout:
        if not layout.panels:
            return layout
        positioned = self._position_panels(layout.panels)
        self._write_report(source_path, layout.title, positioned)
        return PosterLayout(title=layout.title, panels=positioned)

    def _position_panels(self, panels: list[Panel]) -> list[Panel]:
        if self.config.strategy == "template":
            return self._template_grid(panels)
        return self._constrained_split(panels, 0.0, self._CONTENT_Y, 1.0, self._CONTENT_H)

    def _template_grid(self, panels: list[Panel]) -> list[Panel]:
        count = len(panels)
        if count == 1:
            return [replace(panels[0], x=0.0, y=self._CONTENT_Y, w=1.0, h=self._CONTENT_H)]
        columns = 2 if count > 1 else 1
        rows = math.ceil(count / columns)
        gap = self.config.gutter
        cell_w = (1.0 - gap * (columns - 1)) / columns
        cell_h = (self._CONTENT_H - gap * (rows - 1)) / rows
        positioned = []
        for index, panel in enumerate(panels):
            row = index // columns
            col = index % columns
            positioned.append(
                replace(
                    panel,
                    x=col * (cell_w + gap),
                    y=self._CONTENT_Y + row * (cell_h + gap),
                    w=cell_w,
                    h=cell_h,
                )
            )
        return positioned

    def _panel_from_asset(self, asset: StructuredAsset, all_assets: list[StructuredAsset]) -> Panel:
        text_lengths = [len(item.section.text.split()) for item in all_assets]
        visual_counts = [self._asset_visual_count(item) for item in all_assets]
        text_norm = self._normalize(len(asset.section.text.split()), text_lengths)
        visual_norm = self._normalize(self._asset_visual_count(asset), visual_counts)
        importance = self.config.alpha * text_norm + self.config.beta * visual_norm
        return Panel(
            title=asset.section.title,
            body=asset.section.text,
            figures=asset.figures,
            tables=asset.tables,
            importance=importance,
        )

    def _normalize(self, value: int, candidates: list[int]) -> float:
        if not candidates:
            return 0.0
        minimum = min(candidates)
        maximum = max(candidates)
        if maximum == minimum:
            return 1.0
        return (value - minimum) / (maximum - minimum)

    def _layout_weight(self, panel: Panel) -> float:
        return self._panel_area_weight(panel)

    def _panel_area_weight(self, panel: Panel) -> float:
        text_words = len(panel.body.split())
        text_component = text_words
        if not (panel.figures or self._has_renderable_table(panel)) and text_words < 160:
            text_component = min(text_words, 72)
        section_role = self._section_role(panel)
        visual_bonus = self._visual_area_bonus(panel)
        role_bonus = {
            "introduction": 12.0,
            "method": 18.0,
            "results": 16.0,
            "evaluation": 16.0,
            "dataset": 12.0,
            "conclusion": -8.0,
        }.get(section_role, 0.0)
        return max(1.0, text_component + visual_bonus + role_bonus + panel.importance * 45)

    def _visual_area_bonus(self, panel: Panel) -> float:
        figure_bonus = 0.0
        if panel.figures:
            figure = panel.figures[0]
            visual_aspect = self._asset_aspect(figure.path)
            if visual_aspect and visual_aspect > 1.6:
                figure_bonus = 84.0
            elif visual_aspect and visual_aspect < 0.75:
                figure_bonus = 72.0
            else:
                figure_bonus = 76.0
        table_bonus = 0.0
        for table in panel.tables:
            if table.cells and len(table.cells) > 20:
                table_bonus = max(table_bonus, 34.0)
            elif table.cells and len(table.cells) > 12:
                table_bonus = max(table_bonus, 58.0)
            else:
                table_bonus = max(table_bonus, 54.0)
        return figure_bonus + table_bonus

    def _preferred_aspect(self, panel: Panel) -> float:
        section_role = self._section_role(panel)
        visual_aspects = [
            aspect
            for aspect in [*(self._asset_aspect(figure.path) for figure in panel.figures), *(self._asset_aspect(table.path) for table in panel.tables)]
            if aspect is not None
        ]
        if visual_aspects:
            visual_aspect = visual_aspects[0]
            if panel.tables:
                return self._clamp(max(1.65, min(2.6, visual_aspect)), 1.4, 2.8)
            if visual_aspect > 1.6:
                return self._clamp(visual_aspect * 0.9, 1.6, 3.1)
            if visual_aspect < 0.75:
                return 1.0
            return 1.45
        if section_role == "conclusion":
            return 2.8
        if section_role == "introduction":
            return 2.4
        return 2.0 if len(panel.body.split()) < 65 else 1.6

    def _asset_visual_count(self, asset: StructuredAsset) -> int:
        renderable_tables = sum(1 for table in asset.tables if not table.cells or len(table.cells) <= 20)
        return len(asset.figures) + renderable_tables

    def _has_renderable_table(self, panel: Panel) -> bool:
        return any(not table.cells or len(table.cells) <= 20 for table in panel.tables)

    def _constrained_split(self, panels: list[Panel], x: float, y: float, w: float, h: float) -> list[Panel]:
        if len(panels) == 1:
            return [replace(panels[0], x=x, y=y, w=w, h=h)]

        total = sum(self._layout_weight(panel) for panel in panels) or 1.0
        gap = min(self.config.gutter, w if w >= h else h)
        can_split_vertically = max(0.0, w - gap) >= 2 * self._MIN_WIDTH
        can_split_horizontally = max(0.0, h - gap) >= 2 * self._MIN_HEIGHT
        candidates: list[tuple[float, list[Panel], list[Panel], str, float]] = []
        for pivot in range(1, len(panels)):
            first = panels[:pivot]
            second = panels[pivot:]
            first_ratio = (sum(self._layout_weight(panel) for panel in first) or 1.0) / total
            if can_split_vertically:
                available_w = max(0.0, w - gap)
                split_w = self._clamp(available_w * first_ratio, self._MIN_WIDTH, available_w - self._MIN_WIDTH)
                other_w = available_w - split_w
                loss = self._split_loss(first, split_w, h, first_ratio) + self._split_loss(second, other_w, h, 1 - first_ratio)
                loss += self._balance_penalty(first_ratio)
                loss += 0.02 if w < h else 0.0
                if w < 0.7:
                    loss += 2.0
                candidates.append((loss, first, second, "vertical", split_w))
            if can_split_horizontally:
                available_h = max(0.0, h - gap)
                min_h = min(self._MIN_HEIGHT, available_h / 2)
                split_h = self._clamp(available_h * first_ratio, min_h, available_h - min_h)
                other_h = available_h - split_h
                loss = self._split_loss(first, w, split_h, first_ratio) + self._split_loss(second, w, other_h, 1 - first_ratio)
                loss += self._balance_penalty(first_ratio)
                loss += 0.02 if w >= h else 0.0
                if w >= h and len(panels) > 3:
                    loss += 1.5
                candidates.append((loss, first, second, "horizontal", split_h))

        if not candidates:
            return [replace(panel, x=x, y=y, w=w, h=h / len(panels)) for panel in panels]

        _loss, first, second, orientation, split_size = min(candidates, key=lambda item: item[0])
        if orientation == "vertical":
            available_w = max(0.0, w - gap)
            other_w = available_w - split_size
            return self._constrained_split(first, x, y, split_size, h) + self._constrained_split(
                second,
                x + split_size + gap,
                y,
                other_w,
                h,
            )

        available_h = max(0.0, h - gap)
        other_h = available_h - split_size
        return self._constrained_split(first, x, y, w, split_size) + self._constrained_split(
            second,
            x,
            y + split_size + gap,
            w,
            other_h,
        )

    def _split_loss(self, panels: list[Panel], w: float, h: float, area_ratio: float) -> float:
        actual_aspect = self._physical_aspect(w, h)
        target_aspect = self._group_preferred_aspect(panels)
        aspect_loss = abs(math.log(max(0.2, actual_aspect) / max(0.2, target_aspect)))
        if len(panels) == 1 and (panels[0].figures or self._has_renderable_table(panels[0])):
            aspect_loss *= 1.8
        size_penalty = 0.0
        if w < self._MIN_WIDTH:
            size_penalty += (self._MIN_WIDTH - w) * 8.0
        if h < self._MIN_HEIGHT:
            size_penalty += (self._MIN_HEIGHT - h) * 8.0
        if len(panels) == 1:
            size_penalty += self._single_panel_sparse_penalty(panels[0], w, h)
        return aspect_loss + size_penalty + abs(area_ratio - 0.5) * 0.08

    def _group_preferred_aspect(self, panels: list[Panel]) -> float:
        total = sum(self._layout_weight(panel) for panel in panels) or 1.0
        weighted = sum(self._preferred_aspect(panel) * self._layout_weight(panel) for panel in panels)
        return self._clamp(weighted / total, 0.65, 3.2)

    def _single_panel_sparse_penalty(self, panel: Panel, w: float, h: float) -> float:
        area = w * h
        words = len(panel.body.split())
        has_visual = bool(panel.figures or self._has_renderable_table(panel))
        if has_visual:
            return 0.0
        if words < 55 and area > 0.16:
            return (area - 0.16) * 2.5
        return 0.0

    def _balance_penalty(self, ratio: float) -> float:
        return abs(ratio - 0.5) * 0.12

    def _physical_aspect(self, w: float, h: float) -> float:
        return (w * self.config.page_width) / max(0.01, h * self.config.page_height)

    def _section_role(self, panel: Panel) -> str:
        title = panel.title.lower()
        text = panel.body.lower()
        if any(term in title for term in ("introduction", "background", "motivation")):
            return "introduction"
        if any(term in title for term in ("result", "analysis", "experiment", "finding", "detection", "prediction")):
            return "results"
        if any(term in title for term in ("evaluation", "benchmark", "metric")):
            return "evaluation"
        if any(term in title for term in ("dataset", "data set", "example")):
            return "dataset"
        if any(term in title for term in ("method", "model", "framework", "architecture", "system")):
            return "method"
        if "conclusion" in title:
            return "conclusion"
        if any(term in text for term in ("experiment", "outperform", "accuracy", "auroc", "result")):
            return "results"
        return "generic"

    def _asset_aspect(self, path) -> float | None:
        if path is None or not path.is_file():
            return None
        try:
            from PIL import Image

            with Image.open(path) as image:
                return image.width / max(1, image.height)
        except (OSError, ModuleNotFoundError):
            return None

    def _layout_quality(self, panel: Panel) -> dict:
        area = panel.w * panel.h
        words = len(panel.body.split())
        visual_count = len(panel.figures) + len(panel.tables)
        aspect = self._physical_aspect(panel.w, panel.h)
        preferred = self._preferred_aspect(panel)
        density = words / max(0.01, area)
        risk = "ok"
        if not visual_count and words < 55 and area > 0.14:
            risk = "too_sparse"
        elif density > 720:
            risk = "too_dense"
        elif abs(math.log(max(0.2, aspect) / max(0.2, preferred))) > 0.8:
            risk = "shape_mismatch"
        return {
            "title": panel.title,
            "role": self._section_role(panel),
            "x": round(panel.x, 4),
            "y": round(panel.y, 4),
            "w": round(panel.w, 4),
            "h": round(panel.h, 4),
            "area": round(area, 4),
            "area_weight": round(self._panel_area_weight(panel), 3),
            "word_count": words,
            "visual_count": visual_count,
            "aspect_ratio": round(aspect, 3),
            "preferred_aspect_ratio": round(preferred, 3),
            "density": round(density, 3),
            "risk": risk,
        }

    def _write_report(self, source_path: Path | None, title: str, panels: list[Panel]) -> None:
        if self.report_dir is None or source_path is None:
            return
        report = {
            "source_pdf": str(source_path),
            "title": title,
            "layout_method": "fixed_template_grid" if self.config.strategy == "template" else "heuristic_sp_rp_loss_tree_split",
            "panel_count": len(panels),
            "panels": [self._layout_quality(panel) for panel in panels],
        }
        report_path = self.report_dir / source_path.stem / "layout_quality.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write layout quality report: {exc}", RuntimeWarning, stacklevel=2)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        if maximum < minimum:
            return max(0.0, value)
        return max(minimum, min(value, maximum))
