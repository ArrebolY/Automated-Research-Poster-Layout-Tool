from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


def evaluate_report_dir(report_dir: Path) -> dict:
    """Build automatic module-level evaluation metrics from generated reports."""

    reports = {
        "parser": _read_json(report_dir / "parser_quality.json"),
        "section": _read_json(report_dir / "section_selection.json"),
        "asset_selection": _read_json(report_dir / "asset_selection.json"),
        "asset_matching": _read_json(report_dir / "asset_matching.json"),
        "layout": _read_json(report_dir / "layout_quality.json"),
        "panel": _read_json(report_dir / "panel_quality.json"),
        "content_fitting": _read_json(report_dir / "content_fitting.json"),
        "refinement": _read_json(report_dir / "refinement_history.json"),
        "experiment": _read_json(report_dir / "experiment_config.json"),
    }
    result = {
        "paper_id": report_dir.name,
        "source_pdf": _source_pdf(reports),
        "experiment_mode": reports["experiment"].get("experiment_mode", ""),
        "pdf_parsing": evaluate_pdf_parsing(reports["parser"]),
        "section_planning": evaluate_section_planning(reports["section"]),
        "asset_planning": evaluate_asset_planning(
            reports["asset_selection"],
            reports["asset_matching"],
            reports["section"],
        ),
        "layout_refinement": evaluate_layout_refinement(
            reports["panel"],
            reports["content_fitting"],
            reports["refinement"],
        ),
    }
    result["overall"] = _overall_summary(result)
    return result


def write_evaluation(report_dir: Path, output_path: Path | None = None) -> dict:
    evaluation = evaluate_report_dir(report_dir)
    target = output_path or report_dir / "module_eval.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation


def evaluate_pdf_parsing(report: dict) -> dict:
    figure_count = int(report.get("figure_count", 0) or 0)
    table_count = int(report.get("table_count", 0) or 0)
    total_assets = figure_count + table_count
    missing_files = report.get("missing_files", []) or []
    missing_captions = len(report.get("figures_without_captions", []) or []) + len(
        report.get("tables_without_captions", []) or []
    )
    caption_coverage = 1.0 if total_assets == 0 else (total_assets - missing_captions) / total_assets
    return {
        "backend": report.get("backend", ""),
        "title_present": bool(report.get("title_present")),
        "abstract_present": bool(report.get("abstract_present")),
        "section_count": int(report.get("section_count", 0) or 0),
        "suspicious_section_title_count": len(report.get("suspicious_section_titles", []) or []),
        "figure_count": figure_count,
        "table_count": table_count,
        "caption_coverage": round(caption_coverage, 3),
        "missing_file_count": len(missing_files),
        "warning_count": len(report.get("extraction_warnings", []) or []),
    }


def evaluate_section_planning(report: dict) -> dict:
    sections = report.get("selected_sections", []) or []
    categories = [section.get("effective_category") or section.get("category", "").lower() for section in sections]
    bullets = [
        bullet
        for section in sections
        for bullet in section.get("summary_bullets", []) or []
    ]
    source_numbers = {
        section.get("title", ""): _numbers(section.get("source_excerpt", ""))
        for section in sections
    }
    unsupported_numbers = 0
    for section in sections:
        allowed = source_numbers.get(section.get("title", ""), set())
        for bullet in section.get("summary_bullets", []) or []:
            unsupported_numbers += len(_numbers(bullet) - allowed)
    return {
        "selection_method": report.get("selection_method", ""),
        "selection_error": report.get("selection_error", ""),
        "parsed_section_count": int(report.get("parsed_section_count", 0) or 0),
        "candidate_section_count": int(report.get("candidate_section_count", 0) or 0),
        "selected_section_count": int(report.get("selected_section_count", len(sections)) or 0),
        "selected_categories": sorted({category for category in categories if category}),
        "core_category_coverage": _core_category_coverage(categories),
        "average_summary_word_count": round(mean([_word_count(bullet) for bullet in bullets]), 3) if bullets else 0.0,
        "duplicate_bullet_ratio": _duplicate_ratio(bullets),
        "unsupported_number_count": unsupported_numbers,
        "summary_error_count": len(report.get("summary_errors", []) or []),
        "grounding_warning_count": len(report.get("summary_grounding_warnings", []) or []),
    }


def evaluate_asset_planning(selection_report: dict, matching_report: dict, section_report: dict) -> dict:
    assets = selection_report.get("assets", []) or []
    selected_assets = [asset for asset in assets if asset.get("selected")]
    assignments = matching_report.get("assignments", []) or []
    assigned_keys = [(item.get("type", ""), item.get("id", "")) for item in assignments]
    duplicate_count = len(assigned_keys) - len(set(assigned_keys))
    selected_keys = {(item.get("type", ""), item.get("id", "")) for item in selected_assets}
    assigned_key_set = set(assigned_keys)
    selected_panel_count = int(section_report.get("selected_section_count", 0) or 0)
    visual_sections = {item.get("section", "") for item in assignments if item.get("section")}
    explicit_matches = sum(
        1 for item in assignments if _has_explicit_reference(item, selected_assets)
    )
    caption_missing = sum(1 for item in selected_assets if not str(item.get("caption", "")).strip())
    dense_selected = sum(1 for item in selected_assets if int(item.get("table_rows") or 0) > 20)
    return {
        "selection_method": selection_report.get("selection_method", ""),
        "matching_method": matching_report.get("matching_method", ""),
        "candidate_asset_count": int(selection_report.get("candidate_count", len(assets)) or 0),
        "selected_asset_count": len(selected_assets),
        "selected_figure_count": int(selection_report.get("selected_figure_count", 0) or 0),
        "selected_table_count": int(selection_report.get("selected_table_count", 0) or 0),
        "matched_asset_count": len(assignments),
        "unmatched_selected_asset_count": len(selected_keys - assigned_key_set),
        "duplicate_asset_count": duplicate_count,
        "visual_panel_ratio": round(len(visual_sections) / selected_panel_count, 3) if selected_panel_count else 0.0,
        "caption_missing_count": caption_missing,
        "dense_table_selected_count": dense_selected,
        "explicit_reference_match_rate": round(explicit_matches / len(assignments), 3) if assignments else 0.0,
    }


def evaluate_layout_refinement(panel_report: dict, fitting_report: dict, refinement_report: dict) -> dict:
    panels = panel_report.get("panels", []) or []
    issue_counts = panel_report.get("summary", {}).get("issue_counts", {}) or {}
    occupancies = [float(panel.get("text_occupancy", 0.0) or 0.0) for panel in panels]
    visual_ratios = [float(panel.get("visual_area_ratio", 0.0) or 0.0) for panel in panels]
    fitting_panels = fitting_report.get("panels", []) or []
    methods = [panel.get("method", "") for panel in fitting_panels]
    actions = [panel.get("action", "") for panel in fitting_panels]
    iterations = refinement_report.get("iterations", []) or []
    first_issue_count = iterations[0].get("issue_count", 0) if iterations else None
    final_issue_count = panel_report.get("summary", {}).get("issue_count", 0)
    return {
        "final_issue_count": final_issue_count,
        "issue_counts": issue_counts,
        "text_overflow_count": issue_counts.get("text_overflow_risk", 0),
        "blank_space_count": issue_counts.get("excessive_blank_space", 0)
        + issue_counts.get("pure_text_too_spacious", 0)
        + issue_counts.get("visual_panel_unused_text_space", 0),
        "small_visual_count": issue_counts.get("image_too_small", 0) + issue_counts.get("table_too_small", 0),
        "out_of_bounds_count": issue_counts.get("out_of_bounds", 0),
        "average_text_occupancy": round(mean(occupancies), 3) if occupancies else 0.0,
        "average_visual_area_ratio": round(mean(visual_ratios), 3) if visual_ratios else 0.0,
        "llm_text_action_count": sum(1 for method in methods if method in {"llm", "llm_trimmed"}),
        "rule_text_action_count": sum(1 for method in methods if method == "rule"),
        "expand_count": actions.count("expand"),
        "compress_count": actions.count("compress"),
        "iteration_count": len(iterations),
        "initial_issue_count": first_issue_count,
        "issue_reduction": first_issue_count - final_issue_count if first_issue_count is not None else None,
    }


def _overall_summary(evaluation: dict) -> dict:
    parser = evaluation["pdf_parsing"]
    section = evaluation["section_planning"]
    asset = evaluation["asset_planning"]
    layout = evaluation["layout_refinement"]
    return {
        "generation_ready": bool(
            parser["title_present"]
            and parser["section_count"] > 0
            and section["selected_section_count"] > 0
            and layout["out_of_bounds_count"] == 0
        ),
        "final_layout_issue_count": layout["final_issue_count"],
        "strict_llm_rule_text_actions": layout["rule_text_action_count"],
        "visual_panel_ratio": asset["visual_panel_ratio"],
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_pdf(reports: dict[str, dict]) -> str:
    for report in reports.values():
        if report.get("source_pdf"):
            return str(report["source_pdf"])
    return ""


def _core_category_coverage(categories: list[str]) -> dict:
    normalized = {category.lower() for category in categories}
    groups = {
        "introduction": {"introduction", "background", "motivation"},
        "method": {"method", "methodology", "framework", "model"},
        "results": {"results", "evaluation", "experiment", "experiments"},
        "conclusion": {"conclusion", "discussion"},
    }
    return {name: bool(normalized & aliases) for name, aliases in groups.items()}


def _has_explicit_reference(assignment: dict, selected_assets: list[dict]) -> bool:
    asset = next(
        (
            item
            for item in selected_assets
            if item.get("type") == assignment.get("type") and item.get("id") == assignment.get("id")
        ),
        {},
    )
    evidence = " ".join(asset.get("evidence", []) or [])
    return "explicitly referenced" in evidence.lower() and assignment.get("section", "") in evidence


def _duplicate_ratio(items: list[str]) -> float:
    if not items:
        return 0.0
    normalized = [re.sub(r"\s+", " ", item.lower()).strip() for item in items]
    return round((len(normalized) - len(set(normalized))) / len(normalized), 3)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,.]*\b", text))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", text))
