from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LayoutConfig:
    strategy: str = "tree"
    alpha: float = 0.7
    beta: float = 0.3
    page_width: float = 13.333
    page_height: float = 7.5
    margin: float = 0.008
    gutter: float = 0.004


@dataclass(slots=True)
class ContentConfig:
    max_sections: int = 5
    summary_sentence_range: tuple[int, int] = (3, 5)
    selection_excerpt_chars: int = 1200
    max_section_chars: int = 6000
    min_section_words: int = 50
    min_standalone_section_words: int = 15
    min_summary_grounding_ratio: float = 0.25
    require_llm: bool = False


@dataclass(slots=True)
class ContentFittingConfig:
    enabled: bool = True
    sparse_ratio: float = 0.7
    dense_ratio: float = 1.05
    target_ratio: float = 0.84
    target_min_ratio: float = 0.72
    target_max_ratio: float = 0.96
    max_iterations: int = 2
    min_bullets: int = 2
    max_bullets: int = 6
    max_source_chars: int = 5000
    min_grounding_ratio: float = 0.2
    require_llm: bool = False


@dataclass(slots=True)
class AssetSelectionConfig:
    max_total_assets: int = 5
    max_figures: int = 5
    max_tables: int = 3
    max_dense_tables: int = 1
    section_excerpt_chars: int = 900
    describe_assets_with_llm: bool = False
    require_llm: bool = False


@dataclass(slots=True)
class AssetMatchingConfig:
    use_llm: bool = True
    semantic_matching_enabled: bool = True
    rewrite_captions: bool = False
    min_semantic_overlap: int = 2
    min_semantic_score: float = 4.0
    max_figures_per_section: int = 1
    max_tables_per_section: int = 1
    require_llm: bool = False


@dataclass(slots=True)
class LLMConfig:
    provider: str = "rule"
    model: str = ""
    api_key_env: str = ""
    base_url: str = ""
    temperature: float = 0.2
    max_output_tokens: int = 500


@dataclass(slots=True)
class ExportConfig:
    theme_name: str = "academic"
    title_font_size_pt: int = 25
    body_font_size_pt: int = 14
    caption_font_size_pt: int = 10


@dataclass(slots=True)
class PipelineConfig:
    input_pdf: Path
    output_pptx: Path
    report_root: Path = Path("outputs/assets")
    experiment_mode: str = "proposed"
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    content_fitting: ContentFittingConfig = field(default_factory=ContentFittingConfig)
    asset_selection: AssetSelectionConfig = field(default_factory=AssetSelectionConfig)
    asset_matching: AssetMatchingConfig = field(default_factory=AssetMatchingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
