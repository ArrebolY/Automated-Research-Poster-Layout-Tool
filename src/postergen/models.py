from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Section:
    title: str
    text: str
    level: int = 1
    category: str = ""


@dataclass(slots=True)
class Figure:
    figure_id: str
    caption: str
    summary: str = ""
    path: Path | None = None
    referenced_in: list[str] = field(default_factory=list)
    match_score: float = 0.0
    match_method: str = ""

    @property
    def representation(self) -> str:
        return " ".join(part for part in (self.caption, self.summary) if part).strip()


@dataclass(slots=True)
class Table:
    table_id: str
    caption: str
    cells: list[list[str]] = field(default_factory=list)
    summary: str = ""
    referenced_in: list[str] = field(default_factory=list)
    path: Path | None = None
    match_score: float = 0.0
    match_method: str = ""

    @property
    def representation(self) -> str:
        return " ".join(part for part in (self.caption, self.summary) if part).strip()


@dataclass(slots=True)
class PaperDocument:
    title: str
    source_path: Path
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


@dataclass(slots=True)
class SectionSummary:
    title: str
    summary_sentences: list[str]
    source_text: str
    key_facts: dict[str, list[str]] = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.summary_sentences and all(sentence.startswith("- ") for sentence in self.summary_sentences):
            return "\n".join(self.summary_sentences)
        return " ".join(self.summary_sentences)

    @property
    def token_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class StructuredAsset:
    section: SectionSummary
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


@dataclass(slots=True)
class Panel:
    title: str
    body: str
    figures: list[Figure]
    importance: float
    tables: list[Table] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0


@dataclass(slots=True)
class PosterLayout:
    title: str
    panels: list[Panel]
