# Automated Research Poster Layout Tool

This repository contains a modular prototype for automatic research poster generation.

The system follows the pipeline below:

`Paper PDF -> Content Understanding -> Asset Structuring -> Layout Generation -> Panel Refinement -> PPTX Poster`

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── src/postergen
│   ├── cli.py
│   ├── config.py
│   ├── llm.py
│   ├── models.py
│   ├── pipeline.py
│   └── stages
│       ├── asset_structuring.py
│       ├── content_understanding.py
│       ├── layout.py
│       ├── parser.py
│       ├── pptx_exporter.py
│       └── refinement.py
└── tests
    ├── test_asset_structuring.py
    ├── test_cli.py
    ├── test_content_understanding.py
    ├── test_layout.py
    └── test_parser.py
```

## Quick Start

1. Create a virtual environment and install dependencies.
   Install the Docling extra to enable the preferred PDF parser:

```bash
pip install -e '.[docling]'
```

Without this extra, the pipeline automatically uses the legacy PyMuPDF parser.

2. Run the pipeline:

```bash
python -m postergen.cli materials/paper/P2P.pdf --output outputs/poster.pptx
```

This default command uses the offline rule-based baseline. It does not require an API key.

To run with an LLM-backed summarizer, set the provider API key and pass the provider/model:

```bash
export DEEPSEEK_API_KEY=...
python -m postergen.cli materials/paper/P2P.pdf \
  --output outputs/poster_deepseek.pptx \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

```bash
export OPENAI_API_KEY=...
python -m postergen.cli materials/paper/P2P.pdf \
  --output outputs/poster_openai.pptx \
  --llm-provider openai \
  --llm-model gpt-5.4-mini
```

```bash
export GEMINI_API_KEY=...
python -m postergen.cli materials/paper/P2P.pdf \
  --output outputs/poster_gemini.pptx \
  --llm-provider gemini \
  --llm-model gemini-3.1-flash-lite
```

OpenAI-compatible proxy providers can be used through OpenRouter:

```bash
export OPENROUTER_API_KEY=...
python -m postergen.cli materials/paper/P2P.pdf \
  --output outputs/poster_openrouter.pptx \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash
```

## Current Capabilities

The pipeline currently includes:

- data models for paper, sections, figures, panels, and layout
- PDF text, section, caption, and image extraction with `PyMuPDF`
- table detection and table-cell extraction with `pdfplumber`
- offline rule-based section summarization
- optional LLM-backed section selection and section summarization
- figure/table-section alignment by explicit references and keyword fallback
- recursive binary layout generation
- editable PPTX export with panels, bullet text, images, figure captions, and table captions

The active parser uses Docling's `DocumentConverter` and `PdfPipelineOptions`, exports the parsed
document to Markdown, and saves figure and table images under `outputs/assets/<pdf_stem>/`. A
`parser_quality.json` report is written in the same folder. If Docling is unavailable or conversion
fails, the existing PyMuPDF parser is used automatically and the fallback reason appears in the report.

After section selection and summarisation, the asset-selection stage filters the extracted figures and
tables before section matching. It uses explicit references, caption relevance, and asset availability
for the offline rule baseline, or the configured LLM when available. The selected IDs, scores, reasons,
and rejected candidates are recorded in `outputs/assets/<pdf_stem>/asset_selection.json`.

Section selection rejects short overview headings when substantive sections are available, preserves
coverage of the problem, core method, main results, and conclusion, and checks generated bullets for
unsupported numbers or weak source grounding. These checks and any automatic adjustments are recorded
in `section_selection.json`. Figure/table matching ignores caption-only references, applies semantic
relevance thresholds and role compatibility, and records assignments in `asset_matching.json`.

When an LLM is configured, section selection returns structured section IDs, poster roles, and reasons.
Asset matching also uses a structured LLM planner that assigns selected figures/tables to section IDs;
all IDs, duplicate use, per-section limits, explicit references, and semantic relevance are validated
before the assignment is accepted. The rule matcher remains the offline baseline and automatic fallback.
The versioned prompt templates live under `src/postergen/prompts/`.

## PDF Parser Validation

Run the lightweight validation checklist on the representative research papers:

```bash
python -m postergen.parser_validation
```

The report includes normalized title accuracy, section precision/recall/F1, section-level accuracy,
required-body sanity checks, figure/table recall, and caption completeness. The checklist lives at
`validation/parser_gold.json`; it is a small functional validation set, not a parser training dataset.

## Planned Extensions

- add figure image summarization for multimodal LLMs
- improve image-caption binding using PDF layout coordinates
- add embedding-based semantic figure-section matching
- add evaluation scripts for PaperQuiz, fidelity scoring, and VLM-based judging
