# Automated Research Poster Layout Tool

This repository contains a modular prototype for generating editable academic posters from research paper PDFs.

The current pipeline is:

```text
Paper PDF
-> PDF parsing
-> Section selection and summarization
-> Visual asset selection
-> Figure/table-section matching
-> Layout generation
-> Panel quality checking and refinement
-> Content fitting
-> PPTX poster export
```

The repository is intended to track the source code, tests, prompts, and lightweight validation data. Large local inputs, generated outputs, report PDFs, and experiment artifacts are ignored by Git.

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── src/postergen
│   ├── batch_eval.py
│   ├── cli.py
│   ├── config.py
│   ├── evaluation.py
│   ├── final_eval.py
│   ├── llm.py
│   ├── models.py
│   ├── parser_quality.py
│   ├── parser_validation.py
│   ├── pipeline.py
│   ├── prompts/
│   └── stages/
│       ├── asset_selection.py
│       ├── asset_structuring.py
│       ├── content_fitting.py
│       ├── content_understanding.py
│       ├── direct_llm_baseline.py
│       ├── docling_parser.py
│       ├── layout.py
│       ├── panel_quality.py
│       ├── parser.py
│       ├── pptx_exporter.py
│       └── refinement.py
├── tests/
└── validation/
    └── parser_gold.json
```

## Installation

Create a virtual environment and install the package:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e '.[dev]'
```

To enable the preferred Docling-based PDF parser:

```bash
pip install -e '.[docling]'
```

Without the Docling extra, the pipeline falls back to the legacy PyMuPDF/pdfplumber parser.

## Quick Start

Run the pipeline on a local PDF:

```bash
postergen path/to/paper.pdf --output outputs/poster.pptx
```

The same command can also be run as a Python module:

```bash
python -m postergen.cli path/to/paper.pdf --output outputs/poster.pptx
```

By default, the CLI uses `--llm-provider rule` and `--experiment-mode proposed`. This runs the modular pipeline with rule-based/offline fallbacks where no LLM is configured.

## LLM-backed Generation

The CLI supports `rule`, `openai`, `deepseek`, `openrouter`, and `gemini` providers.

Example with DeepSeek:

```bash
export DEEPSEEK_API_KEY=...
postergen path/to/paper.pdf \
  --output outputs/poster_deepseek.pptx \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

Example with OpenAI:

```bash
export OPENAI_API_KEY=...
postergen path/to/paper.pdf \
  --output outputs/poster_openai.pptx \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

For providers using a non-default environment variable or base URL:

```bash
postergen path/to/paper.pdf \
  --output outputs/poster_openrouter.pptx \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash \
  --llm-api-key-env OPENROUTER_API_KEY
```

## Experiment Modes

The main CLI supports the following experiment modes:

- `proposed`
- `direct_llm`
- `template_based`
- `without_semantic_figure_matching`
- `without_llm_figure_description`
- `without_panel_refinement`
- `template_layout_only`

Example:

```bash
postergen path/to/paper.pdf \
  --output outputs/template_based_poster.pptx \
  --experiment-mode template_based
```

For `direct_llm`, use an LLM provider. The output path may use `.html` if you want to save the direct HTML poster:

```bash
postergen path/to/paper.pdf \
  --output outputs/direct_llm_poster.html \
  --experiment-mode direct_llm \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

## Current Capabilities

The pipeline currently includes:

- Docling-based PDF parsing with fallback to the legacy PyMuPDF/pdfplumber parser
- extraction of paper title, abstract, sections, figures, tables, captions, and Markdown text
- parser quality reports written to `outputs/assets/<pdf_stem>/parser_quality.json`
- section selection and poster-oriented summarization
- visual asset description, filtering, and suitability scoring
- figure/table-section matching using explicit references and semantic signals
- caption rewriting for poster presentation when an LLM is configured
- recursive tree-based layout generation and template-based layout modes
- panel quality checking for overflow, blank space, small visual assets, and bounds issues
- local panel refinement and content fitting
- editable PPTX export with native text boxes, panel shapes, images, and captions
- direct LLM HTML baseline generation for comparison experiments
- module-level and final evaluation scripts

Intermediate reports are written under `outputs/assets/<pdf_stem>/`, including files such as:

- `paper.md`
- `experiment_config.json`
- `section_selection.json`
- `asset_selection.json`
- `asset_matching.json`
- `layout_quality.json`
- `panel_quality.json`
- `content_fitting.json`
- `refinement_history.json`
- `module_eval.json`

## Evaluation

Run module-level evaluation on existing reports:

```bash
postergen-evaluate --reports-root outputs/assets
```

Run generation and module-level evaluation for a directory or fixed list of PDFs:

```bash
postergen-evaluate \
  --run-pipeline \
  --paper-list path/to/paper_list.txt \
  --experiment-mode proposed \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

Run the final evaluation protocol:

```bash
postergen-final-evaluate \
  --paper-list path/to/paper_list.txt \
  --reports-root outputs/assets
```

The final evaluator supports PaperQuiz-style QA, faithfulness checks, rule-based visual readability checks, and optional OpenAI VLM-as-judge evaluation.

## Parser Validation

Run the lightweight parser validation checklist:

```bash
postergen-validate-parser
```

or:

```bash
python -m postergen.parser_validation
```

The checklist lives at `validation/parser_gold.json`. It is a small functional validation set, not a parser training dataset.

## Tests

After installing the development extra, run:

```bash
python -m pytest -q
```

## Notes

- `outputs/` is ignored because it contains generated posters, extracted assets, and evaluation artifacts.
- `materials/` is ignored because local paper PDFs and report materials can be large or private.
- Report PDFs, LaTeX report files, and local figure folders are not part of the published code repository.
