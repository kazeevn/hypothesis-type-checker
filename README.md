# Hypothesis Type Checker

Tools for fetching research papers, classifying their hypotheses with the OpenAI API, and comparing results between abstract-only and full-PDF analyses.

## Taxonomy Reference

Hypothesis taxonomy definitions and exemplar classifications live in `hypothesis_model.py`. The schema is based on Andrey Ustyuzhanin’s “Architecture of Inquiry” presentation: https://gamma.app/docs/The-Architecture-of-Inquiry-A-Comprehensive-Taxonomy-of-Scientifi-mqdpe9i68rtf6os

## Prerequisites

- Python 3.13 (managed automatically by [uv](https://docs.astral.sh/uv/))
- `uv` installed (follow the installation guide above)
- OpenAI API access with `OPENAI_API_KEY`

Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=your_api_key_here
```

## WANDB Logging

Runs launched through `classify_hypotheses.py` initialize [Weights & Biases](https://wandb.ai/) logging via `weave.init`. Authenticate once with `wandb login` and set the following keys in `.env` alongside the OpenAI credentials:

```bash
WANDB_PROJECT=hypothesis-type-checker
WANDB_ENTITY=your_team_or_username
```

Set `WANDB_MODE=offline` before executing a script if you prefer to disable remote logging for a run.

## Installation (uv-managed)

This project uses `uv` for dependency management and reproducible environments. From the repository root:

```bash
uv sync
```

`uv sync` creates `.venv/` and installs all dependencies declared in `pyproject.toml`. Use `uv run` for every script invocation so the correct environment and Python version are used automatically.

## Preparing Data

Classification expects two inputs under `data/`:
- `data/abstracts.json`: paper metadata with titles and abstracts
- `data/pdfs/`: PDFs named `paper_<number>.pdf`

You can populate both by sampling accepted papers from OpenReview with `fetch_papers.py`:

```bash
uv run fetch_papers.py ICML.cc/2025/Conference --num-papers 100 --output-dir data
```

The script saves metadata to `data/abstracts.json` and downloads PDFs into `data/pdfs/`. Adjust the venue, paper count, or output directory as needed.

## Running Hypothesis Classification

`classify_hypotheses.py` processes papers twice—once using abstracts and once using the full PDFs.

```bash
uv run classify_hypotheses.py \
  --model gpt-5-nano \
  --abstracts-path data/abstracts.json \
  --pdf-dir data/pdfs \
  --abstract-output data/classifications_abstract.json \
  --max-n-papers 100
```

Key options:
- `--model`: OpenAI model name (default `gpt-5-nano`)
- `--abstracts-path`: location of the metadata JSON (default `data/abstracts.json`)
- `--pdf-dir`: directory with PDFs (default `data/pdfs`)
- `--abstract-output`: output file for abstract-mode results (default `data/classifications_abstract.json`)
- `--max-n-papers`: limit the number of papers processed (optional)

Outputs (defaults):
- `data/classifications_abstract.json`
- `<pdf-dir>/../classifications_pdf.json`

Each output file contains per-paper entries with the hypotheses found, their classifications across the taxonomy axes (`epistemic_type`, `structural_type`, `predictive_type`, `functional_type`, `temporal_type`, `specific_type`), variables, justification, confidence score, and any processing notes. The PDF mode uploads each PDF to OpenAI, so expect longer runtimes.

## Comparing Abstract vs PDF Results

After classification, analyze overlaps and differences:

```bash
uv run compare_results.py
```

This script:
- Builds a combined dataframe of hypotheses across both modes
- Prints per-mode statistics and type distributions
- Computes per-paper overlaps and matching fractions
- Saves a JSON report to `data/comparison_summary.json`

## Example Schema Test

`test_examples.py` demonstrates that `Config.json_schema_extra["examples"]` is not used by the OpenAI model

```bash
uv run test_examples.py
```
