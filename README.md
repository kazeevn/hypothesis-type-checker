# Hypothesis Classification Guide

This guide explains how to use the OpenAI API to classify hypotheses from research papers.

## Prerequisites

1. **Environment Setup**: Ensure you have a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Data**: You need:
   - `data/abstracts.json` - Paper metadata with titles and abstracts
   - `data/pdfs/` - Directory containing PDF files named `paper_<number>.pdf`

## Running the Classification

The main script `classify_hypotheses.py` runs in two modes:

### Mode 1: Abstract-based Classification
Uses only the paper title and abstract to identify and classify hypotheses.

### Mode 2: PDF-based Classification
Uploads the full PDF to OpenAI API and analyzes the entire paper for hypotheses.

### Usage

```bash
# Make sure you're in the virtual environment
cd /home/kna/hypothesis-type-checker

# Run the classification script with default model (gpt-5-nano)
.venv/bin/python classify_hypotheses.py

# Or specify a different model
.venv/bin/python classify_hypotheses.py --model gpt-4o-2024-08-06

# Override data locations (example)
.venv/bin/python classify_hypotheses.py \
   --abstracts-path custom_data/abstracts.json \
   --pdf-dir custom_data/pdfs \
   --abstract-output results/abstract_classifications.json

# See all options
.venv/bin/python classify_hypotheses.py --help
```

The script will (using default paths):
1. Process all papers in abstract mode first
2. Save results to `data/classifications_abstract.json` (customizable via `--abstract-output`)
3. Process all papers in PDF mode
4. Save results to `classifications_pdf.json` in the parent directory of `--pdf-dir`

### Output Format

Each result file contains an array of paper objects with:
- `paper_id`: Unique identifier
- `paper_title`: Paper title
- `hypotheses`: Array of classified hypotheses
- `source_mode`: Either "abstract" or "pdf"
- `processing_notes`: Any errors or notes

Each hypothesis includes:
- `hypothesis_text`: The actual hypothesis text
- Classification across multiple axes:
  - `epistemic_type`: descriptive/associative/causal
  - `structural_type`: simple/complex
  - `predictive_type`: directional/non_directional
  - `functional_type`: scientific/statistical/working
  - `temporal_type`: exploratory/confirmatory
  - `specific_type`: comparative_performance/transferability/implementation/other
- `variables_identified`: List of variables
- `confidence_score`: 0-1 confidence rating
- Justifications for classifications

## Comparing Results

After running the classification, you can analyze and compare the results:

```bash
.venv/bin/python compare_results.py
```

This will:
1. Analyze each mode's results separately
2. Compare hypotheses found in abstract vs PDF mode
3. Generate statistics and distributions
4. Save comparison to `data/comparison_summary.json`

## API Details

### OpenAI PDF Processing

Following the OpenAI documentation at https://platform.openai.com/docs/guides/pdf-files, the script:

1. **Uploads PDF**: Uses the Files API to upload PDFs
   ```python
   file_response = client.files.create(
       file=pdf_file,
       purpose='assistants'
   )
   ```

2. **References in Chat**: Includes the file in the message content
   ```python
   messages=[{
       "role": "user",
       "content": [
           {"type": "text", "text": prompt},
           {"type": "file", "file": {"file_id": file_id}}
       ]
   }]
   ```

3. **Structured Outputs**: Uses `response_format` with Pydantic models
   ```python
   completion = client.beta.chat.completions.parse(
       model=model,  # Configurable via --model argument, default: gpt-5-nano
       messages=messages,
       response_format=HypothesesList,
   )
   ```

4. **Cleanup**: Deletes uploaded files after processing

## Rate Limiting

The script includes delays between API calls:
- 1 second between abstract-mode requests
- 2 seconds between PDF-mode requests

Adjust these if you hit rate limits or want faster processing.

## Cost Considerations

- **Abstract mode**: Uses text-only API calls (cheaper)
- **PDF mode**: Uploads and processes entire PDFs (more expensive, especially for large PDFs)

Monitor your OpenAI API usage dashboard when running on many papers.

## Troubleshooting

### Missing API Key
```
ValueError: OPENAI_API_KEY not found in environment variables
```
**Solution**: Create/update `.env` file with your API key

### PDF Not Found
Check that PDF files are named correctly: `paper_<number>.pdf` matching the `number` field in `abstracts.json`

### API Errors
- Rate limiting: Increase delays in the script
- Token limits: PDFs might be too large; consider splitting or summarizing
- Invalid model: Ensure you're using `gpt-4o-2024-08-06` or compatible model

## Model Information

The hypothesis classification uses the taxonomy from Andrey Ustyuzhanin's presentation:
https://gamma.app/docs/The-Architecture-of-Inquiry-A-Comprehensive-Taxonomy-of-Scientifi-mqdpe9i68rtf6os

All classification categories and examples are defined in `hypothesis_model.py`.
