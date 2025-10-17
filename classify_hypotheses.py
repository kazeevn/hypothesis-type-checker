#!/usr/bin/env python3
"""
Script to classify hypotheses from downloaded papers using OpenAI API with structured outputs.

This script operates in two modes:
1. Abstract mode: Uses only paper title + abstract
2. PDF mode: Uploads PDF files to OpenAI API for analysis

Results are saved to data/ directory.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
import weave

from hypothesis_model import HypothesisClassification, EXAMPLE_CLASSIFICATIONS


CLASSIFICATION_ROLE = (
    "You are an expert in research methodology and hypothesis classification. "
    "Analyze the paper carefully to identify and classify all hypotheses."
)

PROMPT_SHARED_INSTRUCTIONS = """Your task is to identify ALL hypotheses (explicit or implicit) in this paper and classify each one according to the comprehensive taxonomy provided. At the same time, you must also avoid duplication, some hypotheses may be mentioned multiple times in different sections of the paper - but they should only be listed once in your output.

A hypothesis can be:
- An explicit statement about expected relationships or outcomes
- An implicit assumption being tested
- A research question that implies a testable prediction
- A claim about comparative performance of methods
- A prediction about transferability or generalization

For EACH hypothesis you identify, provide a complete classification including:
- The exact hypothesis text (quote it if possible)
- Classification along all axes (epistemic, structural, predictive, functional, temporal, specific)
- Justifications for your classifications
- Variables involved
- Confidence score

If you cannot find any clear hypotheses, return an empty list with an explanation in processing_notes."""

PROMPT_PDF_ADDITIONAL = """Read the ENTIRE paper provided as a PDF. Look throughout the document, including:
- Introduction section (research questions, objectives)
- Related work (comparative claims)
- Methods section (assumptions and predictions)
- Results section (tested hypotheses)
- Discussion/Conclusion (validated or rejected hypotheses)
"""

EXAMPLE_CLASSIFICATIONS_PROMPT = (
    "Example hypothesis classifications using the structured schema:\n"
    + "\n\n".join(
        json.dumps(example.model_dump(), indent=2)
        for example in EXAMPLE_CLASSIFICATIONS
    )
)


class HypothesesList(BaseModel):
    """Structured response schema for hypothesis extraction"""
    hypotheses: List[HypothesisClassification]
    processing_notes: str = ""


def build_prompt(title: str, mode: str, abstract: Optional[str] = None) -> str:
    """Assemble the user prompt for the requested mode"""
    header = [f"Paper Title: {title}"]

    if mode == "abstract" and abstract is not None:
        header.append("\nAbstract:\n" + abstract)
    elif mode == "pdf":
        header.append("\nThe full PDF of this paper is attached for analysis.")

    sections = ["\n".join(header), PROMPT_SHARED_INSTRUCTIONS]

    if mode == "pdf":
        sections.insert(1, PROMPT_PDF_ADDITIONAL)

    sections.append(EXAMPLE_CLASSIFICATIONS_PROMPT)

    return "\n\n".join(sections)


def load_abstracts(abstracts_path: Path) -> List[Dict[str, Any]]:
    """Load paper abstracts from JSON file"""
    with open(abstracts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def resolve_pdf_path(paper: Dict[str, Any], pdfs_dir: Path) -> Path:
    """Find the best-guess path to the local PDF for a paper"""
    paper_id = paper.get('id')
    paper_number = paper.get('number')
    local_pdf_path = paper.get('local_pdf_path')

    candidates: List[Path] = []

    if local_pdf_path:
        local_path = Path(local_pdf_path)
        candidates.append(local_path)
        if not local_path.is_absolute():
            candidates.append(Path.cwd() / local_path)
            candidates.append(pdfs_dir / local_path.name)

    if paper_id:
        candidates.append(pdfs_dir / f"{paper_id}.pdf")

    if paper_number:
        candidates.append(pdfs_dir / f"paper_{paper_number}.pdf")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if paper_id:
        return pdfs_dir / f"{paper_id}.pdf"

    if paper_number:
        return pdfs_dir / f"paper_{paper_number}.pdf"

    return pdfs_dir / "paper_unknown.pdf"


@weave.op()
def classify_paper(
    client: OpenAI,
    paper: Dict[str, Any],
    model: str,
    mode: str,
    pdf_path: Path | None = None,
) -> Dict[str, Any]:
    """Classify hypotheses for a paper in the requested mode"""
    paper_id = paper.get('id', 'unknown')
    title = paper.get('title', '')
    file_id: str | None = None

    if mode == "abstract":
        abstract_text = paper.get('abstract', '')
        prompt = build_prompt(title, mode, abstract=abstract_text)
        user_content: Any = prompt

    elif mode == "pdf":
        if not pdf_path or not pdf_path.exists():
            return {
                "paper_id": paper_id,
                "paper_title": title,
                "hypotheses": [],
                "source_mode": mode,
                "processing_notes": f"PDF file not found: {pdf_path}"
            }

        prompt = build_prompt(title, mode)

        print(f"Uploading PDF for paper {paper_id}...")
        try:
            with open(pdf_path, 'rb') as pdf_file:
                file_response = client.files.create(file=pdf_file, purpose='assistants')
            file_id = file_response.id
            print(f"PDF uploaded with file_id: {file_id}")
        except (OSError, OpenAIError) as exc:
            print(f"Error uploading PDF for paper {paper_id}: {exc}")
            return {
                "paper_id": paper_id,
                "paper_title": title,
                "hypotheses": [],
                "source_mode": mode,
                "processing_notes": f"Error: {exc}"
            }

        user_content = [
            {"role": "user",
            "content": [
                {
                "type": "input_text",
                "text": prompt
            },
            {
                "type": "input_file",
                    "file_id": file_id
            }]}
        ]

    else:
        raise ValueError(f"Unsupported classification mode: {mode}")

    try:
        parsed = client.responses.parse(
            model=model,
            instructions=CLASSIFICATION_ROLE,
            input=user_content,
            text_format=HypothesesList,
        ).output_parsed

        return {
            "paper_id": paper_id,
            "paper_title": title,
            "hypotheses": [hyp.model_dump() for hyp in parsed.hypotheses],
            "source_mode": mode,
            "processing_notes": parsed.processing_notes
        }

    except (OpenAIError, ValueError, TypeError) as exc:
        print(f"Error processing {mode} mode for paper {paper_id}: {exc}")
        return {
            "paper_id": paper_id,
            "paper_title": title,
            "hypotheses": [],
            "source_mode": mode,
            "processing_notes": f"Error: {exc}"
        }

    finally:
        if file_id:
            try:
                client.files.delete(file_id)
                print(f"Deleted uploaded file {file_id}")
            except OpenAIError as exc:
                print(f"Warning: Could not delete file {file_id}: {exc}")


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Classify hypotheses from research papers using OpenAI API"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-nano',
        help='OpenAI model to use for classification (default: gpt-5-nano)'
    )
    parser.add_argument(
        '--abstracts-path',
        type=Path,
        default='data/abstracts.json',
        help='Path to JSON file containing paper abstracts (default: data/abstracts.json)'
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default='data/pdfs',
        help='Directory containing paper PDFs (default: data/pdfs)'
    )
    parser.add_argument(
        '--abstract-output',
        type=Path,
        help='Output path for abstract-mode results (default: data/classifications_abstract.json)'
    )
    parser.add_argument(
        '--max-n-papers',
        type=int,
        help='Maximum number of papers to process'
    )

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    weave.init("hypothesis-type-checker")

    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    print(f"Using model: {args.model}")
    
    # Resolve paths from command-line options
    abstracts_path = Path(args.abstracts_path)
    pdfs_dir = Path(args.pdf_dir)
    if args.abstract_output:
        abstract_output = Path(args.abstract_output)
    else:
        abstract_output = abstracts_path.parent / 'classifications_abstract.json'
    pdf_output = pdfs_dir.parent / 'classifications_pdf.json'

    if not abstracts_path.exists():
        raise FileNotFoundError(f"Abstracts file not found: {abstracts_path}")
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdfs_dir}")
    if not pdfs_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdfs_dir}")

    abstract_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load papers
    papers = load_abstracts(abstracts_path)
    print(f"Loaded {len(papers)} papers")
    
    if args.max_n_papers:
        papers = papers[:args.max_n_papers]
        print(f"Limiting to first {args.max_n_papers} papers")

    # Mode 1: Classify from abstracts
    print("\n" + "="*60)
    print("MODE 1: Classifying hypotheses from abstracts...")
    print("="*60)
    
    abstract_results = []
    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('id', 'unknown')
        print(f"\n[{i}/{len(papers)}] Processing paper {paper_id}: {paper.get('title', 'N/A')[:60]}...")

        result = classify_paper(client, paper, args.model, mode="abstract")
        abstract_results.append(result)
        
        print(f"  Found {len(result['hypotheses'])} hypotheses")
    
    # Save abstract results
    with open(abstract_output, 'w', encoding='utf-8') as f:
        json.dump(abstract_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved abstract classifications to {abstract_output}")
    
    # Mode 2: Classify from PDFs
    print("\n" + "="*60)
    print("MODE 2: Classifying hypotheses from PDFs...")
    print("="*60)
    
    pdf_results = []
    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('id', 'unknown')

        # Find corresponding PDF
        pdf_path = resolve_pdf_path(paper, pdfs_dir)
        
        print(f"\n[{i}/{len(papers)}] Processing paper {paper_id} from PDF...")
        print(f"  Title: {paper.get('title', 'N/A')[:60]}...")
        print(f"  PDF: {pdf_path}")

        result = classify_paper(client, paper, args.model, mode="pdf", pdf_path=pdf_path)
        pdf_results.append(result)

        print(f"  Found {len(result['hypotheses'])} hypotheses")

    
    # Save PDF results
    with open(pdf_output, 'w', encoding='utf-8') as f:
        json.dump(pdf_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved PDF classifications to {pdf_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_hypotheses_abstract = sum(len(r['hypotheses']) for r in abstract_results)
    total_hypotheses_pdf = sum(len(r['hypotheses']) for r in pdf_results)
    
    print(f"Total papers processed: {len(papers)}")
    print(f"Hypotheses found (abstract mode): {total_hypotheses_abstract}")
    print(f"Hypotheses found (PDF mode): {total_hypotheses_pdf}")
    print("\nResults saved to:")
    print(f"  - {abstract_output}")
    print(f"  - {pdf_output}")


if __name__ == "__main__":
    main()
