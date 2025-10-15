#!/usr/bin/env python3
"""
Script to compare and analyze classification results from abstract vs PDF modes.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load classification results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


TYPE_FIELDS = (
    'epistemic_type',
    'structural_type',
    'predictive_type',
    'functional_type',
    'temporal_type',
    'specific_type',
)


def results_to_dataframe(results: List[Dict[str, Any]], mode: str) -> pd.DataFrame:
    """Flatten hypothesis data into a dataframe for the given mode."""
    rows: List[Dict[str, Any]] = []

    for paper in results:
        hypotheses: Iterable[Dict[str, Any]] = paper.get('hypotheses', [])
        for idx, hyp in enumerate(hypotheses):
            row = {
                'mode': mode,
                'paper_id': paper.get('paper_id'),
                'paper_title': paper.get('paper_title'),
                'hypothesis_index': idx,
            }
            row.update({field: hyp.get(field) for field in TYPE_FIELDS})
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_hypotheses(results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Analyze hypothesis classifications"""
    total_papers = len(results)
    total_hypotheses = sum(len(paper['hypotheses']) for paper in results)
    
    papers_with_hypotheses = sum(1 for paper in results if paper['hypotheses'])
    papers_without_hypotheses = total_papers - papers_with_hypotheses
    
    # Count by classification types
    type_counters = {field: Counter() for field in TYPE_FIELDS}

    for paper in results:
        for hyp in paper['hypotheses']:
            for field in TYPE_FIELDS:
                type_counters[field][hyp[field]] += 1
    
    return {
        'mode': mode,
        'total_papers': total_papers,
        'total_hypotheses': total_hypotheses,
        'avg_hypotheses_per_paper': total_hypotheses / total_papers if total_papers > 0 else 0,
        'papers_with_hypotheses': papers_with_hypotheses,
        'papers_without_hypotheses': papers_without_hypotheses,
        'type_distributions': {field: dict(counter) for field, counter in type_counters.items()},
    }


def compare_paper_results(abstract_results: List[Dict], pdf_results: List[Dict]) -> List[Dict[str, Any]]:
    """Compare results for the same papers across both modes"""
    comparisons = []
    
    # Create lookup by paper_id
    abstract_by_id = {paper['paper_id']: paper for paper in abstract_results}
    pdf_by_id = {paper['paper_id']: paper for paper in pdf_results}
    
    all_ids = set(abstract_by_id.keys()) | set(pdf_by_id.keys())
    
    for paper_id in all_ids:
        abstract_paper = abstract_by_id.get(paper_id)
        pdf_paper = pdf_by_id.get(paper_id)
        
        comparison = {
            'paper_id': paper_id,
            'title': (abstract_paper or pdf_paper)['paper_title'],
            'abstract_hypotheses_count': len(abstract_paper['hypotheses']) if abstract_paper else 0,
            'pdf_hypotheses_count': len(pdf_paper['hypotheses']) if pdf_paper else 0,
            'difference': (len(pdf_paper['hypotheses']) if pdf_paper else 0) - 
                         (len(abstract_paper['hypotheses']) if abstract_paper else 0),
        }
        
        comparisons.append(comparison)
    
    return comparisons


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print analysis results"""
    print(f"\n{'='*60}")
    print(f"Analysis for {analysis['mode'].upper()} mode")
    print(f"{'='*60}")
    print(f"Total papers: {analysis['total_papers']}")
    print(f"Total hypotheses found: {analysis['total_hypotheses']}")
    print(f"Average hypotheses per paper: {analysis['avg_hypotheses_per_paper']:.2f}")
    print(f"Papers with hypotheses: {analysis['papers_with_hypotheses']}")
    print(f"Papers without hypotheses: {analysis['papers_without_hypotheses']}")


def print_comparative_type_statistics(df: pd.DataFrame):
    """Print comparative statistics across modes for each hypothesis type field."""
    if df.empty:
        print("\nNo hypothesis data available for comparative statistics.")
        return

    print(f"\n{'='*60}")
    print("Comparative Type Distributions (counts by mode)")
    print(f"{'='*60}")

    for field in TYPE_FIELDS:
        pivot = (
            df.pivot_table(
                index=field,
                columns='mode',
                values='hypothesis_index',
                aggfunc='count',
                fill_value=0,
            )
            .sort_index()
        )
        print(f"\n{field.replace('_', ' ').title()}:")
        print(pivot)


def plot_comparative_type_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Render grouped bar charts comparing type distributions across modes."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, field in zip(axes, TYPE_FIELDS):
        pivot = (
            df.pivot_table(
                index=field,
                columns='mode',
                values='hypothesis_index',
                aggfunc='count',
                fill_value=0,
            )
            .sort_index()
        )
        if pivot.empty:
            ax.axis('off')
            continue

        pivot.plot(kind='bar', ax=ax, xlabel='')
        ax.set_title(field.replace('_', ' ').title())
        # ax.set_xlabel('Classification')
        ax.set_ylabel('Hypothesis count')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.legend(title='Mode')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

    for idx in range(len(TYPE_FIELDS), len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _hypothesis_signature(hypothesis: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(hypothesis.get(field) for field in TYPE_FIELDS)


def _build_hypothesis_set(paper: Dict[str, Any]) -> set:
    return frozenset((_hypothesis_signature(hyp) for hyp in paper.get('hypotheses', [])))


def compute_matching_fractions(
    abstract_results: List[Dict[str, Any]],
    pdf_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Compute per-paper matching fractions between abstract and PDF hypotheses."""

    abstract_by_id = {paper['paper_id']: paper for paper in abstract_results}
    pdf_by_id = {paper['paper_id']: paper for paper in pdf_results}
    all_ids = sorted(set(abstract_by_id) | set(pdf_by_id))

    rows: List[Dict[str, Any]] = []
    for paper_id in all_ids:
        abstract_paper = abstract_by_id.get(paper_id, {})
        pdf_paper = pdf_by_id.get(paper_id, {})

        abstract_set = _build_hypothesis_set(abstract_paper)
        pdf_set = _build_hypothesis_set(pdf_paper)
        intersection = abstract_set & pdf_set
        denominator = len(abstract_set)
        matching_fraction = (len(intersection) / denominator) if denominator else 0.0

        rows.append({
            'paper_id': paper_id,
            'paper_title': (abstract_paper or pdf_paper).get('paper_title'),
            'abstract_hypotheses': len(abstract_set),
            'pdf_hypotheses': len(pdf_set),
            'matching_hypotheses': len(intersection),
            'matching_fraction': matching_fraction,
        })

    return pd.DataFrame(rows)


def main():
    """Main execution function"""
    data_dir = Path('data')
    
    # Load results
    abstract_results = load_results(data_dir / 'classifications_abstract.json')
    pdf_results = load_results(data_dir / 'classifications_pdf.json')

    # Build dataframe with all hypotheses
    abstract_df = results_to_dataframe(abstract_results, 'abstract')
    pdf_df = results_to_dataframe(pdf_results, 'pdf')
    all_hypotheses_df = pd.concat([abstract_df, pdf_df], ignore_index=True)

    print(f"\n{'='*60}")
    print("Combined Hypotheses DataFrame")
    print(f"{'='*60}")
    print(all_hypotheses_df)
    
    # Analyze each mode
    abstract_analysis = analyze_hypotheses(abstract_results, 'abstract')
    pdf_analysis = analyze_hypotheses(pdf_results, 'pdf')
    
    # Print analyses
    print_analysis(abstract_analysis)
    print_analysis(pdf_analysis)
    
    # Comparative statistics
    print_comparative_type_statistics(all_hypotheses_df)

    plot_path = data_dir / 'comparative_type_distributions.webp'
    plot_comparative_type_distributions(all_hypotheses_df, plot_path)

    # Matching fractions
    matching_df = compute_matching_fractions(abstract_results, pdf_results)

    print(f"\n{'='*60}")
    print("Per-Paper Matching Fractions")
    print(f"{'='*60}")
    print(matching_df)

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON: Abstract vs PDF")
    print(f"{'='*60}")
    
    comparisons = compare_paper_results(abstract_results, pdf_results)
    
    # Summary statistics
    total_diff = sum(comp['difference'] for comp in comparisons)
    avg_diff = total_diff / len(comparisons) if comparisons else 0
    avg_matching_fraction = matching_df['matching_fraction'].mean() if not matching_df.empty else 0.0
    
    print("\nSummary:")
    print(f"Total additional hypotheses found in PDF mode: {total_diff}")
    print(f"Average additional hypotheses per paper (PDF vs Abstract): {avg_diff:.2f}")
    print(f"Average matching fraction across papers: {avg_matching_fraction:.2f}")
    
    # Save comparison
    comparison_output = data_dir / 'comparison_summary.json'
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump({
            'abstract_analysis': abstract_analysis,
            'pdf_analysis': pdf_analysis,
            'per_paper_comparison': comparisons,
            'matching_fractions': matching_df.to_dict(orient='records'),
            'summary': {
                'total_additional_hypotheses_in_pdf': total_diff,
                'avg_additional_hypotheses_per_paper': avg_diff
            },
            'all_hypotheses_dataframe': all_hypotheses_df.to_dict(orient='records'),
        }, f, indent=2)
    
    print(f"\n✓ Saved comparison to {comparison_output}")
    print(f"✓ Saved comparative type distribution plot to {plot_path}")


if __name__ == "__main__":
    main()
