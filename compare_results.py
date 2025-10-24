#!/usr/bin/env python3
"""Script to compare and analyze classification results across datasets/modes."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--evaluated-only',
        action='store_true',
        help="Exclude hypotheses whose evaluation_status is 'not_evaluated'.",
    )
    return parser.parse_args()


def results_to_dataframe(
    results: List[Dict[str, Any]],
    mode: str,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Flatten hypothesis data into a dataframe for the given mode (and dataset)."""
    rows: List[Dict[str, Any]] = []

    for paper in results:
        hypotheses: Iterable[Dict[str, Any]] = paper.get('hypotheses', [])
        for idx, hyp in enumerate(hypotheses):
            row = {
                'dataset': dataset,
                'mode': mode,
                'paper_id': paper.get('paper_id'),
                'paper_title': paper.get('paper_title'),
                'hypothesis_index': idx,
                'evaluation_status': hyp.get('evaluation_status'),
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
    evaluation_counter: Counter[str | None] = Counter()

    for paper in results:
        for hyp in paper['hypotheses']:
            for field in TYPE_FIELDS:
                type_counters[field][hyp[field]] += 1
            evaluation_counter[hyp.get('evaluation_status')] += 1
    
    return {
        'mode': mode,
        'total_papers': total_papers,
        'total_hypotheses': total_hypotheses,
        'avg_hypotheses_per_paper': total_hypotheses / total_papers if total_papers > 0 else 0,
        'papers_with_hypotheses': papers_with_hypotheses,
        'papers_without_hypotheses': papers_without_hypotheses,
        'type_distributions': {field: dict(counter) for field, counter in type_counters.items()},
        'evaluation_status_distribution': dict(evaluation_counter),
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
    dataset = analysis.get('dataset')
    mode_label = analysis.get('mode')

    if dataset and mode_label:
        if mode_label.lower() == 'combined':
            heading = f"{dataset} (combined)"
        else:
            heading = f"{dataset} ({mode_label})"
    elif mode_label:
        heading = f"{mode_label.upper()} mode"
    elif dataset:
        heading = dataset
    else:
        heading = 'Analysis'

    print(f"\n{'='*60}")
    print(heading)
    print(f"{'='*60}")
    print(f"Total papers: {analysis['total_papers']}")
    print(f"Total hypotheses found: {analysis['total_hypotheses']}")
    print(f"Average hypotheses per paper: {analysis['avg_hypotheses_per_paper']:.2f}")
    print(f"Papers with hypotheses: {analysis['papers_with_hypotheses']}")
    print(f"Papers without hypotheses: {analysis['papers_without_hypotheses']}")

    eval_dist = analysis.get('evaluation_status_distribution') or {}
    if eval_dist:
        print("Evaluation status distribution:")
        for status, count in sorted(eval_dist.items(), key=lambda item: (-item[1], str(item[0]))):
            print(f"  {status}: {count}")


def print_comparative_type_statistics(df: pd.DataFrame, mode_column: str = 'mode'):
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
                columns=mode_column,
                values='hypothesis_index',
                aggfunc='count',
                fill_value=0,
            )
            .sort_index()
        )
        print(f"\n{field.replace('_', ' ').title()}:")
        print(pivot)


def plot_comparative_type_distributions(
    df: pd.DataFrame,
    output_path: Path,
    mode_column: str = 'mode',
) -> None:
    """Render grouped bar charts comparing type distributions across modes."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, field in zip(axes, TYPE_FIELDS):
        pivot = (
            df.pivot_table(
                index=field,
                columns=mode_column,
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


def _plot_dataset_distribution(
    df: pd.DataFrame,
    output_path: Path,
    normalize: bool,
) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, field in zip(axes, TYPE_FIELDS):
        pivot = (
            df.groupby(['dataset', field])
            .size()
            .reset_index(name='count')
            .pivot(index=field, columns='dataset', values='count')
            .fillna(0)
            .sort_index()
        )

        if pivot.empty:
            ax.axis('off')
            continue

        if normalize:
            col_totals = pivot.sum(axis=0).replace(0, pd.NA)
            pivot = pivot.div(col_totals, axis=1).fillna(0.0)
            ylabel = 'Hypothesis fraction'
        else:
            ylabel = 'Hypothesis count'

        pivot.plot(kind='bar', ax=ax, xlabel='')
        ax.set_title(field.replace('_', ' ').title())
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.legend(title='Dataset')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

    for idx in range(len(TYPE_FIELDS), len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_dataset_comparisons(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Create dataset-level comparison plots (counts and normalized shares)."""
    outputs: Dict[str, Path] = {}

    counts_path = output_dir / 'dataset_type_distributions_counts.webp'
    _plot_dataset_distribution(df, counts_path, normalize=False)
    outputs['counts'] = counts_path

    shares_path = output_dir / 'dataset_type_distributions_shares.webp'
    _plot_dataset_distribution(df, shares_path, normalize=True)
    outputs['shares'] = shares_path

    return outputs


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


def filter_results_by_evaluation_status(
    results: Sequence[Dict[str, Any]],
    excluded_statuses: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return a deep-ish copy of results with hypotheses matching excluded statuses removed."""

    excluded = set(excluded_statuses)
    filtered_results: List[Dict[str, Any]] = []

    for paper in results:
        paper_copy = dict(paper)
        hypotheses = []
        for hyp in paper.get('hypotheses', []):
            if excluded and hyp.get('evaluation_status') in excluded:
                continue
            hypotheses.append(dict(hyp))
        paper_copy['hypotheses'] = hypotheses
        filtered_results.append(paper_copy)

    return filtered_results


def main():
    """Main execution function"""
    args = parse_args()
    data_dir = Path('data')
    datasets = {
        'ICML 2025': {
            'abstract': data_dir / 'classifications_abstract.json',
            'pdf': data_dir / 'classifications_pdf.json',
        },
        'arXiv cond-mat': {
            'abstract': data_dir / 'arxiv/cond-mat/classifications_abstract.json',
            'pdf': data_dir / 'arxiv/cond-mat/classifications_pdf.json',
        },
    }

    dataset_mode_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    mode_analyses: List[Dict[str, Any]] = []
    combined_analyses: List[Dict[str, Any]] = []
    dataset_comparisons: Dict[str, Dict[str, Any]] = {}
    matching_results: Dict[str, pd.DataFrame] = {}
    dataframes: List[pd.DataFrame] = []

    excluded_statuses = ['not_evaluated'] if args.evaluated_only else []

    for dataset, mode_paths in datasets.items():
        dataset_mode_results[dataset] = {}
        combined_results: List[Dict[str, Any]] = []

        for mode, path in mode_paths.items():
            results = load_results(path)
            filtered_results = filter_results_by_evaluation_status(results, excluded_statuses)

            dataset_mode_results[dataset][mode] = filtered_results
            combined_results.extend(filtered_results)

            df = results_to_dataframe(filtered_results, mode, dataset)
            dataframes.append(df)

            analysis = analyze_hypotheses(filtered_results, mode)
            analysis['dataset'] = dataset
            mode_analyses.append(analysis)

        combined_analysis = analyze_hypotheses(combined_results, 'combined')
        combined_analysis['dataset'] = dataset
        combined_analyses.append(combined_analysis)

        abstract_results = dataset_mode_results[dataset].get('abstract', [])
        pdf_results = dataset_mode_results[dataset].get('pdf', [])

        if abstract_results and pdf_results:
            comparisons = compare_paper_results(abstract_results, pdf_results)
            matching_df = compute_matching_fractions(abstract_results, pdf_results)
            matching_results[dataset] = matching_df

            total_diff = sum(comp['difference'] for comp in comparisons)
            avg_diff = total_diff / len(comparisons) if comparisons else 0
            avg_matching_fraction = matching_df['matching_fraction'].mean() if not matching_df.empty else 0.0

            dataset_comparisons[dataset] = {
                'per_paper_comparison': comparisons,
                'matching_fractions': matching_df.to_dict(orient='records'),
                'summary': {
                    'total_additional_hypotheses_in_pdf': total_diff,
                    'avg_additional_hypotheses_per_paper': avg_diff,
                    'avg_matching_fraction': avg_matching_fraction,
                },
            }
        else:
            dataset_comparisons[dataset] = {
                'per_paper_comparison': [],
                'matching_fractions': [],
                'summary': {},
            }

    all_hypotheses_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    if not all_hypotheses_df.empty:
        all_hypotheses_df['mode_label'] = (
            all_hypotheses_df['dataset'] + ' | ' + all_hypotheses_df['mode']
        )

    print(f"\n{'='*60}")
    print("Per-Mode Analyses")
    for analysis in mode_analyses:
        print_analysis(analysis)

    print(f"\n{'='*60}")
    print("Dataset-Level Analyses")
    for analysis in combined_analyses:
        print_analysis(analysis)

    if not all_hypotheses_df.empty:
        print_comparative_type_statistics(all_hypotheses_df, mode_column='mode_label')

    plot_path = data_dir / 'comparative_type_distributions.webp'
    if not all_hypotheses_df.empty:
        plot_comparative_type_distributions(
            all_hypotheses_df, plot_path, mode_column='mode_label'
        )

    dataset_plot_paths: Dict[str, Path] = {}
    if not all_hypotheses_df.empty:
        dataset_plot_paths = plot_dataset_comparisons(all_hypotheses_df, data_dir)

    for dataset, match_df in matching_results.items():
        print(f"\n{'='*60}")
        print(f"Per-Paper Matching Fractions – {dataset}")
        print(match_df)

    comparison_output = data_dir / 'comparison_summary.json'

    dataset_type_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    dataset_type_shares: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not all_hypotheses_df.empty:
        for field in TYPE_FIELDS:
            pivot_counts = (
                all_hypotheses_df.groupby(['dataset', field])
                .size()
                .unstack(level=0, fill_value=0)
            )
            dataset_type_counts[field] = pivot_counts.to_dict(orient='index')

            pivot_shares = pivot_counts.div(
                pivot_counts.sum(axis=0).replace(0, pd.NA), axis=1
            ).fillna(0.0)
            dataset_type_shares[field] = pivot_shares.to_dict(orient='index')

    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'parameters': {
                    'evaluated_only': args.evaluated_only,
                    'excluded_statuses': excluded_statuses,
                },
                'per_mode_analyses': mode_analyses,
                'dataset_analyses': combined_analyses,
                'dataset_comparisons': dataset_comparisons,
                'dataset_type_counts': dataset_type_counts,
                'dataset_type_shares': dataset_type_shares,
                'plots': {
                    'mode_distributions': str(plot_path),
                    **{key: str(path) for key, path in dataset_plot_paths.items()},
                },
                'all_hypotheses_dataframe': all_hypotheses_df.to_dict(orient='records')
                if not all_hypotheses_df.empty
                else [],
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved comparison to {comparison_output}")
    if not all_hypotheses_df.empty:
        print(f"✓ Saved comparative type distribution plot to {plot_path}")
        for label, path in dataset_plot_paths.items():
            print(f"✓ Saved dataset {label} plot to {path}")


if __name__ == "__main__":
    main()
