#!/usr/bin/env python3
"""
Script to fetch random research papers from OpenReview or arXiv.

Usage:
    python fetch_papers.py <venue_id>
    python fetch_papers.py --source arxiv --arxiv-category cs.AI

Examples:
    python fetch_papers.py ICLR.cc/2024/Conference --num-papers 50
    python fetch_papers.py --source arxiv --arxiv-category cs.AI --num-papers 25 \
        --start-date 2024-01-01 --end-date 2024-06-30
"""

import argparse
import json
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arxiv
from arxiv import UnexpectedEmptyPageError
import openreview


def fetch_accepted_papers(client: openreview.api.OpenReviewClient, venue_id: str) -> List[openreview.api.Note]:
    """Fetch all accepted papers from an OpenReview venue."""
    print(f"Fetching venue information for {venue_id}...")
    venue_group = client.get_group(venue_id)

    submission_name = venue_group.content['submission_name']['value']
    print(f"Submission name: {submission_name}")

    print("Fetching accepted papers...")
    accepted_papers = client.get_all_notes(content={'venueid': venue_id})

    print(f"Found {len(accepted_papers)} accepted papers")
    return accepted_papers


def extract_openreview_metadata(note: openreview.api.Note) -> Dict[str, Any]:
    """Extract relevant metadata from an OpenReview note."""
    content = note.content

    metadata = {
        'id': note.id,
        'number': note.number,
        'title': content.get('title', {}).get('value', 'N/A'),
        'abstract': content.get('abstract', {}).get('value', 'N/A'),
        'authors': content.get('authors', {}).get('value', []),
        'keywords': content.get('keywords', {}).get('value', []),
        'venue': content.get('venue', {}).get('value', 'N/A'),
        'venueid': content.get('venueid', {}).get('value', 'N/A'),
        'pdf_url': content.get('pdf', {}).get('value', None),
        'source': 'openreview',
    }

    return metadata


def download_openreview_pdf(
    client: openreview.api.OpenReviewClient,
    note: openreview.api.Note,
    output_dir: Path,
) -> str | None:
    """Download PDF for an OpenReview paper."""
    if not note.content.get('pdf', {}).get('value'):
        print(f"  Warning: No PDF available for paper {note.number}")
        return None

    pdf_filename = f"paper_{note.number}.pdf"
    pdf_path = output_dir / pdf_filename

    try:
        pdf_data = client.get_attachment(field_name='pdf', id=note.id)
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        print(f"  Downloaded PDF for paper {note.number}")
        return str(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  Error downloading PDF for paper {note.number}: {exc}")
        return None


def sanitize_for_filename(value: str) -> str:
    """Replace characters that are unsafe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def default_date_range() -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    end = now
    start = (now - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end


def parse_date(date_str: str, is_end: bool) -> datetime:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if is_end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(tzinfo=timezone.utc)


def resolve_date_range(start_str: str | None, end_str: str | None) -> Tuple[datetime, datetime]:
    default_start, default_end = default_date_range()
    start_dt = parse_date(start_str, False) if start_str else default_start
    end_dt = parse_date(end_str, True) if end_str else default_end
    if start_dt > end_dt:
        raise ValueError("Start date must be on or before end date.")
    return start_dt, end_dt


def format_for_arxiv_query(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime('%Y%m%d%H%M')


def normalize_arxiv_query_category(category: str) -> str:
    category = category.strip()
    if category.endswith('.*'):
        return category
    if '.' not in category:
        return f"{category}.*"
    return category


def fetch_arxiv_results(
    category: str,
    start_dt: datetime,
    end_dt: datetime,
    max_results: int,
) -> List[arxiv.Result]:
    start_str = format_for_arxiv_query(start_dt)
    end_str = format_for_arxiv_query(end_dt)
    query_category = normalize_arxiv_query_category(category)
    query = f"cat:{query_category} AND submittedDate:[{start_str} TO {end_str}]"

    print(
        "Searching arXiv for category "
        f"{category} between {start_dt.date()} and {end_dt.date()}..."
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=200, delay_seconds=3)

    results: List[arxiv.Result] = []

    try:
        for result in client.results(search):
            published = result.published
            if not published:
                continue
            published_utc = published.astimezone(timezone.utc)
            if start_dt <= published_utc <= end_dt:
                results.append(result)
            if len(results) >= max_results:
                break
    except UnexpectedEmptyPageError as exc:
        print(f"Encountered empty page while paging results ({exc}); continuing with collected papers.")

    print(f"Found {len(results)} matching arXiv papers")
    return results


def build_arxiv_metadata(result: arxiv.Result) -> Dict[str, Any]:
    return {
        'id': result.get_short_id(),
        'title': result.title.strip(),
        'abstract': result.summary.strip(),
        'authors': [author.name for author in result.authors],
        'categories': list(result.categories),
        'primary_category': result.primary_category,
        'pdf_url': result.pdf_url,
        'entry_id': result.entry_id,
        'published': result.published.isoformat() if result.published else None,
        'updated': result.updated.isoformat() if result.updated else None,
        'source': 'arxiv',
    }


def download_arxiv_pdf(result: arxiv.Result, output_dir: Path) -> str | None:
    safe_id = sanitize_for_filename(result.get_short_id())
    pdf_filename = f"{safe_id}.pdf"
    pdf_path = output_dir / pdf_filename

    if pdf_path.exists():
        print(f"  PDF already exists for {result.get_short_id()}, skipping download")
        return str(pdf_path)

    try:
        result.download_pdf(dirpath=str(output_dir), filename=pdf_filename)
        print(f"  Downloaded PDF for arXiv paper {result.get_short_id()}")
        return str(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  Error downloading PDF for arXiv paper {result.get_short_id()}: {exc}")
        return None


def select_random_items(items: List[Any], count: int) -> List[Any]:
    if not items or count <= 0:
        return []
    num_to_fetch = min(count, len(items))
    if num_to_fetch == 0:
        return []
    return random.sample(items, num_to_fetch)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch random research papers from OpenReview or arXiv.'
    )
    parser.add_argument(
        'venue_id',
        nargs='?',
        help='OpenReview venue ID (required when --source=openreview)'
    )
    parser.add_argument(
        '--source',
        choices=('openreview', 'arxiv'),
        default='openreview',
        help='Paper source to use (default: openreview)'
    )
    parser.add_argument(
        '--num-papers',
        type=int,
        default=100,
        help='Number of random papers to fetch (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data'),
        help='Output directory for data (default: data/)'
    )
    parser.add_argument(
        '--arxiv-category',
        help='arXiv category (e.g., cs.AI). Required when --source=arxiv.'
    )
    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD) for arXiv search window (default: 1 year ago)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD) for arXiv search window (default: today)'
    )
    parser.add_argument(
        '--arxiv-max-results',
        type=int,
        default=2000,
        help='Maximum arXiv results to inspect before sampling (default: 2000)'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = args.output_dir
    pdf_dir = output_dir / 'pdfs'
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    if args.source == 'openreview':
        if not args.venue_id:
            parser.error("venue_id is required when --source=openreview")

        print("Initializing OpenReview client...")
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

        accepted_papers = fetch_accepted_papers(client, args.venue_id)

        if not accepted_papers:
            print("No accepted papers found. Exiting.")
            return

        selected_papers = select_random_items(accepted_papers, args.num_papers)
        if not selected_papers:
            print("No papers selected. Exiting.")
            return

        print(f"\nProcessing {len(selected_papers)} papers...")
        papers_metadata: List[Dict[str, Any]] = []

        for i, note in enumerate(selected_papers, 1):
            print(f"\n[{i}/{len(selected_papers)}] Processing paper {note.number}...")

            metadata = extract_openreview_metadata(note)
            pdf_path = download_openreview_pdf(client, note, pdf_dir)
            metadata['local_pdf_path'] = pdf_path

            papers_metadata.append(metadata)

    else:
        if not args.arxiv_category:
            parser.error("--arxiv-category is required when --source=arxiv")
        if args.arxiv_max_results <= 0:
            parser.error("--arxiv-max-results must be a positive integer")

        try:
            start_dt, end_dt = resolve_date_range(args.start_date, args.end_date)
        except ValueError as exc:
            parser.error(str(exc))

        arxiv_results = fetch_arxiv_results(
            category=args.arxiv_category,
            start_dt=start_dt,
            end_dt=end_dt,
            max_results=args.arxiv_max_results,
        )

        if not arxiv_results:
            print("No arXiv papers found. Exiting.")
            return

        selected_results = select_random_items(arxiv_results, args.num_papers)
        if not selected_results:
            print("No papers selected. Exiting.")
            return

        print(f"\nProcessing {len(selected_results)} papers...")
        papers_metadata = []

        for i, result in enumerate(selected_results, 1):
            paper_id = result.get_short_id()
            print(f"\n[{i}/{len(selected_results)}] Processing arXiv paper {paper_id}...")

            metadata = build_arxiv_metadata(result)
            metadata['arxiv_category'] = args.arxiv_category
            metadata['time_window'] = {
                'start': start_dt.isoformat(),
                'end': end_dt.isoformat(),
            }

            pdf_path = download_arxiv_pdf(result, pdf_dir)
            metadata['local_pdf_path'] = pdf_path

            papers_metadata.append(metadata)

    if not papers_metadata:
        print("No papers processed. Exiting.")
        return

    metadata_file = output_dir / 'abstracts.json'
    print(f"\nSaving metadata to {metadata_file}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done! Processed {len(papers_metadata)} papers")
    print(f"  - Metadata saved to: {metadata_file}")
    print(f"  - PDFs saved to: {pdf_dir}/")

    successful_downloads = sum(1 for p in papers_metadata if p.get('local_pdf_path'))
    print("\nSummary:")
    print(f"  - Total papers: {len(papers_metadata)}")
    print(f"  - Successful PDF downloads: {successful_downloads}")
    print(f"  - Failed PDF downloads: {len(papers_metadata) - successful_downloads}")


if __name__ == '__main__':
    main()
