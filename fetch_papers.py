#!/usr/bin/env python3
"""
Script to fetch random accepted papers from an OpenReview venue.

Usage:
    python fetch_papers.py <venue_id>
    
Example:
    python fetch_papers.py ICLR.cc/2024/Conference
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import openreview


def fetch_accepted_papers(client: openreview.api.OpenReviewClient, venue_id: str) -> List[openreview.api.Note]:
    """
    Fetch all accepted papers from a venue.
    
    Args:
        client: OpenReview API client
        venue_id: Venue ID (e.g., 'ICLR.cc/2024/Conference')
    
    Returns:
        List of accepted paper notes
    """
    print(f"Fetching venue information for {venue_id}...")
    venue_group = client.get_group(venue_id)
    
    # Get submission name
    submission_name = venue_group.content['submission_name']['value']
    print(f"Submission name: {submission_name}")
    
    # Get accepted papers (papers with venueid matching the original venue_id)
    print("Fetching accepted papers...")
    accepted_papers = client.get_all_notes(content={'venueid': venue_id})
    
    print(f"Found {len(accepted_papers)} accepted papers")
    return accepted_papers


def extract_metadata(note: openreview.api.Note) -> Dict[str, Any]:
    """
    Extract relevant metadata from a paper note.
    
    Args:
        note: OpenReview note object
    
    Returns:
        Dictionary with paper metadata
    """
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
    }
    
    return metadata


def download_pdf(client: openreview.api.OpenReviewClient, note: openreview.api.Note, output_dir: Path) -> str:
    """
    Download PDF for a paper.
    
    Args:
        client: OpenReview API client
        note: OpenReview note object
        output_dir: Directory to save PDFs
    
    Returns:
        Path to downloaded PDF file
    """
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
    except Exception as e:
        print(f"  Error downloading PDF for paper {note.number}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Fetch random accepted papers from an OpenReview venue'
    )
    parser.add_argument(
        'venue_id',
        help='OpenReview venue ID (e.g., ICLR.cc/2024/Conference)'
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
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = args.output_dir
    pdf_dir = output_dir / 'pdfs'
    output_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)
    
    # Initialize OpenReview client (no credentials needed for public data)
    print("Initializing OpenReview client...")
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    # Fetch accepted papers
    accepted_papers = fetch_accepted_papers(client, args.venue_id)
    
    if not accepted_papers:
        print("No accepted papers found. Exiting.")
        return
    
    # Select random papers
    num_to_fetch = min(args.num_papers, len(accepted_papers))
    print(f"\nSelecting {num_to_fetch} random papers...")
    selected_papers = random.sample(accepted_papers, num_to_fetch)
    
    # Process papers
    print(f"\nProcessing {num_to_fetch} papers...")
    papers_metadata = []
    
    for i, note in enumerate(selected_papers, 1):
        print(f"\n[{i}/{num_to_fetch}] Processing paper {note.number}...")
        
        # Extract metadata
        metadata = extract_metadata(note)
        
        # Download PDF
        pdf_path = download_pdf(client, note, pdf_dir)
        metadata['local_pdf_path'] = pdf_path
        
        papers_metadata.append(metadata)
    
    # Save metadata to JSON
    metadata_file = output_dir / 'abstracts.json'
    print(f"\nSaving metadata to {metadata_file}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Done! Processed {len(papers_metadata)} papers")
    print(f"  - Metadata saved to: {metadata_file}")
    print(f"  - PDFs saved to: {pdf_dir}/")
    
    # Summary
    successful_downloads = sum(1 for p in papers_metadata if p['local_pdf_path'])
    print(f"\nSummary:")
    print(f"  - Total papers: {len(papers_metadata)}")
    print(f"  - Successful PDF downloads: {successful_downloads}")
    print(f"  - Failed PDF downloads: {len(papers_metadata) - successful_downloads}")


if __name__ == '__main__':
    main()
