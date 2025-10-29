#!/usr/bin/env python
"""
P06: Batch PDF extraction for corpus expansion.

Processes all trilobite PDFs in pdfs/ directory and creates JSONL corpus entries.

Usage:
    python scripts/p06_batch_extract_pdfs.py

Output:
    - data/pdf_extracted/{basename}_raw.txt (raw text for each PDF)
    - data/corpus_norm/train_{basename}.jsonl (JSONL corpus for each PDF)
    - data/corpus_norm/train_all_pdfs.jsonl (merged corpus)
"""

import sys
import os
import re
import json
import glob
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Install with: pip install PyMuPDF")
    sys.exit(1)


def sanitize_filename(filename: str) -> str:
    """
    Convert PDF filename to sanitized basename for output files.

    Examples:
        "Babcock - 1994 - Systematics.pdf" → "babcock1994"
        "Geyer and Vincent - 2015.pdf" → "geyer_vincent2015"
    """
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Extract author and year
    # Pattern: "Author - Year" or "Author et al - Year"
    match = re.match(r'^([^-]+?)\s*-\s*(\d{4})', name)
    if match:
        author = match.group(1).strip()
        year = match.group(2)

        # Clean author name
        author = author.replace(' and ', '_')
        author = author.replace(' et al', '')
        author = author.replace(',', '')
        author = re.sub(r'\s+', '_', author)
        author = re.sub(r'[^a-zA-Z0-9_]', '', author)
        author = author.lower()

        return f"{author}{year}"
    else:
        # Fallback: just clean the filename
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.lower().strip('_')
        return name[:50]  # Limit length


def extract_text_from_pdf(pdf_path: str, output_path: str) -> Tuple[int, int, int]:
    """
    Extract text from PDF, filter noise.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output text file

    Returns:
        Tuple of (char_count, page_count, lines_kept)
    """
    print(f"  Opening: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)

    extracted_text = []
    total_lines_kept = 0
    total_lines_filtered = 0

    for page_num, page in enumerate(doc):
        text = page.get_text("text")

        # Remove page headers/footers and noise
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Filter rules
            # Skip empty lines
            if not line_stripped:
                total_lines_filtered += 1
                continue

            # Skip page numbers (single or double digits alone)
            if re.match(r'^\d{1,3}$', line_stripped):
                total_lines_filtered += 1
                continue

            # Skip journal headers/footers (common patterns)
            if any(keyword in line_stripped.lower() for keyword in [
                'episodes vol.', 'december 2019', 'iugs episodes',
                'doi:', 'http://', 'www.', 'journal of',
                'proceedings of', 'transactions of', 'bulletin',
                'volume ', 'issue ', 'springer', 'wiley'
            ]):
                total_lines_filtered += 1
                continue

            # Skip very short lines (likely artifacts)
            if len(line_stripped) < 10:
                total_lines_filtered += 1
                continue

            # Skip lines that are mostly numbers/punctuation (likely table data)
            alpha_chars = sum(c.isalpha() for c in line_stripped)
            if len(line_stripped) > 0 and alpha_chars / len(line_stripped) < 0.5:
                total_lines_filtered += 1
                continue

            clean_lines.append(line)
            total_lines_kept += 1

        page_text = '\n'.join(clean_lines)

        if page_text.strip():
            extracted_text.append(page_text)

        if (page_num + 1) % 50 == 0:
            print(f"    Processed {page_num + 1}/{len(doc)} pages...")

    page_count = len(doc)

    # Combine all pages
    full_text = '\n\n'.join(extracted_text)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    doc.close()
    return len(full_text), page_count, total_lines_kept


def normalize_cambrian_units(text: str) -> str:
    """
    Apply normalization for Cambrian stratigraphic units.

    Converts multi-word units to underscore-bound tokens:
    - "Stage 10" → "Stage_10"
    - "Series 2" → "Series_2"
    - "Wheeler Formation" → "Wheeler_Formation"

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    norm_text = text

    # Define normalization patterns (order matters)
    patterns = [
        # Cambrian stages (numbered)
        (r'\b(Stage)\s+(\d+)\b', r'\1_\2'),
        (r'\b(Series)\s+(\d+)\b', r'\1_\2'),
        (r'\b(Epoch)\s+(\d+)\b', r'\1_\2'),

        # Cambrian series (named)
        (r'\b(Terreneuvian)\s+(Series)\b', r'\1_\2'),
        (r'\b(Furongian)\s+(Series)\b', r'\1_\2'),
        (r'\b(Miaolingian)\s+(Series)\b', r'\1_\2'),

        # Common stratigraphic unit types
        (r'\b([A-Z][a-zA-Z]+)\s+(Formation)\b', r'\1_\2'),
        (r'\b([A-Z][a-zA-Z]+)\s+(Member)\b', r'\1_\2'),
        (r'\b([A-Z][a-zA-Z]+)\s+(Group)\b', r'\1_\2'),
        (r'\b([A-Z][a-zA-Z]+)\s+(Limestone)\b', r'\1_\2'),
        (r'\b([A-Z][a-zA-Z]+)\s+(Shale)\b', r'\1_\2'),
        (r'\b([A-Z][a-zA-Z]+)\s+(Sandstone)\b', r'\1_\2'),

        # Multi-word formations (common Cambrian ones)
        (r'\b(Burgess)\s+(Shale)\b', r'\1_\2'),
        (r'\b(Wheeler)\s+(Formation)\b', r'\1_\2'),
        (r'\b(Marjum)\s+(Formation)\b', r'\1_\2'),

        # Geographic localities (multi-word)
        (r'\b(House)\s+(Range)\b', r'\1_\2'),
        (r'\b(Drum)\s+(Mountains)\b', r'\1_\2'),
        (r'\b(Yoho)\s+(National)\s+(Park)\b', r'\1_\2'),
        (r'\b(Death)\s+(Valley)\b', r'\1_\2'),
        (r'\b(Grand)\s+(Canyon)\b', r'\1_\2'),
    ]

    for pattern, replacement in patterns:
        norm_text = re.sub(pattern, replacement, norm_text)

    return norm_text


def create_jsonl_entries(raw_text_path: str, output_jsonl: str, pub_id: str) -> Tuple[int, int]:
    """
    Create JSONL corpus entries from normalized text.

    Args:
        raw_text_path: Path to raw extracted text
        output_jsonl: Path to output JSONL file
        pub_id: Publication ID for this PDF

    Returns:
        Tuple of (entry_count, token_count)
    """
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Apply normalization
    norm_text = normalize_cambrian_units(raw_text)

    # Split into paragraphs
    # Strategy: Split on double newlines, filter by length
    paragraphs = [p.strip() for p in norm_text.split('\n\n') if p.strip()]

    # Filter paragraphs by length (remove very short ones)
    min_length = 100  # Minimum characters per paragraph
    filtered_paragraphs = [p for p in paragraphs if len(p) >= min_length]

    # Create JSONL entries
    entries = []
    for idx, para in enumerate(filtered_paragraphs):
        entry = {
            "pub_id": pub_id,
            "cap_id": f"p{idx:04d}",
            "raw_text": para,
            "norm_text": para,
            "align_map": None,  # Not needed for DAPT
        }
        entries.append(entry)

    # Write JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Calculate token count (approximate)
    total_tokens = sum(len(e['norm_text'].split()) for e in entries)

    return len(entries), total_tokens


def merge_jsonl_files(input_files: List[str], output_file: str) -> Tuple[int, int]:
    """
    Merge multiple JSONL files into a single corpus file.

    Args:
        input_files: List of input JSONL file paths
        output_file: Output merged JSONL file path

    Returns:
        Tuple of (total_entries, total_tokens)
    """
    all_entries = []
    total_tokens = 0

    for jsonl_path in input_files:
        if not os.path.exists(jsonl_path):
            continue

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                all_entries.append(entry)
                total_tokens += len(entry['norm_text'].split())

    # Write merged file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return len(all_entries), total_tokens


def main():
    """Main execution function."""
    print("=" * 80)
    print("P06: Batch PDF Extraction for Corpus Expansion")
    print("=" * 80)

    # Find all PDFs in pdfs/ directory
    pdf_dir = "pdfs"
    pdf_pattern = os.path.join(pdf_dir, "*.pdf")
    pdf_files = sorted(glob.glob(pdf_pattern))

    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}/")
        return 1

    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf_path in pdf_files:
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  - {os.path.basename(pdf_path)} ({size_mb:.1f} MB)")

    # Process each PDF
    print(f"\n[1/{len(pdf_files) + 1}] Processing PDFs...")

    processed_files = []
    total_chars = 0
    total_pages = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")

        # Generate output paths
        basename = sanitize_filename(os.path.basename(pdf_path))
        raw_text_path = f"data/pdf_extracted/{basename}_raw.txt"
        jsonl_path = f"data/corpus_norm/train_{basename}.jsonl"

        try:
            # Extract text
            char_count, page_count, lines_kept = extract_text_from_pdf(pdf_path, raw_text_path)
            print(f"    Pages: {page_count}, Lines: {lines_kept}, Chars: {char_count:,}")

            # Create JSONL entries
            entry_count, token_count = create_jsonl_entries(raw_text_path, jsonl_path, basename)
            print(f"    Entries: {entry_count}, Tokens: {token_count:,}")

            processed_files.append({
                'basename': basename,
                'jsonl_path': jsonl_path,
                'entries': entry_count,
                'tokens': token_count,
                'chars': char_count,
                'pages': page_count,
            })

            total_chars += char_count
            total_pages += page_count

        except Exception as e:
            print(f"    ERROR: Failed to process {pdf_path}: {e}")
            continue

    # Merge all JSONL files
    print(f"\n[{len(pdf_files) + 1}/{len(pdf_files) + 1}] Merging JSONL files...")

    jsonl_files = [f['jsonl_path'] for f in processed_files]

    # Also include existing corpus files
    existing_corpus = "data/corpus_norm/train_geyer2019.jsonl"
    if os.path.exists(existing_corpus):
        jsonl_files.append(existing_corpus)
        print(f"  Including existing: {existing_corpus}")

    merged_output = "data/corpus_norm/train_all_pdfs.jsonl"
    total_entries, total_tokens = merge_jsonl_files(jsonl_files, merged_output)

    # Final summary
    print("\n" + "=" * 80)
    print("✓ P06 BATCH EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nProcessed {len(processed_files)} PDFs:")
    for info in processed_files:
        print(f"  - {info['basename']}: {info['entries']} entries, {info['tokens']:,} tokens")

    print(f"\nTotal Statistics:")
    print(f"  Pages processed: {total_pages:,}")
    print(f"  Characters extracted: {total_chars:,}")
    print(f"  JSONL entries: {total_entries:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"\nMerged corpus: {merged_output}")

    # Token sufficiency analysis
    target_tokens = 40_000_000  # 40M minimum target
    if total_tokens >= target_tokens:
        print(f"\n✓ Corpus size sufficient for DAPT training ({total_tokens:,} >= {target_tokens:,})")
    else:
        shortfall = target_tokens - total_tokens
        print(f"\n⚠ Corpus still below target: {total_tokens:,} / {target_tokens:,}")
        print(f"  Shortfall: {shortfall:,} tokens ({shortfall / total_tokens * 100:.1f}% more needed)")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
