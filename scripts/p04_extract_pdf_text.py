#!/usr/bin/env python
"""
P04 Task 1: Extract text from IUGS Cambrian correlation chart PDF.

Extracts clean text from PDF, applies normalization, and creates JSONL corpus entries.

Usage:
    python scripts/p04_extract_pdf_text.py

Output:
    - data/pdf_extracted/geyer2019_raw.txt (raw extracted text)
    - data/corpus_norm/train_geyer2019.jsonl (normalized JSONL corpus)
"""

import sys
import os
import re
import json
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Install with: pip install PyMuPDF")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str, output_path: str) -> int:
    """
    Extract text from PDF, filter noise.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output text file

    Returns:
        Number of characters extracted
    """
    print(f"Opening PDF: {pdf_path}")
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

            # Skip journal headers/footers
            if any(keyword in line_stripped for keyword in [
                'Episodes Vol.', 'December 2019', 'IUGS Episodes',
                'DOI:', 'http://', 'www.'
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

        if (page_num + 1) % 10 == 0:
            print(f"  Processed {page_num + 1}/{len(doc)} pages...")

    # Combine all pages
    full_text = '\n\n'.join(extracted_text)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\nExtraction Summary:")
    print(f"  Pages processed: {len(doc)}")
    print(f"  Lines kept: {total_lines_kept}")
    print(f"  Lines filtered: {total_lines_filtered}")
    print(f"  Characters extracted: {len(full_text):,}")
    print(f"  Output saved to: {output_path}")

    doc.close()
    return len(full_text)


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


def create_jsonl_entries(raw_text_path: str, output_jsonl: str) -> int:
    """
    Create JSONL corpus entries from normalized text.

    Args:
        raw_text_path: Path to raw extracted text
        output_jsonl: Path to output JSONL file

    Returns:
        Number of entries created
    """
    print(f"\nCreating JSONL corpus from: {raw_text_path}")

    with open(raw_text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Apply normalization
    norm_text = normalize_cambrian_units(raw_text)

    # Count normalizations applied
    if raw_text != norm_text:
        print("  Normalizations applied:")
        # Sample check
        for original, normalized in [
            ('Stage 10', 'Stage_10'),
            ('Series 2', 'Series_2'),
            ('Wheeler Formation', 'Wheeler_Formation'),
        ]:
            count = norm_text.count(normalized)
            if count > 0:
                print(f"    '{original}' → '{normalized}': {count} occurrences")

    # Split into paragraphs
    # Strategy: Split on double newlines, filter by length
    paragraphs = [p.strip() for p in norm_text.split('\n\n') if p.strip()]

    # Filter paragraphs by length (remove very short ones)
    min_length = 100  # Minimum characters per paragraph
    filtered_paragraphs = [p for p in paragraphs if len(p) >= min_length]

    print(f"  Total paragraphs: {len(paragraphs)}")
    print(f"  Filtered (len >= {min_length}): {len(filtered_paragraphs)}")

    # Create JSONL entries
    entries = []
    for idx, para in enumerate(filtered_paragraphs):
        entry = {
            "pub_id": "geyer2019",
            "cap_id": f"p{idx:04d}",
            "raw_text": para,  # Using norm_text as both raw and norm
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

    print(f"\nJSONL Creation Summary:")
    print(f"  Entries created: {len(entries)}")
    print(f"  Approximate tokens: {total_tokens:,}")
    print(f"  Output saved to: {output_jsonl}")

    return len(entries)


def validate_jsonl(jsonl_path: str) -> bool:
    """
    Validate JSONL file format.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating JSONL: {jsonl_path}")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line_num, line in enumerate(f, 1):
                obj = json.loads(line)

                # Check required fields
                required_fields = ['pub_id', 'cap_id', 'raw_text', 'norm_text']
                for field in required_fields:
                    if field not in obj:
                        print(f"  ERROR: Line {line_num} missing field '{field}'")
                        return False

                line_count += 1

        print(f"  ✓ All {line_count} entries valid")
        return True

    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON at line {line_num}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 80)
    print("P04 Task 1: PDF Text Extraction and Corpus Creation")
    print("=" * 80)

    # Paths
    pdf_path = "IUGS042-04-05.pdf"
    raw_text_path = "data/pdf_extracted/geyer2019_raw.txt"
    jsonl_path = "data/corpus_norm/train_geyer2019.jsonl"

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        print("Please ensure IUGS042-04-05.pdf is in the project root directory.")
        return 1

    # Step 1: Extract text from PDF
    print("\n[1/3] Extracting text from PDF...")
    char_count = extract_text_from_pdf(pdf_path, raw_text_path)

    if char_count == 0:
        print("ERROR: No text extracted from PDF")
        return 1

    # Step 2: Create JSONL entries
    print("\n[2/3] Creating JSONL corpus entries...")
    entry_count = create_jsonl_entries(raw_text_path, jsonl_path)

    if entry_count == 0:
        print("ERROR: No JSONL entries created")
        return 1

    # Step 3: Validate JSONL
    print("\n[3/3] Validating JSONL format...")
    if not validate_jsonl(jsonl_path):
        print("ERROR: JSONL validation failed")
        return 1

    # Success summary
    print("\n" + "=" * 80)
    print("✓ P04 Task 1 COMPLETE")
    print("=" * 80)
    print(f"  Raw text: {raw_text_path}")
    print(f"  JSONL corpus: {jsonl_path}")
    print(f"  Entries: {entry_count}")
    print(f"  Characters: {char_count:,}")
    print("\nNext steps:")
    print("  - Task 2: Extract vocabulary from PDF")
    print("  - Task 3: Generate test data from PDF")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
