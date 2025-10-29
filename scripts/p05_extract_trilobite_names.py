#!/usr/bin/env python
"""
P05: Extract trilobite genus names from AVAILABLE_GENERIC_NAMES_FOR_TRILOBITES.pdf

This script parses the comprehensive trilobite genus catalog and extracts:
1. Cambrian trilobite genera (LCAM, MCAM, UCAM)
2. Associated formations and localities
3. Family classifications
4. Creates metadata database

Usage:
    python scripts/p05_extract_trilobite_names.py

Output:
    - data/trilobite_entries.json (all parsed entries)
    - data/trilobite_cambrian.json (Cambrian-only entries)
    - data/trilobite_metadata.json (metadata database)
"""

import sys
import os
import re
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Install with: pip install PyMuPDF")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF."""
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    total_pages = len(doc)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        full_text.append(text)

        if (page_num + 1) % 20 == 0:
            print(f"  Processed {page_num + 1}/{total_pages} pages...")

    doc.close()
    print(f"  Total pages: {total_pages}")

    return '\n'.join(full_text)


def parse_trilobite_entries(text: str) -> List[Dict]:
    """
    Parse trilobite genus entries from the alphabetical list.

    Format example (may span multiple lines):
    Olenellus HALL, 1862 [gilberti] Latham Shale,
    California, USA; OLENELLIDAE; LCAM.

    Returns:
        List of dicts with parsed information
    """
    print("\nParsing trilobite entries...")

    entries = []

    # Split into lines
    lines = text.split('\n')

    # Rejoin lines to handle multi-line entries
    # Strategy: Start a new entry when line starts with Capital letter followed by uppercase author
    rejoined_lines = []
    current_entry = ""

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if this is the start of a new entry
        # Pattern: Starts with capitalized genus name followed by uppercase author
        if re.match(r'^[A-Z][a-z]+(?:[a-z]+)?\s+[A-Z]', line):
            # Save previous entry
            if current_entry:
                rejoined_lines.append(current_entry)
            # Start new entry
            current_entry = line
        else:
            # Continue current entry
            if current_entry:
                current_entry += " " + line

    # Add last entry
    if current_entry:
        rejoined_lines.append(current_entry)

    print(f"  Rejoined into {len(rejoined_lines)} potential entries")

    # Pattern to match trilobite entries
    # Genus AUTHOR, YEAR [species] Formation, Locality; FAMILY; AGE
    pattern = re.compile(
        r'^([A-Z][a-z]+(?:[a-z]+)?)\s+'  # Genus name (capitalized)
        r'([A-Z][A-Z\s&\.]+?),?\s*'  # Author (uppercase, may have spaces, &, .)
        r'(\d{4}[a-z]?)\s+'  # Year
        r'\[([^\]]+)\]\s*'  # Type species in brackets
        r'(.+?)\s*;\s*'  # Locality/Formation info (non-greedy) up to semicolon
        r'([A-Z][A-Z]+(?:IDAE|INAE|OIDEA)?)\s*;?\s*'  # Family name
        r'([LMU]?[A-Z]+(?:[-\/][LMU]?[A-Z]+)?)\s*\.?',  # Age (e.g., LCAM, MCAM, UCAM, LORD-MORD)
        re.DOTALL
    )

    entry_count = 0
    failed_count = 0

    for line in rejoined_lines:
        # Try to match the pattern
        match = pattern.match(line)

        if match:
            genus = match.group(1).strip()
            author = match.group(2).strip()
            year = match.group(3).strip()
            type_species = match.group(4).strip()
            location_info = match.group(5).strip()
            family = match.group(6).strip()
            age = match.group(7).strip()

            # Parse location info (formation, locality, country)
            formation, locality, country = parse_location_info(location_info)

            entry = {
                'genus': genus,
                'author': author,
                'year': year,
                'type_species': type_species,
                'formation': formation,
                'locality': locality,
                'country': country,
                'family': family,
                'age': age,
                'raw_line': line[:200]  # Truncate for readability
            }

            entries.append(entry)
            entry_count += 1

            if entry_count % 500 == 0:
                print(f"  Parsed {entry_count} entries...")
        else:
            failed_count += 1

    print(f"  Total entries parsed: {len(entries)}")
    print(f"  Failed to parse: {failed_count}")

    return entries


def parse_location_info(location_str: str) -> Tuple[str, str, str]:
    """
    Parse location information string into formation, locality, country.

    Examples:
        "Wheeler Formation, Utah, USA" → ("Wheeler Formation", "Utah", "USA")
        "Burgess Shale, British Columbia, Canada" → ("Burgess Shale", "British Columbia", "Canada")
        "Sweden" → (None, "Sweden", None)
    """
    parts = [p.strip() for p in location_str.split(',')]

    formation = None
    locality = None
    country = None

    if len(parts) >= 3:
        # Format: Formation, State/Province, Country
        formation = parts[0]
        locality = parts[1]
        country = parts[2]
    elif len(parts) == 2:
        # Format: Locality, Country OR Formation, Country
        if any(keyword in parts[0] for keyword in ['Formation', 'Shale', 'Limestone', 'Fm', 'Member']):
            formation = parts[0]
            country = parts[1]
        else:
            locality = parts[0]
            country = parts[1]
    elif len(parts) == 1:
        # Just country or locality
        locality = parts[0]

    return formation, locality, country


def filter_cambrian_entries(entries: List[Dict]) -> List[Dict]:
    """Filter entries that are Cambrian age."""
    print("\nFiltering Cambrian entries...")

    cambrian_entries = []

    for entry in entries:
        age = entry['age']

        # Check if age contains 'CAM'
        if 'CAM' in age.upper():
            cambrian_entries.append(entry)

    print(f"  Cambrian entries: {len(cambrian_entries)} / {len(entries)}")

    # Statistics by period
    age_stats = defaultdict(int)
    for entry in cambrian_entries:
        age_stats[entry['age']] += 1

    print(f"  Age distribution:")
    for age, count in sorted(age_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"    {age}: {count}")

    return cambrian_entries


def extract_vocabulary_terms(entries: List[Dict]) -> Dict[str, Set[str]]:
    """
    Extract vocabulary terms from entries.

    Returns:
        Dict with keys: 'taxa', 'strat_units', 'localities'
    """
    print("\nExtracting vocabulary terms...")

    vocab = {
        'taxa': set(),
        'strat_units': set(),
        'localities': set()
    }

    for entry in entries:
        # Taxa: genus names
        vocab['taxa'].add(entry['genus'])

        # Formations
        if entry['formation']:
            # Normalize formation name
            formation = entry['formation']

            # Replace spaces with underscores
            formation_normalized = formation.replace(' ', '_')
            vocab['strat_units'].add(formation_normalized)

        # Localities
        if entry['locality']:
            locality = entry['locality']
            # Normalize locality name
            locality_normalized = locality.replace(' ', '_')
            vocab['localities'].add(locality_normalized)

        if entry['country']:
            country = entry['country']
            country_normalized = country.replace(' ', '_')
            vocab['localities'].add(country_normalized)

    print(f"  Taxa: {len(vocab['taxa'])}")
    print(f"  Formations: {len(vocab['strat_units'])}")
    print(f"  Localities: {len(vocab['localities'])}")

    return vocab


def create_metadata_database(entries: List[Dict]) -> Dict[str, Dict]:
    """
    Create metadata database with genus as key.

    Returns:
        Dict mapping genus name to metadata
    """
    print("\nCreating metadata database...")

    metadata = {}

    for entry in entries:
        genus = entry['genus']

        # Aggregate information for each genus
        if genus not in metadata:
            metadata[genus] = {
                'family': entry['family'],
                'age': entry['age'],
                'type_species': entry['type_species'],
                'formations': [],
                'localities': [],
                'countries': []
            }

        # Add formation
        if entry['formation']:
            formation = entry['formation'].replace(' ', '_')
            if formation not in metadata[genus]['formations']:
                metadata[genus]['formations'].append(formation)

        # Add locality
        if entry['locality']:
            locality = entry['locality'].replace(' ', '_')
            if locality not in metadata[genus]['localities']:
                metadata[genus]['localities'].append(locality)

        # Add country
        if entry['country']:
            country = entry['country'].replace(' ', '_')
            if country not in metadata[genus]['countries']:
                metadata[genus]['countries'].append(country)

    print(f"  Metadata entries: {len(metadata)}")

    return metadata


def save_json(data: any, output_path: str):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("P05: Extract Trilobite Names from PDF")
    print("=" * 80)

    # Paths
    pdf_path = "AVAILABLE_GENERIC_NAMES_FOR_TRILOBITES.pdf"

    output_dir = "data"
    all_entries_path = os.path.join(output_dir, "trilobite_entries.json")
    cambrian_entries_path = os.path.join(output_dir, "trilobite_cambrian.json")
    metadata_path = os.path.join(output_dir, "trilobite_metadata.json")

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return 1

    # Step 1: Extract text from PDF
    print("\n[1/6] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Parse trilobite entries
    print("\n[2/6] Parsing trilobite entries...")
    entries = parse_trilobite_entries(text)

    if len(entries) == 0:
        print("ERROR: No entries parsed from PDF")
        return 1

    # Save all entries
    save_json(entries, all_entries_path)

    # Step 3: Filter Cambrian entries
    print("\n[3/6] Filtering Cambrian entries...")
    cambrian_entries = filter_cambrian_entries(entries)

    if len(cambrian_entries) == 0:
        print("WARNING: No Cambrian entries found")
    else:
        save_json(cambrian_entries, cambrian_entries_path)

    # Step 4: Extract vocabulary terms
    print("\n[4/6] Extracting vocabulary terms...")
    vocab = extract_vocabulary_terms(cambrian_entries)

    # Step 5: Create metadata database
    print("\n[5/6] Creating metadata database...")
    metadata = create_metadata_database(cambrian_entries)
    save_json(metadata, metadata_path)

    # Step 6: Summary statistics
    print("\n[6/6] Summary statistics...")
    print(f"\n  Total trilobite genera: {len(entries)}")
    print(f"  Cambrian genera: {len(cambrian_entries)}")
    print(f"  Cambrian percentage: {len(cambrian_entries) / len(entries) * 100:.1f}%")
    print(f"\n  Vocabulary terms extracted:")
    print(f"    Taxa: {len(vocab['taxa'])}")
    print(f"    Formations: {len(vocab['strat_units'])}")
    print(f"    Localities: {len(vocab['localities'])}")

    # Show sample entries
    print(f"\n  Sample Cambrian genera (first 10):")
    for i, entry in enumerate(cambrian_entries[:10]):
        print(f"    {entry['genus']} ({entry['age']}) - {entry['family']}")

    # Success summary
    print("\n" + "=" * 80)
    print("✓ P05 Phase 1 COMPLETE: Trilobite Data Extraction")
    print("=" * 80)
    print(f"  Output files:")
    print(f"    - {all_entries_path}")
    print(f"    - {cambrian_entries_path}")
    print(f"    - {metadata_path}")
    print(f"\n  Next: Phase 2 - Update vocabulary files")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
