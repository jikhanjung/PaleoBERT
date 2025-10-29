#!/usr/bin/env python
"""
P04 Task 2: Extract domain vocabulary from PDF text.

Extracts taxa, stratigraphic units, chronostratigraphic units, and localities
from the Geyer 2019 PDF, then merges with existing vocabulary files.

Usage:
    python scripts/p04_extract_vocabulary.py

Output:
    - Updated artifacts/vocab/*.txt files
"""

import sys
import os
import re
from collections import defaultdict
from typing import Dict, Set

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_taxa(text: str) -> Set[str]:
    """
    Extract taxonomic names from text.

    Uses heuristics to identify genus names:
    - Capitalized Latin binomials (Genus species)
    - Filters out common English words

    Args:
        text: Input text

    Returns:
        Set of extracted taxa (genus names)
    """
    taxa = set()

    # Pattern: Capitalized Latin binomials
    # Match: Genus species, Genus sp., etc.
    pattern = r'\b([A-Z][a-z]{3,})\s+([a-z]{3,}|sp\.)\b'
    matches = re.findall(pattern, text)

    # Common English words to filter out (not taxa)
    stopwords = {
        'The', 'This', 'These', 'Some', 'Many', 'Figure', 'Table',
        'However', 'Therefore', 'Although', 'Such', 'Other', 'Several',
        'More', 'Most', 'Less', 'Also', 'Only', 'First', 'Second',
        'During', 'After', 'Before', 'Where', 'When', 'Which', 'While',
        'Their', 'There', 'Then', 'Than', 'About', 'Above', 'Below',
        'Upper', 'Lower', 'Middle', 'Early', 'Late', 'Recent', 'Modern',
        'North', 'South', 'East', 'West', 'Central', 'Western', 'Eastern',
    }

    for genus, species in matches:
        # Filter stopwords
        if genus in stopwords:
            continue

        # Add genus name
        taxa.add(genus)

    return taxa


def extract_formations(text: str) -> Set[str]:
    """
    Extract formation names from text.

    Pattern: X Formation, X Member, X Limestone, etc.

    Args:
        text: Input text

    Returns:
        Set of extracted formation names (with underscores)
    """
    formations = set()

    # Pattern: One or more capitalized words + unit type
    unit_types = [
        'Formation', 'Member', 'Group', 'Supergroup',
        'Limestone', 'Shale', 'Sandstone', 'Dolomite',
        'Beds', 'Series'  # Note: "Series" here means rock series, not time series
    ]

    for unit_type in unit_types:
        # Match 1-3 capitalized words before unit type
        pattern = rf'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){{0,2}})\s+({unit_type})\b'
        matches = re.findall(pattern, text)

        for name, utype in matches:
            # Normalize to underscores
            full_name = f"{name}_{utype}".replace(' ', '_')
            formations.add(full_name)

    return formations


def extract_chrono_units(text: str) -> Set[str]:
    """
    Extract chronostratigraphic units from text.

    Includes Cambrian stages, series, and epochs.

    Args:
        text: Input text

    Returns:
        Set of extracted chrono units (with underscores)
    """
    chrono_units = set()

    # Cambrian stages (official names)
    stages = [
        'Fortunian', 'Stage_2', 'Stage_3', 'Stage_4', 'Stage_5',
        'Wuliuan', 'Drumian', 'Guzhangian',
        'Paibian', 'Jiangshanian', 'Stage_10'
    ]

    # Cambrian series
    series = [
        'Terreneuvian', 'Terreneuvian_Series',
        'Series_2',
        'Miaolingian', 'Miaolingian_Series',
        'Furongian', 'Furongian_Series'
    ]

    # Cambrian epochs (same as series for Cambrian)
    epochs = [
        'Epoch_2', 'Epoch_3', 'Epoch_4'
    ]

    # Check for each unit in text
    for unit in stages + series + epochs:
        # Normalize underscores for search
        search_term = unit.replace('_', r'\s+')
        if re.search(search_term, text, re.IGNORECASE):
            chrono_units.add(unit)

    # Also search for generic "Stage N" and "Series N" patterns
    for match in re.finditer(r'\b(Stage|Series|Epoch)\s+(\d+)\b', text):
        unit_type = match.group(1)
        number = match.group(2)
        chrono_units.add(f"{unit_type}_{number}")

    return chrono_units


def extract_localities(text: str) -> Set[str]:
    """
    Extract geographic localities from text.

    Uses a combination of known localities and pattern matching.

    Args:
        text: Input text

    Returns:
        Set of extracted locality names (with underscores)
    """
    localities = set()

    # Known Cambrian localities (curated list)
    known_localities = [
        # North America
        'House_Range', 'Drum_Mountains', 'Yoho_National_Park',
        'Death_Valley', 'Grand_Canyon', 'Marble_Mountains',
        'White-Inyo_Mountains', 'Avalon_Peninsula',

        # Asia
        'Yangtze_Platform', 'Siberian_Platform', 'Tarim_Basin',
        'Altai_Mountains', 'Sayan_Mountains',

        # Australia
        'Flinders_Ranges', 'Amadeus_Basin',

        # Europe
        'Cantabrian_Mountains', 'Holy_Cross_Mountains',

        # Africa
        'Atlas_Mountains', 'Anti-Atlas',

        # Antarctica
        'Transantarctic_Mountains',
    ]

    for loc in known_localities:
        # Normalize underscores for search
        search_term = loc.replace('_', r'\s+').replace('-', r'[-\s]')
        if re.search(search_term, text, re.IGNORECASE):
            localities.add(loc)

    # Pattern-based extraction for "X Range", "X Mountains", "X Basin"
    geographic_types = ['Range', 'Mountains', 'Basin', 'Peninsula', 'Platform']
    for geo_type in geographic_types:
        pattern = rf'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+({geo_type})\b'
        matches = re.findall(pattern, text)

        for name, gtype in matches:
            full_name = f"{name}_{gtype}".replace(' ', '_')
            localities.add(full_name)

    return localities


def extract_all_terms(text_path: str) -> Dict[str, Set[str]]:
    """
    Extract all vocabulary terms from text file.

    Args:
        text_path: Path to input text file

    Returns:
        Dictionary mapping category to set of terms
    """
    print(f"Extracting vocabulary from: {text_path}")

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  Text length: {len(text):,} characters")

    # Extract each category
    print("  Extracting taxa...")
    taxa = extract_taxa(text)

    print("  Extracting stratigraphic units...")
    formations = extract_formations(text)

    print("  Extracting chronostratigraphic units...")
    chrono_units = extract_chrono_units(text)

    print("  Extracting localities...")
    localities = extract_localities(text)

    terms = {
        'taxa': taxa,
        'strat_units': formations,
        'chrono_units': chrono_units,
        'localities': localities,
    }

    return terms


def merge_vocabulary(category: str, new_terms: Set[str], vocab_dir: str = "artifacts/vocab") -> int:
    """
    Merge new terms into existing vocabulary file.

    Args:
        category: Vocabulary category (taxa, strat_units, etc.)
        new_terms: Set of new terms to add
        vocab_dir: Directory containing vocabulary files

    Returns:
        Number of new terms added
    """
    vocab_file = os.path.join(vocab_dir, f"{category}.txt")

    # Load existing terms
    existing_terms = set()
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            existing_terms = set(line.strip() for line in f if line.strip())

    # Find truly new terms
    truly_new = new_terms - existing_terms

    # Merge
    all_terms = existing_terms | new_terms

    # Sort and save
    sorted_terms = sorted(all_terms)

    os.makedirs(vocab_dir, exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            f.write(term + '\n')

    return len(truly_new)


def main():
    """Main execution function."""
    print("=" * 80)
    print("P04 Task 2: Vocabulary Extraction from PDF")
    print("=" * 80)

    # Paths
    text_path = "data/pdf_extracted/geyer2019_raw.txt"
    vocab_dir = "artifacts/vocab"

    # Check if extracted text exists
    if not os.path.exists(text_path):
        print(f"ERROR: Extracted text not found at {text_path}")
        print("Please run p04_extract_pdf_text.py first (Task 1).")
        return 1

    # Extract terms
    print("\n[1/2] Extracting vocabulary terms...")
    terms = extract_all_terms(text_path)

    # Print extraction summary
    print("\nExtraction Summary:")
    for category, term_set in terms.items():
        print(f"  {category}: {len(term_set)} terms extracted")
        # Show sample (first 10)
        if term_set:
            print(f"    Sample: {', '.join(sorted(term_set)[:10])}")

    # Merge with existing vocabulary
    print(f"\n[2/2] Merging with existing vocabulary in {vocab_dir}...")
    total_new = 0
    for category, term_set in terms.items():
        new_count = merge_vocabulary(category, term_set, vocab_dir)
        total_new += new_count

        vocab_file = os.path.join(vocab_dir, f"{category}.txt")
        with open(vocab_file, 'r') as f:
            total_count = sum(1 for line in f if line.strip())

        print(f"  {category}.txt: +{new_count} new terms (total: {total_count})")

    # Success summary
    print("\n" + "=" * 80)
    print("✓ P04 Task 2 COMPLETE")
    print("=" * 80)
    print(f"  New terms added: {total_new}")
    print(f"  Vocabulary directory: {vocab_dir}")
    print("\nVocabulary files updated:")
    for category in terms.keys():
        vocab_file = os.path.join(vocab_dir, f"{category}.txt")
        if os.path.exists(vocab_file):
            print(f"  - {vocab_file}")

    print("\nNext steps:")
    print("  - Task 3: Generate test data from PDF")
    print("  - Rebuild tokenizer if needed: python scripts/build_tokenizer.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
