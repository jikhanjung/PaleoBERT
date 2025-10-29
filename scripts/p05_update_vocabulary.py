#!/usr/bin/env python
"""
P05 Phase 2: Update vocabulary files with trilobite data.

Merges extracted trilobite genera, formations, and localities with existing
vocabulary files, using frequency-based selection to avoid vocabulary explosion.

Usage:
    python scripts/p05_update_vocabulary.py

Output:
    - artifacts/vocab/taxa.txt (updated)
    - artifacts/vocab/strat_units.txt (updated)
    - artifacts/vocab/localities.txt (updated)
"""

import sys
import os
import json
from typing import Set, Dict, List
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_existing_vocabulary(vocab_file: str) -> Set[str]:
    """Load existing vocabulary from file."""
    if not os.path.exists(vocab_file):
        return set()

    with open(vocab_file, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def load_trilobite_data(json_file: str) -> List[Dict]:
    """Load trilobite data from JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_frequent_genera(entries: List[Dict], top_n: int = 200) -> Set[str]:
    """
    Select most frequent/important genera.

    Strategy:
    1. All genera from LCAM, MCAM, UCAM (early/middle/late Cambrian)
    2. Limit to top N most frequent if needed
    """
    # Count frequency by family (proxy for importance)
    family_counts = Counter(e['family'] for e in entries if e['family'])

    # Score each genus
    genus_scores = {}

    for entry in entries:
        genus = entry['genus']
        age = entry['age']
        family = entry['family']

        score = 0

        # Higher score for pure Cambrian ages
        if age in ['LCAM', 'MCAM', 'UCAM']:
            score += 10
        elif 'CAM' in age and '-' in age:
            # Mixed ages (e.g., MCAM-UCAM)
            score += 5
        else:
            score += 1

        # Bonus for common families
        if family:
            family_freq = family_counts.get(family, 0)
            score += min(family_freq / 10, 5)  # Cap at +5

        genus_scores[genus] = score

    # Sort by score
    sorted_genera = sorted(genus_scores.items(), key=lambda x: -x[1])

    # Take top N
    selected = {genus for genus, score in sorted_genera[:top_n]}

    print(f"  Selected {len(selected)} genera out of {len(entries)}")
    print(f"  Score range: {sorted_genera[0][1]:.1f} to {sorted_genera[min(top_n-1, len(sorted_genera)-1)][1]:.1f}")

    return selected


def select_frequent_formations(entries: List[Dict], top_n: int = 100) -> Set[str]:
    """Select most frequent formations."""
    formations = []

    for entry in entries:
        if entry['formation']:
            formation = entry['formation'].replace(' ', '_')
            formations.append(formation)

    # Count frequencies
    formation_counts = Counter(formations)

    # Take top N
    top_formations = {fm for fm, count in formation_counts.most_common(top_n)}

    print(f"  Selected {len(top_formations)} formations")
    if formation_counts:
        most_common = formation_counts.most_common(1)[0]
        print(f"  Most common: {most_common[0]} ({most_common[1]} occurrences)")

    return top_formations


def select_frequent_localities(entries: List[Dict], top_n: int = 100) -> Set[str]:
    """Select most frequent localities and countries."""
    localities = []

    for entry in entries:
        if entry['locality']:
            locality = entry['locality'].replace(' ', '_')
            localities.append(locality)

        if entry['country']:
            country = entry['country'].replace(' ', '_')
            localities.append(country)

    # Count frequencies
    locality_counts = Counter(localities)

    # Take top N
    top_localities = {loc for loc, count in locality_counts.most_common(top_n)}

    print(f"  Selected {len(top_localities)} localities/countries")
    if locality_counts:
        most_common = locality_counts.most_common(1)[0]
        print(f"  Most common: {most_common[0]} ({most_common[1]} occurrences)")

    return top_localities


def merge_and_save_vocabulary(
    existing: Set[str],
    new_terms: Set[str],
    vocab_file: str,
    category: str
):
    """Merge new terms with existing and save."""
    # Merge
    all_terms = existing | new_terms

    # Sort alphabetically
    sorted_terms = sorted(all_terms)

    # Save
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            f.write(term + '\n')

    # Statistics
    added = len(new_terms - existing)
    total = len(all_terms)

    print(f"\n{category}:")
    print(f"  Existing: {len(existing)}")
    print(f"  New: {added}")
    print(f"  Total: {total}")
    print(f"  Saved to: {vocab_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("P05 Phase 2: Update Vocabulary Files")
    print("=" * 80)

    # Paths
    trilobite_json = "data/trilobite_cambrian.json"
    vocab_dir = "artifacts/vocab"

    taxa_file = os.path.join(vocab_dir, "taxa.txt")
    strat_file = os.path.join(vocab_dir, "strat_units.txt")
    loc_file = os.path.join(vocab_dir, "localities.txt")

    # Check input file
    if not os.path.exists(trilobite_json):
        print(f"ERROR: Trilobite data not found at {trilobite_json}")
        print("Please run: python scripts/p05_extract_trilobite_names.py first")
        return 1

    # Load trilobite data
    print("\n[1/6] Loading trilobite data...")
    entries = load_trilobite_data(trilobite_json)
    print(f"  Loaded {len(entries)} Cambrian trilobite entries")

    # Load existing vocabularies
    print("\n[2/6] Loading existing vocabulary...")
    existing_taxa = load_existing_vocabulary(taxa_file)
    existing_strat = load_existing_vocabulary(strat_file)
    existing_loc = load_existing_vocabulary(loc_file)

    print(f"  Existing taxa: {len(existing_taxa)}")
    print(f"  Existing formations: {len(existing_strat)}")
    print(f"  Existing localities: {len(existing_loc)}")

    # Select terms from trilobite data
    print("\n[3/6] Selecting trilobite genera...")
    selected_genera = select_frequent_genera(entries, top_n=200)

    print("\n[4/6] Selecting formations...")
    selected_formations = select_frequent_formations(entries, top_n=100)

    print("\n[5/6] Selecting localities...")
    selected_localities = select_frequent_localities(entries, top_n=100)

    # Merge and save
    print("\n[6/6] Merging and saving vocabulary files...")

    merge_and_save_vocabulary(
        existing_taxa,
        selected_genera,
        taxa_file,
        "TAXA"
    )

    merge_and_save_vocabulary(
        existing_strat,
        selected_formations,
        strat_file,
        "STRATIGRAPHIC UNITS"
    )

    merge_and_save_vocabulary(
        existing_loc,
        selected_localities,
        loc_file,
        "LOCALITIES"
    )

    # Summary
    total_existing = len(existing_taxa) + len(existing_strat) + len(existing_loc)
    total_new = (
        len(selected_genera - existing_taxa) +
        len(selected_formations - existing_strat) +
        len(selected_localities - existing_loc)
    )
    total_final = (
        len(existing_taxa | selected_genera) +
        len(existing_strat | selected_formations) +
        len(existing_loc | selected_localities)
    )

    print("\n" + "=" * 80)
    print("✓ P05 Phase 2 COMPLETE: Vocabulary Updated")
    print("=" * 80)
    print(f"  Vocabulary size:")
    print(f"    Before: {total_existing} tokens")
    print(f"    Added: {total_new} tokens")
    print(f"    After: {total_final} tokens")
    print(f"\n  Next: Rebuild tokenizer")
    print("  Command: python scripts/build_tokenizer.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
