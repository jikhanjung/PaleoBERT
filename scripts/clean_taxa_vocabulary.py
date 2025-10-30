#!/usr/bin/env python
"""
Clean taxa.txt vocabulary file by removing:
1. Common English words (non-biological terms)
2. Chronostratigraphic terms (belongs in chrono_units.txt)
3. Generic terms that are not taxonomic names

Usage:
    python scripts/clean_taxa_vocabulary.py
"""

import os
from pathlib import Path

def load_vocab_file(filepath):
    """Load vocabulary from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_vocab_file(filepath, terms):
    """Save vocabulary to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for term in sorted(terms):
            f.write(term + '\n')

def clean_taxa_vocabulary():
    """Clean taxa.txt by removing non-taxonomic terms."""

    taxa_file = Path("artifacts/vocab/taxa.txt")
    chrono_file = Path("artifacts/vocab/chrono_units.txt")

    print("=" * 80)
    print("Taxa Vocabulary Cleaning")
    print("=" * 80)

    # Load current taxa
    print(f"\nLoading taxa from: {taxa_file}")
    taxa = load_vocab_file(taxa_file)
    print(f"  Current count: {len(taxa)} terms")

    # Load chrono units for reference
    chrono_units = set()
    if chrono_file.exists():
        chrono_units = set(load_vocab_file(chrono_file))
        print(f"\nLoaded {len(chrono_units)} chronostratigraphic units for comparison")

    # Define terms to remove
    # 1. Common English words (clearly not taxa)
    common_words = {
        'Abandoned', 'Abstracts', 'Additional', 'Whether', 'Workshops',
        'Zone',  # Generic term
        'Asynchronous', 'Correlation', 'Evolution', 'International',
        'Section', 'Series', 'Structure',  # Round 2
    }

    # 2. Chronostratigraphic terms (should be in chrono_units.txt)
    chrono_terms = {
        'Cambrian', 'Wuliuan',  # These are time periods, not organisms
    }

    # 3. Other problematic terms
    other_problematic = {
        'Acritarchs',  # This is a group name, not a genus
        'Vserossiyskaya',  # Russian institutional name (All-Russian)
    }

    # Combine all terms to remove
    terms_to_remove = common_words | chrono_terms | other_problematic

    # Filter taxa
    cleaned_taxa = [term for term in taxa if term not in terms_to_remove]

    # Report removals
    removed = set(taxa) - set(cleaned_taxa)

    print("\n" + "=" * 80)
    print("Cleaning Results")
    print("=" * 80)

    print(f"\nTerms removed ({len(removed)}):")
    for term in sorted(removed):
        reason = ""
        if term in common_words:
            reason = "Common English word"
        elif term in chrono_terms:
            reason = f"Chronostratigraphic term (exists in chrono_units.txt: {term in chrono_units})"
        elif term in other_problematic:
            reason = "Group name, not genus"
        print(f"  - {term:20s} [{reason}]")

    print(f"\nVocabulary size:")
    print(f"  Before: {len(taxa)}")
    print(f"  After:  {len(cleaned_taxa)}")
    print(f"  Removed: {len(removed)}")

    # Save cleaned vocabulary
    backup_file = taxa_file.with_suffix('.txt.backup')
    print(f"\nCreating backup: {backup_file}")
    save_vocab_file(backup_file, taxa)

    print(f"Saving cleaned vocabulary: {taxa_file}")
    save_vocab_file(taxa_file, cleaned_taxa)

    print("\n" + "=" * 80)
    print("✓ Cleaning Complete")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Review the changes above")
    print("  2. Rebuild tokenizer: python scripts/build_tokenizer.py")
    print("  3. If needed, restore backup: mv artifacts/vocab/taxa.txt.backup artifacts/vocab/taxa.txt")

    return len(removed)

if __name__ == "__main__":
    removed_count = clean_taxa_vocabulary()
    exit(0 if removed_count > 0 else 1)
