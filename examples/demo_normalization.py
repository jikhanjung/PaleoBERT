#!/usr/bin/env python
"""
Demo script for PaleoBERT text normalization module.

This script demonstrates:
1. Basic text normalization
2. Character-level alignment maps
3. Span projection (raw ↔ norm)
4. Round-trip consistency
5. Realistic paleontology examples

Usage:
    python examples/demo_normalization.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.normalization import (
    normalize_text,
    project_span,
    create_inverse_map,
    get_normalization_stats,
)


def print_separator():
    print("\n" + "=" * 80 + "\n")


def demo_basic_normalization():
    """Demonstrate basic text normalization."""
    print("DEMO 1: Basic Text Normalization")
    print_separator()

    examples = [
        "Olenellus wheeleri",
        "Wheeler Formation",
        "Cambrian Stage 10",
        "House Range",
        "Yoho National Park",
    ]

    for raw in examples:
        norm, _ = normalize_text(raw)
        print(f"Raw:  {raw}")
        print(f"Norm: {norm}")
        print()


def demo_alignment_map():
    """Demonstrate character-level alignment map."""
    print("DEMO 2: Character-Level Alignment Map")
    print_separator()

    raw = "Wheeler Formation"
    norm, align = normalize_text(raw)

    print(f"Raw:  '{raw}'")
    print(f"Norm: '{norm}'")
    print()

    print("Alignment map (first 10 characters):")
    for i in range(min(10, len(raw))):
        raw_char = raw[i]
        norm_idx = align[i]
        norm_char = norm[norm_idx] if norm_idx < len(norm) else '?'
        print(f"  raw[{i}]='{raw_char}' → norm[{norm_idx}]='{norm_char}'")


def demo_span_projection():
    """Demonstrate span projection between raw and normalized text."""
    print("DEMO 3: Span Projection (Raw ↔ Norm)")
    print_separator()

    raw = "Olenellus wheeleri occurs in Wheeler Formation"
    norm, align = normalize_text(raw)

    print(f"Raw text:  '{raw}'")
    print(f"Norm text: '{norm}'")
    print()

    # Define entity spans in raw text
    entities_raw = [
        ("TAXON", (0, 18), "Olenellus wheeleri"),
        ("STRAT", (29, 46), "Wheeler Formation"),
    ]

    print("Raw → Norm projection:")
    for entity_type, raw_span, expected_text in entities_raw:
        # Project to normalized text
        norm_span = project_span(raw_span, align, "raw_to_norm")

        # Extract from norm text
        norm_entity = norm[norm_span[0]:norm_span[1]]

        print(f"  {entity_type}: raw{raw_span} → norm{norm_span}")
        print(f"    Raw text:  '{expected_text}'")
        print(f"    Norm text: '{norm_entity}'")
        print()


def demo_round_trip():
    """Demonstrate round-trip span projection."""
    print("DEMO 4: Round-Trip Span Projection")
    print_separator()

    raw = "Olenellus wheeleri from Wheeler Formation in House Range"
    norm, align = normalize_text(raw)

    print(f"Raw:  '{raw}'")
    print(f"Norm: '{norm}'")
    print()

    # Test spans
    test_spans = [
        (0, 18),   # "Olenellus wheeleri"
        (24, 41),  # "Wheeler Formation"
        (45, 56),  # "House Range"
    ]

    print("Round-trip test (raw → norm → raw):")
    for original_span in test_spans:
        # Extract original text
        original_text = raw[original_span[0]:original_span[1]]

        # Raw → Norm
        norm_span = project_span(original_span, align, "raw_to_norm")
        norm_text = norm[norm_span[0]:norm_span[1]]

        # Norm → Raw (should recover original)
        recovered_span = project_span(norm_span, align, "norm_to_raw")
        recovered_text = raw[recovered_span[0]:recovered_span[1]]

        # Check consistency
        consistent = (original_span == recovered_span)
        status = "✓" if consistent else "✗"

        print(f"  {status} '{original_text}'")
        print(f"     Original: {original_span}")
        print(f"     Norm:     {norm_span} → '{norm_text}'")
        print(f"     Recovered: {recovered_span} → '{recovered_text}'")
        print()


def demo_realistic_example():
    """Demonstrate realistic paleontology text processing."""
    print("DEMO 5: Realistic Paleontology Example")
    print_separator()

    # Realistic figure caption
    raw = (
        "Olenellus wheeleri and Elrathia kingii from the Wheeler Formation, "
        "House Range, western Utah. Cambrian Stage 10."
    )

    norm, align = normalize_text(raw)

    print("Original caption:")
    print(f"  {raw}")
    print()

    print("Normalized caption:")
    print(f"  {norm}")
    print()

    # Simulate NER extracting entities (find actual positions)
    print("Simulated NER results (on normalized text):")

    # Define entities to find in normalized text
    entities_to_find = [
        ("TAXON", "Olenellus_wheeleri"),
        ("TAXON", "Elrathia_kingii"),
        ("STRAT", "Wheeler_Formation"),
        ("LOC", "House_Range"),
        ("CHRONO", "Cambrian_Stage_10"),
    ]

    for entity_type, norm_entity in entities_to_find:
        # Find entity in normalized text
        start = norm.find(norm_entity)
        if start == -1:
            print(f"  {entity_type:8s} | '{norm_entity}' NOT FOUND")
            continue

        end = start + len(norm_entity)
        norm_span = (start, end)

        # Project back to raw text
        raw_span = project_span(norm_span, align, "norm_to_raw")
        raw_entity = raw[raw_span[0]:raw_span[1]]

        print(f"  {entity_type:8s} | norm: '{norm_entity}' → raw: '{raw_entity}'")


def demo_statistics():
    """Demonstrate normalization statistics."""
    print("DEMO 6: Normalization Statistics")
    print_separator()

    examples = [
        "Olenellus wheeleri occurs in Wheeler Formation",
        "The Upper Wheeler Formation contains Cambrian Stage 10 trilobites",
        "This is normal text without special terms",
    ]

    for raw in examples:
        norm, align = normalize_text(raw)
        stats = get_normalization_stats(raw, norm)

        print(f"Text: '{raw[:50]}...'")
        print(f"  Raw length:  {stats['char_count_raw']} chars")
        print(f"  Norm length: {stats['char_count_norm']} chars")
        print(f"  Difference:  {stats['char_diff']}")
        print(f"  Spaces (raw): {stats['spaces_raw']}")
        print(f"  Underscores (norm): {stats['underscores_norm']}")
        print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("PaleoBERT Text Normalization - Demo Script")
    print("=" * 80)

    demo_basic_normalization()
    demo_alignment_map()
    demo_span_projection()
    demo_round_trip()
    demo_realistic_example()
    demo_statistics()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
