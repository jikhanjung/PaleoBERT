#!/usr/bin/env python3
"""
Validate PaleoBERT tokenizer by measuring fragmentation rates and testing on sample text.

Usage:
    python scripts/validate_tokenizer.py
    python scripts/validate_tokenizer.py --tokenizer artifacts/tokenizer_v1

Validates:
    - Fragmentation rate for domain vocabulary
    - Token counts for sample paleontology text
    - Comparison with base tokenizer
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer


def load_vocab_file(filepath):
    """Load vocabulary terms from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms


def calculate_fragmentation_rate(tokenizer, terms):
    """
    Calculate the percentage of terms that are fragmented (split into >1 token).

    Args:
        tokenizer: Hugging Face tokenizer
        terms: List of vocabulary terms

    Returns:
        Tuple of (fragmentation_rate, fragmented_terms, single_token_terms)
    """
    fragmented = []
    single_token = []

    for term in terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        if len(tokens) > 1:
            fragmented.append((term, tokens))
        else:
            single_token.append(term)

    fragmentation_rate = len(fragmented) / len(terms) * 100 if terms else 0
    return fragmentation_rate, fragmented, single_token


def test_sample_text(tokenizer, text, name="Sample"):
    """
    Tokenize sample text and return statistics.

    Args:
        tokenizer: Hugging Face tokenizer
        text: Input text
        name: Name for this test sample

    Returns:
        Dict with token count and tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    return {
        "name": name,
        "text": text,
        "token_count": len(tokens),
        "tokens": token_strs
    }


def validate_tokenizer(
    tokenizer_path="artifacts/tokenizer_v1",
    vocab_dir="artifacts/vocab",
    base_model="microsoft/deberta-v3-base",
    compare_base=False
):
    """
    Validate PaleoBERT tokenizer.

    Args:
        tokenizer_path: Path to PaleoBERT tokenizer
        vocab_dir: Directory containing vocabulary files
        base_model: Base model for comparison (optional)
        compare_base: Whether to compare with base tokenizer
    """
    print("="*70)
    print("PALEOBERT TOKENIZER VALIDATION")
    print("="*70)

    # Load PaleoBERT tokenizer
    print(f"\nLoading PaleoBERT tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer):,}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("\nNote: Run 'python scripts/build_tokenizer.py' first to create the tokenizer.")
        return

    # Load vocabulary files
    vocab_dir = Path(vocab_dir)
    vocab_files = {
        "taxa": vocab_dir / "taxa.txt",
        "strat_units": vocab_dir / "strat_units.txt",
        "chrono_units": vocab_dir / "chrono_units.txt",
        "localities": vocab_dir / "localities.txt",
    }

    print("\n" + "="*70)
    print("FRAGMENTATION RATE ANALYSIS")
    print("="*70)

    all_results = {}

    for category, filepath in vocab_files.items():
        if not filepath.exists():
            print(f"\n✗ {category}: File not found ({filepath})")
            continue

        terms = load_vocab_file(filepath)
        frag_rate, fragmented, single_token = calculate_fragmentation_rate(tokenizer, terms)

        all_results[category] = {
            "total": len(terms),
            "fragmented": len(fragmented),
            "single_token": len(single_token),
            "frag_rate": frag_rate,
            "fragmented_examples": fragmented[:5]  # First 5 examples
        }

        print(f"\n{category.upper()}")
        print(f"  Total terms:        {len(terms)}")
        print(f"  Single token:       {len(single_token)} ({len(single_token)/len(terms)*100:.1f}%)")
        print(f"  Fragmented:         {len(fragmented)} ({frag_rate:.1f}%)")

        if fragmented and len(fragmented) <= 10:
            print(f"  Fragmented terms:")
            for term, tokens in fragmented[:5]:
                token_strs = tokenizer.convert_ids_to_tokens(tokens)
                print(f"    • {term:25s} → {token_strs}")

    # Overall statistics
    total_terms = sum(r["total"] for r in all_results.values())
    total_single = sum(r["single_token"] for r in all_results.values())
    total_fragmented = sum(r["fragmented"] for r in all_results.values())

    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total domain terms:     {total_terms}")
    print(f"Single token:           {total_single} ({total_single/total_terms*100:.1f}%)")
    print(f"Fragmented:             {total_fragmented} ({total_fragmented/total_terms*100:.1f}%)")
    print(f"\n✓ Target: 100% single token (0% fragmentation)")

    if total_fragmented == 0:
        print("✓ EXCELLENT: All domain terms are single tokens!")
    elif total_fragmented / total_terms < 0.05:
        print("✓ GOOD: <5% fragmentation rate")
    else:
        print("⚠ WARNING: Fragmentation rate higher than expected")

    # Sample text testing
    print("\n" + "="*70)
    print("SAMPLE TEXT TOKENIZATION")
    print("="*70)

    samples = [
        (
            "Sample 1: Entity-rich caption",
            "Olenellus wheeleri from the Wheeler Formation, House Range, Utah. Cambrian Stage 10."
        ),
        (
            "Sample 2: Systematic description",
            "Asaphiscus bonnensis occurs in the Marjum Formation Middle Member."
        ),
        (
            "Sample 3: Multiple taxa",
            "Elrathia kingii and Peronopsis interstricta are common in the Spence Shale."
        ),
    ]

    for name, text in samples:
        result = test_sample_text(tokenizer, text, name)
        print(f"\n{result['name']}")
        print(f"  Text: {result['text']}")
        print(f"  Token count: {result['token_count']}")
        print(f"  Tokens: {' | '.join(result['tokens'][:15])}")
        if len(result['tokens']) > 15:
            print(f"          ... ({len(result['tokens']) - 15} more)")

    # Comparison with base tokenizer (optional)
    if compare_base:
        print("\n" + "="*70)
        print("COMPARISON WITH BASE TOKENIZER")
        print("="*70)

        try:
            base_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            print(f"\nBase model: {base_model}")
            print(f"Base vocabulary size: {len(base_tokenizer):,}")

            # Compare on sample text
            sample_text = "Olenellus wheeleri from the Wheeler Formation."

            paleo_result = test_sample_text(tokenizer, sample_text, "PaleoBERT")
            base_result = test_sample_text(base_tokenizer, sample_text, "Base")

            print(f"\nSample: {sample_text}")
            print(f"\nPaleoBERT tokenizer:")
            print(f"  Tokens: {paleo_result['token_count']}")
            print(f"  {' | '.join(paleo_result['tokens'])}")

            print(f"\nBase tokenizer:")
            print(f"  Tokens: {base_result['token_count']}")
            print(f"  {' | '.join(base_result['tokens'])}")

            improvement = (base_result['token_count'] - paleo_result['token_count']) / base_result['token_count'] * 100
            print(f"\nToken count reduction: {improvement:.1f}%")

        except Exception as e:
            print(f"\n✗ Could not load base tokenizer for comparison: {e}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate PaleoBERT tokenizer"
    )
    parser.add_argument(
        "--tokenizer",
        default="artifacts/tokenizer_v1",
        help="Path to PaleoBERT tokenizer (default: artifacts/tokenizer_v1)"
    )
    parser.add_argument(
        "--vocab-dir",
        default="artifacts/vocab",
        help="Directory containing vocabulary files (default: artifacts/vocab)"
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/deberta-v3-base",
        help="Base model for comparison (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Compare with base tokenizer"
    )

    args = parser.parse_args()

    validate_tokenizer(
        tokenizer_path=args.tokenizer,
        vocab_dir=args.vocab_dir,
        base_model=args.base_model,
        compare_base=args.compare_base
    )


if __name__ == "__main__":
    main()
