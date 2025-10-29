#!/usr/bin/env python3
"""
Build PaleoBERT tokenizer by adding domain-specific vocabulary to DeBERTa-v3-base.

Usage:
    python scripts/build_tokenizer.py

Outputs:
    artifacts/tokenizer_v1/ - Saved tokenizer with added tokens
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer


def load_vocab_file(filepath):
    """Load vocabulary terms from a text file (one term per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms


def build_tokenizer(
    base_model="microsoft/deberta-v3-base",
    vocab_dir="artifacts/vocab",
    output_dir="artifacts/tokenizer_v1"
):
    """
    Build PaleoBERT tokenizer with domain vocabulary.

    Args:
        base_model: Hugging Face model ID for base tokenizer
        vocab_dir: Directory containing vocab files (taxa.txt, strat_units.txt, etc.)
        output_dir: Output directory for tokenizer_v1
    """
    print(f"Loading base tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size:,}")

    # Load domain vocabulary files
    vocab_dir = Path(vocab_dir)
    vocab_files = {
        "taxa": vocab_dir / "taxa.txt",
        "strat_units": vocab_dir / "strat_units.txt",
        "chrono_units": vocab_dir / "chrono_units.txt",
        "localities": vocab_dir / "localities.txt",
    }

    all_tokens = []
    stats = {}

    for category, filepath in vocab_files.items():
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        tokens = load_vocab_file(filepath)
        stats[category] = len(tokens)
        all_tokens.extend(tokens)
        print(f"Loaded {len(tokens)} tokens from {category}")

    # Add tokens to tokenizer
    print(f"\nAdding {len(all_tokens)} domain tokens to tokenizer...")
    num_added = tokenizer.add_tokens(all_tokens)
    print(f"Successfully added {num_added} new tokens")

    new_vocab_size = len(tokenizer)
    print(f"New vocabulary size: {new_vocab_size:,}")
    print(f"Increase: +{new_vocab_size - original_vocab_size:,} tokens")

    # Save tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTokenizer saved to: {output_dir}")

    # Print summary
    print("\n" + "="*60)
    print("TOKENIZER BUILD SUMMARY")
    print("="*60)
    print(f"Base model:        {base_model}")
    print(f"Original vocab:    {original_vocab_size:,}")
    print(f"Added tokens:      {num_added}")
    print(f"Final vocab:       {new_vocab_size:,}")
    print("\nTokens by category:")
    for category, count in stats.items():
        print(f"  {category:15s}: {count:3d} tokens")
    print("="*60)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Build PaleoBERT tokenizer with domain vocabulary"
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/deberta-v3-base",
        help="Base model for tokenizer (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--vocab-dir",
        default="artifacts/vocab",
        help="Directory containing vocabulary files (default: artifacts/vocab)"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/tokenizer_v1",
        help="Output directory for tokenizer (default: artifacts/tokenizer_v1)"
    )

    args = parser.parse_args()

    build_tokenizer(
        base_model=args.base_model,
        vocab_dir=args.vocab_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
