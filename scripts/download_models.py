#!/usr/bin/env python3
"""
Download required models from Hugging Face for offline use.

This script pre-downloads models to HuggingFace cache to avoid
download delays during training.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoConfig
)

def download_models():
    """Download all required models."""

    print("=" * 60)
    print("Downloading Hugging Face models...")
    print("=" * 60)

    # Model to download
    model_name = "microsoft/deberta-v3-base"

    print(f"\n1. Downloading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"   ✓ Tokenizer downloaded (vocab size: {len(tokenizer):,})")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n2. Downloading config: {model_name}")
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"   ✓ Config downloaded")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n3. Downloading model for MLM: {model_name}")
    try:
        model_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        print(f"   ✓ MLM model downloaded ({model_mlm.num_parameters():,} parameters)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n4. Downloading model for Token Classification: {model_name}")
    try:
        # Just download the base model, we'll add the head during training
        model_ner = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=9,  # For NER
            ignore_mismatched_sizes=True
        )
        print(f"   ✓ Token Classification model downloaded")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    print("\nModels are cached in ~/.cache/huggingface/")
    print("You can now run training scripts offline.")

    return True

if __name__ == "__main__":
    success = download_models()
    exit(0 if success else 1)
