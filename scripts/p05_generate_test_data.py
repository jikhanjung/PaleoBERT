#!/usr/bin/env python
"""
P05 Phase 3: Generate synthetic test data from trilobite metadata.

Creates NER and RE test examples based on trilobite-formation-locality relationships
extracted from the trilobite catalog PDF.

Usage:
    python scripts/p05_generate_test_data.py

Output:
    - data/ner/test_trilobite.jsonl (NER test set)
    - data/re/test_trilobite.jsonl (RE test set)
"""

import sys
import os
import json
import random
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Sentence templates for generation
SENTENCE_TEMPLATES = [
    # occurs_in relation
    "{taxon} occurs in the {formation}.",
    "The {formation} yields {taxon}.",
    "{taxon} is found in the {formation}.",
    "Abundant {taxon} from the {formation}.",
    "{taxon} specimens occur in the {formation}.",

    # found_at relation
    "{taxon} from {locality}.",
    "{taxon} was collected at {locality}.",
    "Specimens of {taxon} from {locality}.",
    "{taxon} is known from {locality}.",

    # occurs_in + found_at
    "{taxon} from the {formation}, {locality}.",
    "{taxon} occurs in the {formation} at {locality}.",
    "The {formation} in {locality} yields {taxon}.",
    "{taxon} from the {formation} of {locality}.",

    # assigned_to (temporal)
    "{taxon} from {chrono}.",
    "{taxon} is characteristic of {chrono}.",
    "{taxon} occurs in {chrono} strata.",

    # Complex multi-entity
    "{taxon} from the {formation}, {locality}, {chrono}.",
    "{taxon} occurs in the {formation} of {locality} during {chrono}.",
    "The {formation} yields {taxon} in {chrono} strata at {locality}.",
]


def load_trilobite_metadata(json_file: str) -> Dict[str, Dict]:
    """Load trilobite metadata."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_formation(formation: str) -> str:
    """Normalize formation name (spaces to underscores)."""
    return formation.replace(' ', '_')


def normalize_locality(locality: str) -> str:
    """Normalize locality name (spaces to underscores)."""
    return locality.replace(' ', '_')


def age_to_chrono(age: str) -> str:
    """Convert age code to chronostratigraphic term."""
    age_map = {
        'LCAM': 'Lower_Cambrian',
        'MCAM': 'Middle_Cambrian',
        'UCAM': 'Upper_Cambrian',
        'CAM': 'Cambrian',
        'LCAM-MCAM': 'Lower_to_Middle_Cambrian',
        'MCAM-UCAM': 'Middle_to_Upper_Cambrian',
    }
    return age_map.get(age, 'Cambrian')


def generate_ner_example(
    taxon: str,
    formation: str = None,
    locality: str = None,
    chrono: str = None,
    template: str = None
) -> Dict:
    """
    Generate a single NER example.

    Returns:
        Dict with 'text' and 'entities' fields
    """
    # Select template
    if template is None:
        # Choose template based on available entities
        if formation and locality and chrono:
            candidates = [t for t in SENTENCE_TEMPLATES if '{formation}' in t and '{locality}' in t and '{chrono}' in t]
        elif formation and locality:
            candidates = [t for t in SENTENCE_TEMPLATES if '{formation}' in t and '{locality}' in t and '{chrono}' not in t]
        elif formation:
            candidates = [t for t in SENTENCE_TEMPLATES if '{formation}' in t and '{locality}' not in t and '{chrono}' not in t]
        elif locality:
            candidates = [t for t in SENTENCE_TEMPLATES if '{locality}' in t and '{formation}' not in t and '{chrono}' not in t]
        elif chrono:
            candidates = [t for t in SENTENCE_TEMPLATES if '{chrono}' in t and '{formation}' not in t and '{locality}' not in t]
        else:
            candidates = ["{taxon}."]

        if candidates:
            template = random.choice(candidates)
        else:
            template = "{taxon}."

    # Fill template
    text = template.format(
        taxon=taxon,
        formation=formation or '',
        locality=locality or '',
        chrono=chrono or ''
    )

    # Extract entity positions
    entities = []

    # Find taxon
    pos = text.find(taxon)
    if pos >= 0:
        entities.append({
            'start': pos,
            'end': pos + len(taxon),
            'label': 'TAXON',
            'text': taxon
        })

    # Find formation
    if formation:
        pos = text.find(formation)
        if pos >= 0:
            entities.append({
                'start': pos,
                'end': pos + len(formation),
                'label': 'STRAT',
                'text': formation
            })

    # Find locality
    if locality:
        pos = text.find(locality)
        if pos >= 0:
            entities.append({
                'start': pos,
                'end': pos + len(locality),
                'label': 'LOC',
                'text': locality
            })

    # Find chrono
    if chrono:
        pos = text.find(chrono)
        if pos >= 0:
            entities.append({
                'start': pos,
                'end': pos + len(chrono),
                'label': 'CHRONO',
                'text': chrono
            })

    return {
        'text': text,
        'entities': sorted(entities, key=lambda x: x['start'])
    }


def generate_re_example(
    taxon: str,
    formation: str = None,
    locality: str = None,
    chrono: str = None,
    template: str = None
) -> Dict:
    """
    Generate a single RE example.

    Returns:
        Dict with 'text', 'entities', and 'relations' fields
    """
    # Generate NER example first
    ner_example = generate_ner_example(taxon, formation, locality, chrono, template)

    # Add entity IDs
    entities = []
    for i, ent in enumerate(ner_example['entities']):
        entities.append({
            'id': f'e{i+1}',
            'start': ent['start'],
            'end': ent['end'],
            'label': ent['label'],
            'text': ent['text']
        })

    # Generate relations
    relations = []

    # Find entity IDs by label
    taxon_id = None
    formation_id = None
    locality_id = None
    chrono_id = None

    for ent in entities:
        if ent['label'] == 'TAXON':
            taxon_id = ent['id']
        elif ent['label'] == 'STRAT':
            formation_id = ent['id']
        elif ent['label'] == 'LOC':
            locality_id = ent['id']
        elif ent['label'] == 'CHRONO':
            chrono_id = ent['id']

    # Add relations
    if taxon_id and formation_id:
        relations.append({
            'head': taxon_id,
            'tail': formation_id,
            'label': 'occurs_in'
        })

    if taxon_id and locality_id:
        relations.append({
            'head': taxon_id,
            'tail': locality_id,
            'label': 'found_at'
        })

    if formation_id and chrono_id:
        relations.append({
            'head': formation_id,
            'tail': chrono_id,
            'label': 'assigned_to'
        })

    return {
        'text': ner_example['text'],
        'entities': entities,
        'relations': relations
    }


def generate_test_datasets(
    metadata: Dict[str, Dict],
    n_examples: int = 100
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate NER and RE test datasets.

    Args:
        metadata: Trilobite metadata dict
        n_examples: Number of examples to generate

    Returns:
        Tuple of (ner_examples, re_examples)
    """
    print(f"\nGenerating {n_examples} test examples...")

    ner_examples = []
    re_examples = []

    # Get all genera
    genera = list(metadata.keys())

    # Sample genera
    if len(genera) > n_examples:
        sampled_genera = random.sample(genera, n_examples)
    else:
        sampled_genera = genera

    for genus in sampled_genera:
        meta = metadata[genus]

        # Get available data
        formations = meta.get('formations', [])
        localities = meta.get('localities', [])
        age = meta.get('age', 'CAM')
        chrono = age_to_chrono(age)

        # Select formation and locality (if available)
        formation = random.choice(formations) if formations else None
        locality = random.choice(localities) if localities else None

        # Normalize
        if formation:
            formation = normalize_formation(formation)
        if locality:
            locality = normalize_locality(locality)

        # Generate examples
        ner_example = generate_ner_example(genus, formation, locality, chrono)
        re_example = generate_re_example(genus, formation, locality, chrono)

        ner_examples.append(ner_example)
        re_examples.append(re_example)

    print(f"  Generated {len(ner_examples)} NER examples")
    print(f"  Generated {len(re_examples)} RE examples")

    return ner_examples, re_examples


def save_jsonl(data: List[Dict], output_path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved: {output_path}")


def validate_dataset(data: List[Dict], dataset_type: str):
    """Validate dataset format."""
    print(f"\nValidating {dataset_type} dataset...")

    errors = 0

    for i, example in enumerate(data):
        # Check required fields
        if 'text' not in example:
            print(f"  ERROR: Example {i} missing 'text'")
            errors += 1

        if 'entities' not in example:
            print(f"  ERROR: Example {i} missing 'entities'")
            errors += 1
            continue

        # Validate entities
        for j, entity in enumerate(example['entities']):
            if 'start' not in entity or 'end' not in entity or 'label' not in entity:
                print(f"  ERROR: Example {i}, entity {j} missing required fields")
                errors += 1
                continue

            # Validate span
            start = entity['start']
            end = entity['end']
            text = example['text']

            if start < 0 or end > len(text) or start >= end:
                print(f"  ERROR: Example {i}, entity {j} invalid span: {start}-{end}")
                errors += 1
            else:
                extracted = text[start:end]
                expected = entity.get('text', '')
                if extracted != expected:
                    print(f"  WARNING: Example {i}, entity {j} span mismatch")
                    print(f"    Expected: '{expected}'")
                    print(f"    Extracted: '{extracted}'")

    if errors == 0:
        print(f"  ✓ All {len(data)} examples valid")
    else:
        print(f"  ✗ Found {errors} errors")

    return errors == 0


def print_sample_examples(ner_examples: List[Dict], re_examples: List[Dict], n: int = 5):
    """Print sample examples."""
    print(f"\n{'='*80}")
    print(f"SAMPLE NER EXAMPLES (first {n})")
    print('='*80)

    for i, example in enumerate(ner_examples[:n], 1):
        print(f"\nExample {i}:")
        print(f"  Text: {example['text']}")
        print(f"  Entities:")
        for ent in example['entities']:
            print(f"    - {ent['label']:8} | {ent['text']:30} | [{ent['start']}, {ent['end']}]")

    print(f"\n{'='*80}")
    print(f"SAMPLE RE EXAMPLES (first {n})")
    print('='*80)

    for i, example in enumerate(re_examples[:n], 1):
        print(f"\nExample {i}:")
        print(f"  Text: {example['text']}")
        print(f"  Entities:")
        for ent in example['entities']:
            print(f"    - {ent['id']:3} | {ent['label']:8} | {ent['text']}")
        print(f"  Relations:")
        if example['relations']:
            for rel in example['relations']:
                print(f"    - {rel['label']:15} | {rel['head']} → {rel['tail']}")
        else:
            print(f"    (none)")


def main():
    """Main execution function."""
    print("=" * 80)
    print("P05 Phase 3: Generate Synthetic Test Data")
    print("=" * 80)

    # Paths
    metadata_file = "data/trilobite_metadata.json"
    ner_output = "data/ner/test_trilobite.jsonl"
    re_output = "data/re/test_trilobite.jsonl"

    # Check input file
    if not os.path.exists(metadata_file):
        print(f"ERROR: Metadata not found at {metadata_file}")
        print("Please run: python scripts/p05_extract_trilobite_names.py first")
        return 1

    # Load metadata
    print("\n[1/5] Loading trilobite metadata...")
    metadata = load_trilobite_metadata(metadata_file)
    print(f"  Loaded metadata for {len(metadata)} genera")

    # Set random seed for reproducibility
    random.seed(42)

    # Generate test datasets
    print("\n[2/5] Generating test examples...")
    ner_examples, re_examples = generate_test_datasets(metadata, n_examples=100)

    # Validate datasets
    print("\n[3/5] Validating datasets...")
    ner_valid = validate_dataset(ner_examples, "NER")
    re_valid = validate_dataset(re_examples, "RE")

    if not (ner_valid and re_valid):
        print("\nERROR: Dataset validation failed")
        return 1

    # Save datasets
    print("\n[4/5] Saving datasets...")
    save_jsonl(ner_examples, ner_output)
    save_jsonl(re_examples, re_output)

    # Print samples
    print("\n[5/5] Sample examples...")
    print_sample_examples(ner_examples, re_examples, n=5)

    # Summary statistics
    entity_counts = {'TAXON': 0, 'STRAT': 0, 'LOC': 0, 'CHRONO': 0}
    relation_counts = {'occurs_in': 0, 'found_at': 0, 'assigned_to': 0}

    for example in ner_examples:
        for ent in example['entities']:
            entity_counts[ent['label']] = entity_counts.get(ent['label'], 0) + 1

    for example in re_examples:
        for rel in example['relations']:
            relation_counts[rel['label']] = relation_counts.get(rel['label'], 0) + 1

    # Success summary
    print("\n" + "=" * 80)
    print("✓ P05 Phase 3 COMPLETE: Test Data Generated")
    print("=" * 80)
    print(f"  Output files:")
    print(f"    - {ner_output} ({len(ner_examples)} examples)")
    print(f"    - {re_output} ({len(re_examples)} examples)")
    print(f"\n  Entity coverage:")
    for label, count in sorted(entity_counts.items()):
        print(f"    {label:8}: {count:3} entities")
    print(f"\n  Relation coverage:")
    for label, count in sorted(relation_counts.items()):
        print(f"    {label:15}: {count:3} relations")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
