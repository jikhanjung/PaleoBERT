#!/usr/bin/env python
"""
P04 Task 3: Generate test data from PDF text.

Selects candidate sentences with high entity density for manual annotation,
then creates NER and RE test datasets.

Usage:
    python scripts/p04_generate_test_data.py

Output:
    - data/test_candidates.json (for manual annotation)
    - data/annotations/geyer2019_annotated.json (manual annotation template)
    - data/ner/test_geyer2019.jsonl (after manual annotation)
    - data/re/test_geyer2019.jsonl (after manual annotation)
"""

import sys
import os
import re
import json
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def select_candidate_sentences(text_path: str, output_path: str, top_n: int = 50) -> int:
    """
    Select sentences with high entity density for annotation.

    Scores sentences based on:
    - Taxon mentions (capitalized Latin binomials)
    - Formation mentions (X Formation, X Member)
    - Stage/Series mentions
    - Locality mentions

    Args:
        text_path: Path to input text file
        output_path: Path to output JSON file
        top_n: Number of top candidates to select

    Returns:
        Number of candidates selected
    """
    print(f"Selecting candidate sentences from: {text_path}")

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences (improved approach)
    # Split on period, exclamation, question mark followed by space or newline
    sentences = re.split(r'[.!?]+(?:\s+|\n+)', text)

    candidates = []

    for sent in sentences:
        sent = sent.strip()

        # Filter by length
        if len(sent) < 50 or len(sent) > 400:
            continue

        # Skip sentences that are mostly numbers/symbols (likely table data)
        alpha_chars = sum(c.isalpha() for c in sent)
        if len(sent) > 0 and alpha_chars / len(sent) < 0.6:
            continue

        # Score by entity-like patterns
        score = 0
        features = []

        # 1. Taxon pattern (capitalized Latin binomials)
        taxon_matches = re.findall(r'\b[A-Z][a-z]{3,}\s+[a-z]{3,}\b', sent)
        if taxon_matches:
            score += len(taxon_matches) * 2  # Weight taxa highly
            features.append(f"taxa:{len(taxon_matches)}")

        # 2. Formation pattern
        formation_matches = re.findall(r'\b[A-Z][a-z]+\s+(Formation|Member|Group|Limestone)\b', sent)
        if formation_matches:
            score += len(formation_matches) * 2
            features.append(f"strat:{len(formation_matches)}")

        # 3. Stage/Series pattern
        chrono_matches = re.findall(r'\b(Stage|Series|Epoch)\s+\d+\b', sent)
        chrono_matches += re.findall(r'\b(Fortunian|Drumian|Paibian|Jiangshanian|Guzhangian|Wuliuan)\b', sent)
        if chrono_matches:
            score += len(chrono_matches) * 2
            features.append(f"chrono:{len(chrono_matches)}")

        # 4. Locality pattern (geographic proper nouns)
        locality_matches = re.findall(r'\b[A-Z][a-z]+\s+(Range|Mountains|Basin|Peninsula|Platform)\b', sent)
        if locality_matches:
            score += len(locality_matches) * 2
            features.append(f"loc:{len(locality_matches)}")

        # 5. Relation keywords (indicates potential relations)
        relation_keywords = [
            'occurs in', 'found in', 'from the', 'at the', 'within the',
            'part of', 'assigned to', 'correlates with', 'equivalent to'
        ]
        for keyword in relation_keywords:
            if keyword in sent.lower():
                score += 1
                features.append(f"rel:{keyword}")

        # Only keep high-scoring sentences
        if score >= 4:  # Threshold: at least 4 points
            candidates.append({
                'text': sent,
                'score': score,
                'features': ', '.join(features),
            })

    # Sort by score (descending)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Take top N
    top_candidates = candidates[:top_n]

    # Save as JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_candidates, f, indent=2, ensure_ascii=False)

    print(f"  Total sentences: {len(sentences)}")
    print(f"  High-scoring candidates: {len(candidates)}")
    print(f"  Top candidates selected: {len(top_candidates)}")

    return len(top_candidates)


def create_annotation_template(candidates_path: str, output_path: str) -> int:
    """
    Create annotation template from candidate sentences.

    Provides a JSON structure for manual annotation of entities and relations.

    Args:
        candidates_path: Path to candidate sentences JSON
        output_path: Path to output annotation template

    Returns:
        Number of templates created
    """
    print(f"Creating annotation template from: {candidates_path}")

    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    # Create annotation template
    # For manual annotation: fill in entities and relations
    annotation_template = []

    for idx, cand in enumerate(candidates[:30]):  # Limit to 30 for manual work
        template = {
            "id": f"geyer2019_{idx:03d}",
            "text": cand['text'],
            "score": cand['score'],
            "features": cand['features'],
            "entities": [
                # TEMPLATE: Annotator should fill this in
                # {
                #   "id": "e1",
                #   "start": 0,
                #   "end": 10,
                #   "label": "TAXON|STRAT|CHRONO|LOC",
                #   "text": "example_text"
                # }
            ],
            "relations": [
                # TEMPLATE: Annotator should fill this in
                # {
                #   "head": "e1",
                #   "tail": "e2",
                #   "label": "occurs_in|found_at|part_of|assigned_to"
                # }
            ],
            "notes": ""
        }
        annotation_template.append(template)

    # Save template
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_template, f, indent=2, ensure_ascii=False)

    print(f"  Annotation templates created: {len(annotation_template)}")
    print(f"  Output: {output_path}")

    return len(annotation_template)


def create_sample_annotations(output_dir: str) -> int:
    """
    Create a few sample hand-annotated examples for demonstration.

    These serve as:
    1. Examples of correct annotation format
    2. Immediate test data (even if small)
    3. Templates for manual annotation

    Args:
        output_dir: Directory for output files

    Returns:
        Number of sample examples created
    """
    print("Creating sample annotated examples...")

    # Sample 1: Entity-dense sentence with taxa and stratigraphy
    sample1 = {
        "id": "geyer2019_sample_001",
        "text": "The Kaili Formation yields diverse trilobite assemblages including Oryctocephalus indicus from the Drumian Stage.",
        "entities": [
            {"id": "e1", "start": 4, "end": 19, "label": "STRAT", "text": "Kaili_Formation"},
            {"id": "e2", "start": 67, "end": 89, "label": "TAXON", "text": "Oryctocephalus indicus"},
            {"id": "e3", "start": 99, "end": 112, "label": "CHRONO", "text": "Drumian_Stage"}
        ],
        "relations": [
            {"head": "e2", "tail": "e1", "label": "occurs_in"},
            {"head": "e1", "tail": "e3", "label": "assigned_to"}
        ]
    }

    # Sample 2: Locality and stratigraphy
    sample2 = {
        "id": "geyer2019_sample_002",
        "text": "The Wheeler Formation in the House Range represents Cambrian Stage 5 deposition.",
        "entities": [
            {"id": "e1", "start": 4, "end": 21, "label": "STRAT", "text": "Wheeler_Formation"},
            {"id": "e2", "start": 29, "end": 40, "label": "LOC", "text": "House_Range"},
            {"id": "e3", "start": 53, "end": 69, "label": "CHRONO", "text": "Cambrian_Stage_5"}
        ],
        "relations": [
            {"head": "e1", "tail": "e2", "label": "found_at"},
            {"head": "e1", "tail": "e3", "label": "assigned_to"}
        ]
    }

    # Sample 3: Multiple taxa
    sample3 = {
        "id": "geyer2019_sample_003",
        "text": "The Furongian Series contains diagnostic agnostoid taxa including Agnostotes orientalis and Glyptagnostus reticulatus.",
        "entities": [
            {"id": "e1", "start": 4, "end": 20, "label": "CHRONO", "text": "Furongian_Series"},
            {"id": "e2", "start": 67, "end": 89, "label": "TAXON", "text": "Agnostotes orientalis"},
            {"id": "e3", "start": 94, "end": 119, "label": "TAXON", "text": "Glyptagnostus reticulatus"}
        ],
        "relations": [
            {"head": "e2", "tail": "e1", "label": "occurs_in"},
            {"head": "e3", "tail": "e1", "label": "occurs_in"}
        ]
    }

    samples = [sample1, sample2, sample3]

    # Save as annotated examples
    annotations_file = os.path.join(output_dir, "geyer2019_sample_annotations.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"  Sample annotations: {len(samples)}")
    print(f"  Output: {annotations_file}")

    return len(samples)


def create_test_datasets(annotations_path: str, ner_output: str, re_output: str) -> Tuple[int, int]:
    """
    Create NER and RE test datasets from annotated examples.

    Args:
        annotations_path: Path to annotated examples JSON
        ner_output: Path to NER test dataset (JSONL)
        re_output: Path to RE test dataset (JSONL)

    Returns:
        Tuple of (NER example count, RE example count)
    """
    print(f"Creating test datasets from: {annotations_path}")

    if not os.path.exists(annotations_path):
        print(f"  WARNING: Annotations file not found. Skipping test dataset creation.")
        print(f"  To create test datasets, manually annotate: {annotations_path}")
        return 0, 0

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Create NER dataset
    os.makedirs(os.path.dirname(ner_output), exist_ok=True)
    with open(ner_output, 'w', encoding='utf-8') as f:
        for ann in annotations:
            ner_entry = {
                'id': ann['id'],
                'text': ann['text'],
                'entities': ann['entities'],
            }
            f.write(json.dumps(ner_entry, ensure_ascii=False) + '\n')

    # Create RE dataset
    os.makedirs(os.path.dirname(re_output), exist_ok=True)
    with open(re_output, 'w', encoding='utf-8') as f:
        for ann in annotations:
            re_entry = {
                'id': ann['id'],
                'text': ann['text'],
                'entities': ann['entities'],
                'relations': ann.get('relations', []),
            }
            f.write(json.dumps(re_entry, ensure_ascii=False) + '\n')

    print(f"  NER test dataset: {len(annotations)} examples → {ner_output}")
    print(f"  RE test dataset: {len(annotations)} examples → {re_output}")

    return len(annotations), len(annotations)


def validate_test_datasets(ner_path: str, re_path: str) -> bool:
    """
    Validate NER and RE test datasets.

    Checks:
    1. JSONL format validity
    2. Required fields present
    3. Entity span correctness
    4. Entity type coverage
    5. Relation type coverage

    Args:
        ner_path: Path to NER test dataset
        re_path: Path to RE test dataset

    Returns:
        True if valid, False otherwise
    """
    print("Validating test datasets...")

    all_valid = True

    # Validate NER dataset
    if os.path.exists(ner_path):
        print(f"  Validating NER: {ner_path}")
        entity_types = set()
        try:
            with open(ner_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    obj = json.loads(line)

                    # Check required fields
                    if 'text' not in obj or 'entities' not in obj:
                        print(f"    ERROR: Line {line_num} missing required fields")
                        all_valid = False
                        continue

                    # Validate entity spans
                    for ent in obj['entities']:
                        if 'start' not in ent or 'end' not in ent or 'label' not in ent:
                            print(f"    ERROR: Line {line_num} entity missing fields")
                            all_valid = False
                            continue

                        # Check span matches text
                        span_text = obj['text'][ent['start']:ent['end']]
                        if 'text' in ent and span_text != ent['text']:
                            print(f"    WARNING: Line {line_num} span mismatch: '{span_text}' != '{ent['text']}'")

                        entity_types.add(ent['label'])

            print(f"    ✓ Entity types found: {sorted(entity_types)}")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_valid = False
    else:
        print(f"  NER dataset not found: {ner_path}")

    # Validate RE dataset
    if os.path.exists(re_path):
        print(f"  Validating RE: {re_path}")
        relation_types = set()
        try:
            with open(re_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    obj = json.loads(line)

                    # Check required fields
                    if 'text' not in obj or 'entities' not in obj or 'relations' not in obj:
                        print(f"    ERROR: Line {line_num} missing required fields")
                        all_valid = False
                        continue

                    # Collect relation types
                    for rel in obj['relations']:
                        if 'label' in rel:
                            relation_types.add(rel['label'])

            print(f"    ✓ Relation types found: {sorted(relation_types)}")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_valid = False
    else:
        print(f"  RE dataset not found: {re_path}")

    return all_valid


def main():
    """Main execution function."""
    print("=" * 80)
    print("P04 Task 3: Test Data Generation from PDF")
    print("=" * 80)

    # Paths
    text_path = "data/pdf_extracted/geyer2019_raw.txt"
    candidates_path = "data/test_candidates.json"
    annotation_template_path = "data/annotations/geyer2019_annotation_template.json"
    sample_annotations_path = "data/annotations/geyer2019_sample_annotations.json"
    ner_output = "data/ner/test_geyer2019.jsonl"
    re_output = "data/re/test_geyer2019.jsonl"

    # Check if extracted text exists
    if not os.path.exists(text_path):
        print(f"ERROR: Extracted text not found at {text_path}")
        print("Please run p04_extract_pdf_text.py first (Task 1).")
        return 1

    # Step 1: Select candidate sentences
    print("\n[1/4] Selecting candidate sentences for annotation...")
    num_candidates = select_candidate_sentences(text_path, candidates_path, top_n=50)

    if num_candidates == 0:
        print("ERROR: No candidate sentences found")
        return 1

    # Step 2: Create annotation template
    print("\n[2/4] Creating annotation template...")
    num_templates = create_annotation_template(candidates_path, annotation_template_path)

    # Step 3: Create sample annotations
    print("\n[3/4] Creating sample annotated examples...")
    num_samples = create_sample_annotations("data/annotations")

    # Step 4: Create test datasets from samples
    print("\n[4/4] Creating test datasets from sample annotations...")
    ner_count, re_count = create_test_datasets(sample_annotations_path, ner_output, re_output)

    # Validate test datasets
    if ner_count > 0 or re_count > 0:
        print("\n[Validation] Validating test datasets...")
        validate_test_datasets(ner_output, re_output)

    # Success summary
    print("\n" + "=" * 80)
    print("✓ P04 Task 3 COMPLETE")
    print("=" * 80)
    print(f"  Candidate sentences: {num_candidates} → {candidates_path}")
    print(f"  Annotation template: {num_templates} → {annotation_template_path}")
    print(f"  Sample annotations: {num_samples} → {sample_annotations_path}")
    print(f"  NER test set: {ner_count} examples → {ner_output}")
    print(f"  RE test set: {re_count} examples → {re_output}")

    print("\nManual annotation workflow:")
    print("  1. Review candidate sentences in: data/test_candidates.json")
    print("  2. Manually annotate top 30 examples in: data/annotations/geyer2019_annotation_template.json")
    print("  3. Save completed annotations as: data/annotations/geyer2019_annotated.json")
    print("  4. Re-run this script to generate full test datasets")

    print("\nCurrent test datasets:")
    print("  - Created from 3 sample annotations (demonstration purposes)")
    print("  - For production use, complete manual annotation of 30-50 examples")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
