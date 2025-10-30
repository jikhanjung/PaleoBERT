#!/usr/bin/env python
"""
Generate Relation Extraction (RE) training data from NER annotations.

Extracts entity pairs from NER-annotated sentences, applies pattern-based
relation labeling, and performs negative sampling to create balanced RE dataset.

Usage:
    python scripts/generate_re_samples.py

Input:
    - artifacts/ner_data/train.jsonl (NER annotations)
    - artifacts/ner_data/dev.jsonl
    - artifacts/ner_data/test.jsonl

Output:
    - artifacts/re_data/train.jsonl (RE training pairs)
    - artifacts/re_data/dev.jsonl (RE dev pairs)
    - artifacts/re_data/test.jsonl (RE test pairs)
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Relation label mapping
LABEL_MAP = {
    "NO_RELATION": 0,
    "occurs_in": 1,
    "found_at": 2,
    "assigned_to": 3,
    "part_of": 4
}

# Valid entity type pairs for each relation
VALID_PAIRS = {
    "occurs_in": [("TAXON", "STRAT")],
    "found_at": [("TAXON", "LOC")],
    "assigned_to": [("STRAT", "CHRONO")],
    "part_of": [("STRAT", "STRAT")],
}


class REDataGenerator:
    """Generate RE training data from NER annotations."""

    def __init__(self, max_distance: int = 20, negative_ratio: float = 2.0):
        """
        Initialize RE data generator.

        Args:
            max_distance: Maximum token distance between entities in a pair
            negative_ratio: Ratio of negative to positive examples (2.0 = 2:1)
        """
        self.max_distance = max_distance
        self.negative_ratio = negative_ratio
        self.relation_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for relation detection."""
        patterns = {
            "occurs_in": [
                # "X occurs in Y", "X found in Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\b(occurs?|found|collected|recorded|recovered)\b[^[]*?\bin\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "X from Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bfrom\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "Y yields X", "Y contains X"
                re.compile(r'\[OBJ\][^\]]+\[/OBJ\][^[]*?\b(yields?|contains?|produces?)\b[^[]*?\[SUBJ\]', re.IGNORECASE),
                # "X in Y" (direct)
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bin\b[^[]*?the\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "X of Y" (context-dependent)
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bof\b[^[]*?the\b[^[]*?\[OBJ\]', re.IGNORECASE),
            ],
            "found_at": [
                # "X found at Y", "X from Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\b(found|collected|recovered)\b[^[]*?\b(at|from|in)\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "X from Y" (when Y is LOC)
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bfrom\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "Y yields X"
                re.compile(r'\[OBJ\][^\]]+\[/OBJ\][^[]*?\byields?\b[^[]*?\[SUBJ\]', re.IGNORECASE),
                # "X at Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bat\b[^[]*?\[OBJ\]', re.IGNORECASE),
            ],
            "assigned_to": [
                # "X assigned to Y", "X correlated to Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\b(assigned|correlated|dated|attributed)\b[^[]*?\b(to|with)\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "X of Y age"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\bof\b[^[]*?\[OBJ\][^\]]+\[/OBJ\][^[]*?\bage\b', re.IGNORECASE),
                # "Y X" (chrono before strat)
                re.compile(r'\[OBJ\][^\]]+\[/OBJ\][^[]*?\[SUBJ\]', re.IGNORECASE),
            ],
            "part_of": [
                # "X Member of Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\b(Member|Bed|Unit)\b[^[]*?\bof\b[^[]*?\[OBJ\]', re.IGNORECASE),
                # "X within Y", "X part of Y"
                re.compile(r'\[SUBJ\][^\]]+\[/SUBJ\][^[]*?\b(within|part of|in)\b[^[]*?the\b[^[]*?\[OBJ\]', re.IGNORECASE),
            ],
        }
        return patterns

    def load_ner_data(self, filepath: str) -> List[Dict]:
        """Load NER annotated data from JSONL."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def extract_entities_from_ner(self, ner_example: Dict) -> List[Dict]:
        """
        Extract entity mentions from NER BIO tags.

        Args:
            ner_example: NER example with tokens and ner_tags

        Returns:
            List of entity dictionaries with type, text, start, end
        """
        tokens = ner_example['tokens']
        tags = ner_example['ner_tags']

        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                entity_type = tag[2:]  # Remove 'B-'
                current_entity = {
                    'type': entity_type,
                    'tokens': [token],
                    'start': i,
                    'end': i + 1,
                }

            elif tag.startswith('I-') and current_entity:
                # Continue current entity
                current_entity['tokens'].append(token)
                current_entity['end'] = i + 1

            elif current_entity:
                # End of entity
                entities.append(current_entity)
                current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

        # Add text field
        for ent in entities:
            ent['text'] = ' '.join(ent['tokens'])

        return entities

    def is_valid_pair(self, subj_type: str, obj_type: str) -> Tuple[bool, List[str]]:
        """
        Check if entity type pair is valid for any relation.

        Args:
            subj_type: Subject entity type
            obj_type: Object entity type

        Returns:
            (is_valid, list_of_possible_relations)
        """
        possible_relations = []

        for relation, valid_pairs in VALID_PAIRS.items():
            if (subj_type, obj_type) in valid_pairs:
                possible_relations.append(relation)

        return len(possible_relations) > 0, possible_relations

    def create_marked_text(self, tokens: List[str], subj: Dict, obj: Dict) -> str:
        """
        Create entity-marked text with [SUBJ] and [OBJ] markers.

        Args:
            tokens: List of tokens
            subj: Subject entity dict
            obj: Object entity dict

        Returns:
            Marked text string
        """
        # Create list to build marked text
        marked_tokens = tokens.copy()

        # Determine which entity comes first
        if subj['start'] < obj['start']:
            first, second = subj, obj
            first_marker = ('[SUBJ]', '[/SUBJ]')
            second_marker = ('[OBJ]', '[/OBJ]')
        else:
            first, second = obj, subj
            first_marker = ('[OBJ]', '[/OBJ]')
            second_marker = ('[SUBJ]', '[/SUBJ]')

        # Insert markers (from back to front to avoid index shifting)
        # Second entity
        marked_tokens.insert(second['end'], second_marker[1])
        marked_tokens.insert(second['start'], second_marker[0])

        # First entity
        marked_tokens.insert(first['end'], first_marker[1])
        marked_tokens.insert(first['start'], first_marker[0])

        return ' '.join(marked_tokens)

    def apply_relation_patterns(self, marked_text: str, subj_type: str, obj_type: str,
                                possible_relations: List[str]) -> str:
        """
        Apply regex patterns to determine relation.

        Args:
            marked_text: Text with entity markers
            subj_type: Subject entity type
            obj_type: Object entity type
            possible_relations: List of possible relations for this type pair

        Returns:
            Detected relation (or "NO_RELATION")
        """
        for relation in possible_relations:
            patterns = self.relation_patterns.get(relation, [])

            for pattern in patterns:
                if pattern.search(marked_text):
                    return relation

        return "NO_RELATION"

    def generate_pairs(self, ner_example: Dict) -> List[Dict]:
        """
        Generate entity pairs from NER example.

        Args:
            ner_example: NER annotated example

        Returns:
            List of entity pair dictionaries
        """
        entities = self.extract_entities_from_ner(ner_example)

        if len(entities) < 2:
            return []

        pairs = []
        tokens = ner_example['tokens']
        text = ' '.join(tokens)

        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue

                # Check distance
                distance = abs(subj['start'] - obj['start'])
                if distance > self.max_distance:
                    continue

                # Check if valid type pair
                is_valid, possible_relations = self.is_valid_pair(subj['type'], obj['type'])

                if not is_valid and random.random() > 0.1:
                    # Skip most invalid pairs (keep 10% for NO_RELATION)
                    continue

                # Create marked text
                marked_text = self.create_marked_text(tokens, subj, obj)

                # Apply patterns to detect relation
                if is_valid:
                    relation = self.apply_relation_patterns(
                        marked_text, subj['type'], obj['type'], possible_relations
                    )
                else:
                    relation = "NO_RELATION"

                # Create pair
                pair = {
                    'text': text,
                    'marked_text': marked_text,
                    'subject': {
                        'type': subj['type'],
                        'text': subj['text'],
                        'start': subj['start'],
                        'end': subj['end'],
                    },
                    'object': {
                        'type': obj['type'],
                        'text': obj['text'],
                        'start': obj['start'],
                        'end': obj['end'],
                    },
                    'relation': relation,
                    'label_id': LABEL_MAP[relation],
                    'metadata': ner_example.get('metadata', {}),
                }

                pairs.append(pair)

        return pairs

    def negative_sampling(self, pairs: List[Dict]) -> List[Dict]:
        """
        Balance dataset with negative sampling.

        Args:
            pairs: List of all entity pairs

        Returns:
            Balanced list with appropriate negative ratio
        """
        positive_pairs = [p for p in pairs if p['relation'] != 'NO_RELATION']
        negative_pairs = [p for p in pairs if p['relation'] == 'NO_RELATION']

        target_negative = int(len(positive_pairs) * self.negative_ratio)

        if len(negative_pairs) > target_negative:
            # Downsample negatives
            negative_pairs = random.sample(negative_pairs, target_negative)
        elif len(negative_pairs) < target_negative:
            # Need more negatives (but we'll use what we have)
            logger.warning(f"Not enough negative samples: {len(negative_pairs)} < {target_negative}")

        balanced_pairs = positive_pairs + negative_pairs
        random.shuffle(balanced_pairs)

        return balanced_pairs

    def process_ner_file(self, ner_filepath: str, output_filepath: str):
        """
        Process NER file and generate RE pairs.

        Args:
            ner_filepath: Path to NER JSONL file
            output_filepath: Path to output RE JSONL file
        """
        logger.info(f"Processing {ner_filepath}")

        # Load NER data
        ner_data = self.load_ner_data(ner_filepath)
        logger.info(f"  Loaded {len(ner_data)} NER examples")

        # Generate pairs
        all_pairs = []
        for ner_example in ner_data:
            pairs = self.generate_pairs(ner_example)
            all_pairs.extend(pairs)

        logger.info(f"  Generated {len(all_pairs)} total pairs")

        # Count relations before balancing
        relation_counts = Counter(p['relation'] for p in all_pairs)
        logger.info(f"  Relation distribution (before balancing):")
        for rel, count in relation_counts.most_common():
            logger.info(f"    {rel}: {count}")

        # Apply negative sampling
        balanced_pairs = self.negative_sampling(all_pairs)
        logger.info(f"  After balancing: {len(balanced_pairs)} pairs")

        # Count relations after balancing
        relation_counts = Counter(p['relation'] for p in balanced_pairs)
        logger.info(f"  Relation distribution (after balancing):")
        for rel, count in relation_counts.most_common():
            pct = count / len(balanced_pairs) * 100
            logger.info(f"    {rel}: {count} ({pct:.1f}%)")

        # Save
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            for pair in balanced_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        logger.info(f"  Saved to {output_filepath}")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("RE Data Generation from NER Annotations")
    logger.info("=" * 80)

    # Configuration
    max_distance = 20  # Maximum token distance between entities
    negative_ratio = 2.0  # 2 negatives per 1 positive

    generator = REDataGenerator(max_distance=max_distance, negative_ratio=negative_ratio)

    # Process each split
    splits = [
        ("artifacts/ner_data/train.jsonl", "artifacts/re_data/train.jsonl"),
        ("artifacts/ner_data/dev.jsonl", "artifacts/re_data/dev.jsonl"),
        ("artifacts/ner_data/test.jsonl", "artifacts/re_data/test.jsonl"),
    ]

    for ner_file, re_file in splits:
        if Path(ner_file).exists():
            generator.process_ner_file(ner_file, re_file)
        else:
            logger.warning(f"NER file not found: {ner_file}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ RE Data Generation Complete")
    logger.info("=" * 80)
    logger.info("\nGenerated files:")
    logger.info("  - artifacts/re_data/train.jsonl")
    logger.info("  - artifacts/re_data/dev.jsonl")
    logger.info("  - artifacts/re_data/test.jsonl")
    logger.info("\nNext steps:")
    logger.info("  1. Review data statistics above")
    logger.info("  2. Inspect sample pairs: head -5 artifacts/re_data/train.jsonl")
    logger.info("  3. Train RE model: python scripts/train_re.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
