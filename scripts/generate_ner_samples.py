#!/usr/bin/env python3
"""
Auto-generate NER training samples from corpus using vocabulary lists.

This script:
1. Loads existing corpus (1.57M tokens)
2. Uses vocabulary lists to auto-tag entities
3. Generates initial BIO-tagged samples
4. Creates train/dev/test split
5. Outputs JSONL format for NER training

Expected accuracy: ~60-70% (requires manual review)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NERAutoAnnotator:
    """Auto-annotate entities using vocabulary lists."""

    def __init__(self, vocab_dir: Path):
        """Initialize with vocabulary directory."""
        self.vocab_dir = vocab_dir
        self.taxa = set()
        self.strat_units = set()
        self.chrono_units = set()
        self.localities = set()

        # Load vocabularies
        self._load_vocabularies()

        # Compile patterns
        self._compile_patterns()

    def _load_vocabularies(self):
        """Load all vocabulary files."""
        logger.info("Loading vocabularies...")

        # Load taxa
        taxa_file = self.vocab_dir / "taxa.txt"
        if taxa_file.exists():
            with open(taxa_file) as f:
                self.taxa = {line.strip() for line in f if line.strip()}
            logger.info(f"  Loaded {len(self.taxa)} taxa")

        # Load stratigraphic units
        strat_file = self.vocab_dir / "strat_units.txt"
        if strat_file.exists():
            with open(strat_file) as f:
                self.strat_units = {line.strip() for line in f if line.strip()}
            logger.info(f"  Loaded {len(self.strat_units)} stratigraphic units")

        # Load chronostratigraphic units
        chrono_file = self.vocab_dir / "chrono_units.txt"
        if chrono_file.exists():
            with open(chrono_file) as f:
                self.chrono_units = {line.strip() for line in f if line.strip()}
            logger.info(f"  Loaded {len(self.chrono_units)} chrono units")

        # Load localities
        loc_file = self.vocab_dir / "localities.txt"
        if loc_file.exists():
            with open(loc_file) as f:
                self.localities = {line.strip() for line in f if line.strip()}
            logger.info(f"  Loaded {len(self.localities)} localities")

    def _compile_patterns(self):
        """Compile regex patterns for entity recognition."""
        # Stratigraphic unit patterns
        self.strat_patterns = [
            r'\b(\w+)\s+(Formation|Member|Group|Shale|Limestone|Sandstone)\b',
            r'\b(Lower|Middle|Upper)\s+(\w+)\s+(Formation|Member)\b',
        ]

        # Chronostratigraphic patterns
        self.chrono_patterns = [
            r'\b(Cambrian|Ordovician)\s+(Stage|Series)\s+(\d+)\b',
            r'\b(Lower|Middle|Upper|Early|Late)\s+(Cambrian|Ordovician)\b',
            r'\b(Furongian|Miaolingian|Terreneuvian)\b',
            r'\b(Jiangshanian|Paibian|Guzhangian|Drumian|Wuliuan)\b',
        ]

    def _tokenize_sentence(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        # Split on whitespace and punctuation
        tokens = []
        current_token = ""

        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '.,;:!?()[]{}':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens

    def _match_multiword_entity(self, tokens: List[str], start_idx: int, entity_set: Set[str]) -> Tuple[str, int]:
        """Match multi-word entities from vocabulary."""
        best_match = None
        best_length = 0

        # Try matching 1-5 token spans
        for length in range(1, min(6, len(tokens) - start_idx + 1)):
            candidate = " ".join(tokens[start_idx:start_idx + length])

            # Exact match
            if candidate in entity_set:
                if length > best_length:
                    best_match = candidate
                    best_length = length

            # Case-insensitive match
            elif candidate.lower() in {e.lower() for e in entity_set}:
                if length > best_length:
                    best_match = candidate
                    best_length = length

        return best_match, best_length

    def annotate_sentence(self, text: str) -> Dict:
        """Auto-annotate a sentence with NER tags."""
        tokens = self._tokenize_sentence(text)
        ner_tags = ["O"] * len(tokens)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            matched = False

            # Try matching each entity type
            # Priority: TAXON > STRAT > CHRONO > LOC

            # TAXON
            match, length = self._match_multiword_entity(tokens, i, self.taxa)
            if match:
                ner_tags[i] = "B-TAXON"
                for j in range(1, length):
                    ner_tags[i + j] = "I-TAXON"
                i += length
                matched = True
                continue

            # STRAT
            match, length = self._match_multiword_entity(tokens, i, self.strat_units)
            if match:
                ner_tags[i] = "B-STRAT"
                for j in range(1, length):
                    ner_tags[i + j] = "I-STRAT"
                i += length
                matched = True
                continue

            # CHRONO
            match, length = self._match_multiword_entity(tokens, i, self.chrono_units)
            if match:
                ner_tags[i] = "B-CHRONO"
                for j in range(1, length):
                    ner_tags[i + j] = "I-CHRONO"
                i += length
                matched = True
                continue

            # LOC
            match, length = self._match_multiword_entity(tokens, i, self.localities)
            if match:
                ner_tags[i] = "B-LOC"
                for j in range(1, length):
                    ner_tags[i + j] = "I-LOC"
                i += length
                matched = True
                continue

            # Pattern-based matching for STRAT
            for pattern in self.strat_patterns:
                match = re.match(pattern, " ".join(tokens[i:i+5]))
                if match:
                    # Tag matched tokens
                    matched_text = match.group(0)
                    matched_tokens = self._tokenize_sentence(matched_text)
                    ner_tags[i] = "B-STRAT"
                    for j in range(1, len(matched_tokens)):
                        if i + j < len(tokens):
                            ner_tags[i + j] = "I-STRAT"
                    i += len(matched_tokens)
                    matched = True
                    break

            if matched:
                continue

            # Pattern-based matching for CHRONO
            for pattern in self.chrono_patterns:
                match = re.match(pattern, " ".join(tokens[i:i+5]))
                if match:
                    matched_text = match.group(0)
                    matched_tokens = self._tokenize_sentence(matched_text)
                    ner_tags[i] = "B-CHRONO"
                    for j in range(1, len(matched_tokens)):
                        if i + j < len(tokens):
                            ner_tags[i + j] = "I-CHRONO"
                    i += len(matched_tokens)
                    matched = True
                    break

            if not matched:
                i += 1

        return {
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags
        }

def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # Split on period followed by space and capital letter
    sentences = re.split(r'\.(\s+)(?=[A-Z])', text)

    # Reconstruct sentences
    result = []
    i = 0
    while i < len(sentences):
        if i + 2 < len(sentences):
            sent = sentences[i] + "." + sentences[i+1]
            result.append(sent)
            i += 2
        else:
            if sentences[i].strip():
                result.append(sentences[i])
            i += 1

    return [s.strip() for s in result if s.strip() and len(s.strip()) > 10]

def main():
    """Generate NER samples from corpus."""
    logger.info("="*80)
    logger.info("NER Auto-Annotation from Corpus")
    logger.info("="*80)

    # Paths
    vocab_dir = Path("artifacts/vocab")
    corpus_file = Path("data/corpus_norm/train_all_pdfs.jsonl")
    output_dir = Path("artifacts/ner_data")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize annotator
    annotator = NERAutoAnnotator(vocab_dir)

    # Load corpus
    logger.info(f"Loading corpus from {corpus_file}...")
    paragraphs = []
    with open(corpus_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                paragraphs.append(data)

    logger.info(f"  Loaded {len(paragraphs)} paragraphs")

    # Process paragraphs into sentences
    logger.info("Splitting into sentences and annotating...")
    all_samples = []
    entity_counts = defaultdict(int)

    for i, para in enumerate(paragraphs):
        if i % 100 == 0:
            logger.info(f"  Processing paragraph {i}/{len(paragraphs)}...")

        text = para.get("norm_text") or para.get("raw_text", "")
        if not text:
            continue

        sentences = split_into_sentences(text)

        for sent in sentences:
            if len(sent) < 20 or len(sent) > 500:  # Skip very short/long
                continue

            annotated = annotator.annotate_sentence(sent)

            # Count entities
            for tag in annotated["ner_tags"]:
                if tag.startswith("B-"):
                    entity_type = tag.split("-")[1]
                    entity_counts[entity_type] += 1

            # Add metadata
            annotated["metadata"] = {
                "doc_id": para.get("pub_id", "unknown"),
                "para_id": i
            }

            all_samples.append(annotated)

    logger.info(f"Generated {len(all_samples)} annotated sentences")
    logger.info("Entity distribution:")
    for entity_type, count in sorted(entity_counts.items()):
        logger.info(f"  {entity_type}: {count}")

    # Split into train/dev/test (80/10/10)
    import random
    random.seed(42)
    random.shuffle(all_samples)

    n_train = int(len(all_samples) * 0.8)
    n_dev = int(len(all_samples) * 0.1)

    train_samples = all_samples[:n_train]
    dev_samples = all_samples[n_train:n_train + n_dev]
    test_samples = all_samples[n_train + n_dev:]

    # Save splits
    logger.info("Saving splits...")

    def save_jsonl(samples, filename):
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    save_jsonl(train_samples, output_dir / "train.jsonl")
    save_jsonl(dev_samples, output_dir / "dev.jsonl")
    save_jsonl(test_samples, output_dir / "test.jsonl")

    logger.info(f"  Train: {len(train_samples)} samples → {output_dir / 'train.jsonl'}")
    logger.info(f"  Dev:   {len(dev_samples)} samples → {output_dir / 'dev.jsonl'}")
    logger.info(f"  Test:  {len(test_samples)} samples → {output_dir / 'test.jsonl'}")

    logger.info("="*80)
    logger.info("Auto-annotation complete!")
    logger.info("="*80)
    logger.info("")
    logger.info("IMPORTANT: These annotations are AUTO-GENERATED and need manual review.")
    logger.info("Expected accuracy: 60-70%")
    logger.info("Next step: Manual review and correction using annotation tool")
    logger.info("")

if __name__ == "__main__":
    main()
