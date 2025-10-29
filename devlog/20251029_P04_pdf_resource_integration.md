# P04: PDF Resource Integration - Corpus, Vocabulary, and Test Data

**Date:** 2025-10-29
**Status:** ✅ COMPLETE
**Author:** Claude Code
**Milestone:** P04 - PDF Resource Integration

---

## Executive Summary

This document describes the integration of the Cambrian correlation chart PDF (IUGS042-04-05.pdf) into the PaleoBERT training pipeline. The PDF provides a high-quality scientific resource containing domain-specific vocabulary, well-structured text for corpus augmentation, and ground-truth examples for test data generation.

**PDF Source:**
- **Title:** "A comprehensive Cambrian correlation chart"
- **Author:** Gerd Geyer (Universität Würzburg)
- **Publication:** Episodes Vol. 42, No. 4 (December 2019), pp. 321-374
- **DOI:** 10.18814/epiiugs/2019/019026
- **Size:** 5.0 MB
- **Pages:** 54 pages
- **Content:** Comprehensive biostratigraphic and chemostratigraphic correlation across all major Cambrian continents

**Three Integration Tasks:**

1. **Task 1: PDF to Corpus** - Extract text, normalize, and add to DAPT training corpus
2. **Task 2: Vocabulary Expansion** - Extract domain terms (taxa, formations, stages, localities) and expand tokenizer vocabulary
3. **Task 3: Test Data Generation** - Create annotated NER/RE examples for validation

**Expected Outcomes:**
- +50k-100k tokens added to DAPT corpus (high-quality domain text)
- +50-100 new vocabulary terms across four categories
- 30-50 hand-crafted test examples with gold-standard annotations

---

## Task 1: PDF Text Extraction and Corpus Integration

### Objective

Extract readable text from IUGS042-04-05.pdf, apply P02 normalization, and add to the DAPT training corpus in JSONL format.

### Implementation Steps

#### 1.1 PDF Text Extraction

**Tool:** PyMuPDF (fitz) or pdfplumber

**Strategy:**
- Extract text page-by-page
- Preserve paragraph boundaries
- Remove headers/footers (page numbers, journal info)
- Skip tables, figures, and captions (low-quality OCR text)
- Focus on main body text sections

**Code Template:**

```python
#!/usr/bin/env python
"""
Extract text from IUGS Cambrian correlation chart PDF.

Output: Raw text file for normalization pipeline.
"""
import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path: str, output_path: str):
    """Extract text from PDF, filter noise."""
    doc = fitz.open(pdf_path)

    extracted_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")

        # Remove page headers/footers
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            # Skip page numbers
            if re.match(r'^\d+$', line.strip()):
                continue
            # Skip journal headers
            if 'Episodes' in line or 'Vol. 42' in line:
                continue
            # Skip very short lines (likely noise)
            if len(line.strip()) < 10:
                continue

            clean_lines.append(line)

        page_text = '\n'.join(clean_lines)
        extracted_text.append(page_text)

    # Combine all pages
    full_text = '\n\n'.join(extracted_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Extracted {len(full_text)} characters from {len(doc)} pages")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    extract_text_from_pdf("IUGS042-04-05.pdf", "data/pdf_extracted/geyer2019_raw.txt")
```

**Expected Output:**
- `data/pdf_extracted/geyer2019_raw.txt` - ~200-300 KB raw text

#### 1.2 Text Normalization

**Apply P02 normalization pipeline** to extracted text:

1. Load raw text
2. Apply normalization transforms:
   - `Stage 10` → `Stage_10`
   - `Series 2` → `Series_2`
   - `Cambrian Epoch 2` → `Cambrian_Epoch_2`
   - Other multi-token units
3. Create alignment map (raw → norm)
4. Save as JSONL

**Code Template:**

```python
#!/usr/bin/env python
"""
Normalize PDF-extracted text and create JSONL corpus entries.
"""
import json
from typing import List, Tuple

def normalize_cambrian_units(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Apply normalization for Cambrian stratigraphic units."""
    # Define normalization patterns
    patterns = [
        (r'Stage (\d+)', r'Stage_\1'),
        (r'Series (\d+)', r'Series_\1'),
        (r'Cambrian Epoch (\d+)', r'Cambrian_Epoch_\1'),
        (r'Furongian Series', r'Furongian_Series'),
        (r'Miaolingian Series', r'Miaolingian_Series'),
        # Add more as needed
    ]

    norm_text = text
    align_map = []  # Character-level alignment

    # Simple implementation (full P02 pipeline has more sophisticated alignment)
    import re
    for pattern, replacement in patterns:
        norm_text = re.sub(pattern, replacement, norm_text)

    # For simplicity, create identity alignment
    # (Full P02 implementation tracks each character)
    align_map = list(enumerate(range(len(norm_text))))

    return norm_text, align_map

def create_jsonl_entries(raw_text_path: str, output_jsonl: str):
    """Create JSONL corpus entries from normalized text."""
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Normalize
    norm_text, align_map = normalize_cambrian_units(raw_text)

    # Split into paragraphs (simple approach)
    paragraphs = [p.strip() for p in norm_text.split('\n\n') if len(p.strip()) > 100]

    entries = []
    for idx, para in enumerate(paragraphs):
        entry = {
            "pub_id": "geyer2019",
            "cap_id": f"p{idx:04d}",
            "raw_text": para,  # Using norm_text as raw for simplicity
            "norm_text": para,
            "align_map": None,  # Not needed for DAPT
        }
        entries.append(entry)

    # Write JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(entries)} corpus entries")
    print(f"Saved to: {output_jsonl}")

if __name__ == "__main__":
    create_jsonl_entries(
        "data/pdf_extracted/geyer2019_raw.txt",
        "data/corpus_norm/train_geyer2019.jsonl"
    )
```

**Expected Output:**
- `data/corpus_norm/train_geyer2019.jsonl` - 100-200 entries, 50-100k tokens

#### 1.3 Validation

**Validation Checks:**

1. **Token Count:**
   ```bash
   wc -w data/pdf_extracted/geyer2019_raw.txt
   # Expected: 50,000-100,000 words
   ```

2. **JSONL Format Validation:**
   ```python
   import json
   count = 0
   with open("data/corpus_norm/train_geyer2019.jsonl", 'r') as f:
       for line in f:
           obj = json.loads(line)
           assert 'pub_id' in obj
           assert 'norm_text' in obj
           count += 1
   print(f"✓ {count} valid JSONL entries")
   ```

3. **Normalization Spot Check:**
   - Verify "Stage 10" → "Stage_10"
   - Verify "Series 2" → "Series_2"
   - Check alignment map correctness

**Success Criteria:**
- ✅ 50k-100k tokens extracted
- ✅ Valid JSONL format (100% parse success)
- ✅ Normalization applied correctly (spot check 10 examples)
- ✅ No duplicate paragraphs

---

## Task 2: Vocabulary Expansion from PDF

### Objective

Extract domain-specific terms (taxa, formations, stages, localities) mentioned in the PDF and add them to existing vocabulary files for tokenizer expansion.

### Implementation Steps

#### 2.1 Term Extraction

**Categories:**

1. **Taxa** (Trilobites, Brachiopods, etc.)
   - Pattern: Italicized genus/species names
   - Examples: *Olenellus*, *Paradoxides*, *Oryctocephalus*

2. **Stratigraphic Units**
   - Pattern: Formation, Member, Limestone, Shale
   - Examples: Wheeler Formation, Marjum Formation, Burgess Shale

3. **Chronostratigraphic Units**
   - Pattern: Stage names, Series names
   - Examples: Paibian, Jiangshanian, Guzhangian, Stage_10, Series_2

4. **Localities**
   - Pattern: Geographic names + descriptive context
   - Examples: House Range, Drum Mountains, Yoho National Park

**Extraction Strategy:**

```python
#!/usr/bin/env python
"""
Extract domain vocabulary from PDF text.
"""
import re
from collections import defaultdict
from typing import Dict, Set

def extract_taxa(text: str) -> Set[str]:
    """Extract taxonomic names (italicized in PDF, marked by pattern)."""
    # Note: PDF text extraction loses formatting, so use heuristics
    taxa = set()

    # Pattern 1: Capitalized Latin binomials
    pattern = r'\b([A-Z][a-z]+)\s+([a-z]+)\b'
    matches = re.findall(pattern, text)

    for genus, species in matches:
        # Filter out common English words
        if genus in {'The', 'This', 'These', 'Some', 'Many', 'Figure', 'Table'}:
            continue

        # Genus name
        taxa.add(genus)

        # Species (optional, can be too many)
        # taxa.add(f"{genus}_{species}")

    return taxa

def extract_formations(text: str) -> Set[str]:
    """Extract formation names."""
    formations = set()

    # Pattern: X Formation, X Member, X Limestone, etc.
    pattern = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(Formation|Member|Limestone|Shale|Sandstone|Group)'
    matches = re.findall(pattern, text)

    for name, unit_type in matches:
        full_name = f"{name}_{unit_type}"
        formations.add(full_name)

    return formations

def extract_chrono_units(text: str) -> Set[str]:
    """Extract chronostratigraphic units."""
    chrono_units = set()

    # Cambrian stages
    stages = ['Fortunian', 'Stage_2', 'Stage_3', 'Stage_4', 'Stage_5',
              'Wuliuan', 'Drumian', 'Guzhangian',
              'Paibian', 'Jiangshanian', 'Stage_10']

    # Series
    series = ['Terreneuvian', 'Series_2', 'Miaolingian', 'Furongian']

    for stage in stages + series:
        # Normalize underscores for search
        search_term = stage.replace('_', r'\s+')
        if re.search(search_term, text, re.IGNORECASE):
            chrono_units.add(stage)

    return chrono_units

def extract_localities(text: str) -> Set[str]:
    """Extract geographic localities."""
    localities = set()

    # Known Cambrian localities (could expand with NER)
    known_localities = [
        'House_Range', 'Drum_Mountains', 'Yoho_National_Park',
        'Avalon_Peninsula', 'White-Inyo_Mountains',
        'Siberian_Platform', 'Yangtze_Platform',
    ]

    for loc in known_localities:
        search_term = loc.replace('_', r'\s+')
        if re.search(search_term, text, re.IGNORECASE):
            localities.add(loc)

    return localities

def extract_all_terms(text_path: str) -> Dict[str, Set[str]]:
    """Extract all vocabulary terms from text."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    terms = {
        'taxa': extract_taxa(text),
        'strat_units': extract_formations(text),
        'chrono_units': extract_chrono_units(text),
        'localities': extract_localities(text),
    }

    return terms

if __name__ == "__main__":
    terms = extract_all_terms("data/pdf_extracted/geyer2019_raw.txt")

    for category, term_set in terms.items():
        print(f"\n{category}: {len(term_set)} terms")
        for term in sorted(term_set)[:10]:
            print(f"  - {term}")
```

**Expected Output:**
- Taxa: 50-100 genera
- Formations: 20-40 stratigraphic units
- Chrono units: 10-15 stages/series
- Localities: 10-20 geographic names

#### 2.2 Merge with Existing Vocabulary

**Strategy:**
1. Load existing vocab files from `artifacts/vocab/`
2. Add new terms (deduplicate)
3. Sort alphabetically
4. Save updated vocab files

**Code Template:**

```python
#!/usr/bin/env python
"""
Merge extracted terms with existing vocabulary files.
"""
import os
from typing import Set

def merge_vocabulary(category: str, new_terms: Set[str], vocab_dir: str = "artifacts/vocab"):
    """Merge new terms into existing vocabulary file."""
    vocab_file = os.path.join(vocab_dir, f"{category}.txt")

    # Load existing terms
    existing_terms = set()
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            existing_terms = set(line.strip() for line in f if line.strip())

    # Merge
    all_terms = existing_terms | new_terms

    # Sort and save
    sorted_terms = sorted(all_terms)

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            f.write(term + '\n')

    print(f"{category}: {len(existing_terms)} → {len(all_terms)} (+{len(new_terms - existing_terms)} new)")

if __name__ == "__main__":
    terms = extract_all_terms("data/pdf_extracted/geyer2019_raw.txt")

    for category, term_set in terms.items():
        merge_vocabulary(category, term_set)
```

**Expected Output:**
- `artifacts/vocab/taxa.txt` - Updated with +30-60 new genera
- `artifacts/vocab/strat_units.txt` - Updated with +10-30 new formations
- `artifacts/vocab/chrono_units.txt` - Updated with +5-10 new stages
- `artifacts/vocab/localities.txt` - Updated with +5-15 new localities

#### 2.3 Rebuild Tokenizer

**After vocabulary expansion, rebuild the tokenizer:**

```bash
# Rebuild tokenizer with expanded vocabulary
python scripts/build_tokenizer.py

# Validate fragmentation rate
python scripts/validate_tokenizer.py
```

**Expected Outcome:**
- New tokenizer at `artifacts/tokenizer_v1/` (or v2 if major change)
- Fragmentation rate should remain low (< 20%)

---

## Task 3: Test Data Generation from PDF

### Objective

Create hand-annotated NER and RE examples from PDF sentences for validation and test sets.

### Implementation Steps

#### 3.1 Sentence Selection

**Selection Criteria:**

1. **High Entity Density** - Sentences with 3+ entities
2. **Diverse Entity Types** - Mix of TAXON, STRAT, CHRONO, LOC
3. **Clear Relations** - Explicit relation mentions (occurs_in, found_at, assigned_to)
4. **Readable Context** - Well-formed sentences, not figure captions

**Strategy:**

```python
#!/usr/bin/env python
"""
Select candidate sentences from PDF for annotation.
"""
import re
import json
from typing import List, Dict

def select_candidate_sentences(text_path: str, output_path: str, top_n: int = 50):
    """Select sentences with high entity density."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)

    candidates = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 50 or len(sent) > 300:
            continue

        # Score by entity-like patterns
        score = 0

        # Taxon pattern (capitalized Latin)
        score += len(re.findall(r'\b[A-Z][a-z]+\s+[a-z]+\b', sent))

        # Formation pattern
        score += len(re.findall(r'\b[A-Z][a-z]+\s+Formation\b', sent))

        # Stage/Series pattern
        score += len(re.findall(r'\b(Stage|Series|Epoch)\s+\d+\b', sent))

        # Locality pattern (proper nouns)
        score += len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', sent))

        if score >= 3:
            candidates.append({'text': sent, 'score': score})

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Take top N
    top_candidates = candidates[:top_n]

    # Save as JSON for manual annotation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_candidates, f, indent=2)

    print(f"Selected {len(top_candidates)} candidate sentences")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    select_candidate_sentences(
        "data/pdf_extracted/geyer2019_raw.txt",
        "data/test_candidates.json",
        top_n=50
    )
```

**Expected Output:**
- `data/test_candidates.json` - 50 candidate sentences for manual review

#### 3.2 Manual Annotation

**Annotation Format (NER):**

```json
{
  "text": "The Wheeler Formation in the House Range yields abundant Asaphiscus wheeleri and Elrathia kingii from Stage 5.",
  "entities": [
    {"start": 4, "end": 21, "label": "STRAT", "text": "Wheeler_Formation"},
    {"start": 29, "end": 40, "label": "LOC", "text": "House_Range"},
    {"start": 56, "end": 75, "label": "TAXON", "text": "Asaphiscus_wheeleri"},
    {"start": 80, "end": 95, "label": "TAXON", "text": "Elrathia_kingii"},
    {"start": 101, "end": 108, "label": "CHRONO", "text": "Stage_5"}
  ]
}
```

**Annotation Format (RE):**

```json
{
  "text": "The Wheeler Formation in the House Range yields abundant Asaphiscus wheeleri and Elrathia kingii from Stage 5.",
  "entities": [
    {"id": "e1", "start": 4, "end": 21, "label": "STRAT", "text": "Wheeler_Formation"},
    {"id": "e2", "start": 29, "end": 40, "label": "LOC", "text": "House_Range"},
    {"id": "e3", "start": 56, "end": 75, "label": "TAXON", "text": "Asaphiscus_wheeleri"},
    {"id": "e4", "start": 80, "end": 95, "label": "TAXON", "text": "Elrathia_kingii"},
    {"id": "e5", "start": 101, "end": 108, "label": "CHRONO", "text": "Stage_5"}
  ],
  "relations": [
    {"head": "e3", "tail": "e1", "label": "occurs_in"},
    {"head": "e4", "tail": "e1", "label": "occurs_in"},
    {"head": "e3", "tail": "e2", "label": "found_at"},
    {"head": "e4", "tail": "e2", "label": "found_at"},
    {"head": "e1", "tail": "e5", "label": "assigned_to"}
  ]
}
```

**Annotation Tool:**

Manual annotation in JSON editor or use annotation tool like Label Studio, Doccano, or brat.

**Target:** 30-50 fully annotated examples

#### 3.3 Create Test Datasets

**Output Format:**

```
data/ner/test_geyer2019.jsonl  # NER test set
data/re/test_geyer2019.jsonl   # RE test set
```

**Code Template:**

```python
#!/usr/bin/env python
"""
Convert annotated examples to NER/RE test datasets.
"""
import json

def create_ner_dataset(annotations_path: str, output_path: str):
    """Create NER test dataset in JSONL format."""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    with open(output_path, 'w') as f:
        for ann in annotations:
            # Convert to NER format (BIO tags or span format)
            ner_entry = {
                'text': ann['text'],
                'entities': ann['entities'],
            }
            f.write(json.dumps(ner_entry) + '\n')

    print(f"Created NER test dataset: {len(annotations)} examples")

def create_re_dataset(annotations_path: str, output_path: str):
    """Create RE test dataset in JSONL format."""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    with open(output_path, 'w') as f:
        for ann in annotations:
            # Convert to RE format
            re_entry = {
                'text': ann['text'],
                'entities': ann['entities'],
                'relations': ann.get('relations', []),
            }
            f.write(json.dumps(re_entry) + '\n')

    print(f"Created RE test dataset: {len(annotations)} examples")

if __name__ == "__main__":
    # Assuming manual annotations saved to this file
    annotations_file = "data/annotations/geyer2019_annotated.json"

    create_ner_dataset(annotations_file, "data/ner/test_geyer2019.jsonl")
    create_re_dataset(annotations_file, "data/re/test_geyer2019.jsonl")
```

**Expected Output:**
- `data/ner/test_geyer2019.jsonl` - 30-50 NER test examples
- `data/re/test_geyer2019.jsonl` - 30-50 RE test examples

#### 3.4 Validation

**Quality Checks:**

1. **Entity Coverage:**
   - All 4 entity types represented (TAXON, STRAT, CHRONO, LOC)
   - Minimum 5 examples per entity type

2. **Relation Coverage:**
   - All 4 relation types represented
   - Minimum 3 examples per relation type

3. **Span Accuracy:**
   - Character offsets match text exactly
   - No overlapping entities
   - Normalized form used (e.g., "Stage_5" not "Stage 5")

4. **Format Validation:**
   ```python
   # Validate JSONL format
   with open("data/ner/test_geyer2019.jsonl", 'r') as f:
       for line in f:
           obj = json.loads(line)
           assert 'text' in obj
           assert 'entities' in obj
           for ent in obj['entities']:
               assert 'start' in ent
               assert 'end' in ent
               assert 'label' in ent
               # Validate span
               assert obj['text'][ent['start']:ent['end']] == ent['text']
   ```

**Success Criteria:**
- ✅ 30-50 annotated examples
- ✅ All entity types represented (≥5 each)
- ✅ All relation types represented (≥3 each)
- ✅ 100% span validation pass rate
- ✅ Valid JSONL format

---

## Dependencies and Requirements

### Python Packages

```bash
# PDF processing
pip install PyMuPDF  # fitz
# OR
pip install pdfplumber

# Already installed (from requirements.txt)
# - transformers
# - tokenizers
# - datasets
```

### File Structure

```
PaleoBERT/
├── IUGS042-04-05.pdf                    # Source PDF
├── data/
│   ├── pdf_extracted/
│   │   └── geyer2019_raw.txt            # Task 1 output
│   ├── corpus_norm/
│   │   └── train_geyer2019.jsonl        # Task 1 output
│   ├── test_candidates.json             # Task 3 intermediate
│   ├── annotations/
│   │   └── geyer2019_annotated.json     # Task 3 manual work
│   ├── ner/
│   │   └── test_geyer2019.jsonl         # Task 3 output
│   └── re/
│       └── test_geyer2019.jsonl         # Task 3 output
├── artifacts/
│   └── vocab/
│       ├── taxa.txt                     # Task 2 updated
│       ├── strat_units.txt              # Task 2 updated
│       ├── chrono_units.txt             # Task 2 updated
│       └── localities.txt               # Task 2 updated
└── scripts/
    ├── p04_extract_pdf_text.py          # Task 1 script
    ├── p04_extract_vocabulary.py        # Task 2 script
    └── p04_generate_test_data.py        # Task 3 script
```

---

## Implementation Timeline

### Phase 1: PDF Text Extraction (30-60 min)
- [ ] Install PDF processing library (PyMuPDF)
- [ ] Write extraction script (`p04_extract_pdf_text.py`)
- [ ] Extract text from PDF → `geyer2019_raw.txt`
- [ ] Normalize text and create JSONL
- [ ] Validate token count and format

### Phase 2: Vocabulary Expansion (30-45 min)
- [ ] Write term extraction script (`p04_extract_vocabulary.py`)
- [ ] Extract taxa, formations, stages, localities
- [ ] Merge with existing vocab files
- [ ] Rebuild tokenizer (if needed)
- [ ] Validate fragmentation rate

### Phase 3: Test Data Generation (60-90 min)
- [ ] Write sentence selection script (`p04_generate_test_data.py`)
- [ ] Select top 50 candidate sentences
- [ ] Manual annotation (30-50 examples)
- [ ] Create NER/RE test datasets
- [ ] Validate format and coverage

**Total Estimated Time:** 2-3 hours (mostly manual annotation)

---

## Validation Criteria

### Task 1: PDF to Corpus

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Text extraction quality | Clean paragraphs, no tables/captions | Manual spot check (10 samples) |
| Token count | 50k-100k tokens | `wc -w geyer2019_raw.txt` |
| JSONL format validity | 100% parse success | `json.loads()` test |
| Normalization correctness | "Stage 10" → "Stage_10" | Regex check on norm_text |
| No duplicates | 0 duplicate paragraphs | Hash comparison |

### Task 2: Vocabulary Expansion

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Taxa extracted | 30-60 new genera | Count in taxa.txt |
| Formations extracted | 10-30 new units | Count in strat_units.txt |
| Chrono units extracted | 5-10 new stages | Count in chrono_units.txt |
| Localities extracted | 5-15 new names | Count in localities.txt |
| Fragmentation rate | < 20% | `validate_tokenizer.py` |
| No duplicates | 0 duplicates in vocab files | Sort + uniq check |

### Task 3: Test Data Generation

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Annotated examples | 30-50 examples | Count lines in JSONL |
| Entity type coverage | All 4 types, ≥5 each | Count by label |
| Relation type coverage | All 4 types, ≥3 each | Count by relation |
| Span accuracy | 100% match | Extract and compare spans |
| JSONL format validity | 100% parse success | `json.loads()` test |

---

## Known Limitations

### Task 1: PDF Text Extraction

1. **OCR Quality:**
   - PDF is text-based (not scanned), so OCR quality is good
   - However, some formatting artifacts may remain (e.g., column breaks)

2. **Table/Figure Handling:**
   - Tables and figures are skipped (complex layout, low information density)
   - May miss some entity mentions in figure captions

3. **Paragraph Segmentation:**
   - Simple newline-based splitting may not perfectly preserve semantic paragraphs
   - Some long paragraphs may be split; some short paragraphs may be merged

### Task 2: Vocabulary Expansion

1. **Term Extraction Accuracy:**
   - Heuristic-based extraction may have false positives (e.g., common words matching capitalization pattern)
   - Manual review recommended before merging

2. **Taxonomic Completeness:**
   - Only genus-level names extracted
   - Species-level names omitted (too many, too specific)

3. **Locality Coverage:**
   - Known localities matched by predefined list
   - Novel localities may be missed without NER

### Task 3: Test Data Generation

1. **Manual Annotation Bottleneck:**
   - Requires human annotation time (1-2 hours)
   - Quality depends on annotator expertise

2. **Coverage Limitations:**
   - Only 30-50 examples may not cover all edge cases
   - Focused on positive examples; negative examples need separate generation

3. **Relation Ambiguity:**
   - Some relations may be implicit or require inference
   - Annotator judgment required for borderline cases

---

## Success Metrics

### Overall P04 Success Criteria

| Task | Success Metric | Target |
|------|---------------|--------|
| Task 1 | Corpus size increase | +50k-100k tokens |
| Task 1 | JSONL validity | 100% |
| Task 2 | Vocabulary expansion | +50-100 terms |
| Task 2 | Fragmentation rate | < 20% |
| Task 3 | Test examples created | 30-50 |
| Task 3 | Entity/relation coverage | All types ≥3 examples |

### Downstream Impact

**Expected improvements after P04 integration:**

1. **DAPT Training:**
   - Corpus diversity increased → lower domain-specific perplexity
   - Better coverage of Cambrian terminology

2. **Tokenizer Quality:**
   - Lower fragmentation rate for common terms
   - Improved encoding efficiency

3. **NER/RE Evaluation:**
   - Gold-standard test set for validation
   - Baseline performance metrics established

---

## Troubleshooting

### Issue: PDF Text Extraction Fails

**Error:** `ImportError: No module named 'fitz'`

**Solution:**
```bash
pip install PyMuPDF
```

**Alternative:** Use pdfplumber
```bash
pip install pdfplumber
```

### Issue: Vocabulary Extraction Has Many False Positives

**Example:** Extracting "The" and "This" as taxa

**Solution:** Expand stopword list in extraction script:
```python
stopwords = {'The', 'This', 'These', 'Some', 'Many', 'Figure', 'Table',
             'However', 'Therefore', 'Although', 'Such', 'Other'}
```

### Issue: JSONL Format Invalid

**Error:** `json.decoder.JSONDecodeError`

**Solution:** Validate JSON before writing:
```python
import json
# Before writing
json_str = json.dumps(entry)  # Validates serialization
f.write(json_str + '\n')
```

### Issue: Tokenizer Fragmentation Rate Increased

**Example:** After adding vocabulary, fragmentation rate went from 15% to 35%

**Solution:**
1. Review added vocabulary for quality
2. Remove very long or unusual terms
3. Consider using multi-word tokens (e.g., "Marjum_Formation" as single token)

---

## Next Steps After P04

### Immediate Next Steps

1. **Run P03 DAPT Training with Updated Corpus:**
   ```bash
   python scripts/train_dapt.py --config config/dapt_config.yaml
   ```

2. **Evaluate DAPT Performance:**
   - Compare rare-token perplexity before/after PDF corpus integration
   - Check if fragmentation rate improved

3. **Baseline NER/RE Evaluation:**
   - Test pre-DAPT model on geyer2019 test set
   - Establish baseline F1 scores

### Future Milestones

- **P05:** NER Training Script (use geyer2019 test set for validation)
- **P06:** RE Training Script
- **P07:** End-to-End Inference Pipeline
- **P08:** Performance Benchmarking

---

## References

### PDF Source

- **Geyer, G.** (2019). A comprehensive Cambrian correlation chart. *Episodes*, 42(4), 321-374. DOI: 10.18814/epiiugs/2019/019026

### Related Devlog Documents

- **P01:** `devlog/20251029_P01_tokenizer_build.md` - Tokenizer build with domain vocabulary
- **P02:** `devlog/20251029_P02_normalization.md` - Text normalization pipeline
- **P03:** `devlog/20251029_P03_dapt_training_script.md` - DAPT training script
- **P03 Phase 2:** `devlog/20251029_004_P03_Phase2_validation_metrics_complete.md` - DAPT validation metrics

### External Documentation

- **PyMuPDF Documentation:** https://pymupdf.readthedocs.io/
- **HuggingFace Datasets:** https://huggingface.co/docs/datasets/
- **JSONL Format:** http://jsonlines.org/

---

## Appendix: Sample Outputs

### A. Sample JSONL Entry

```json
{
  "pub_id": "geyer2019",
  "cap_id": "p0042",
  "raw_text": "The Wheeler Formation yields diverse trilobite assemblages including Asaphiscus wheeleri, Elrathia kingii, and Bolaspidella housensis. These taxa are characteristic of Cambrian Stage 5 in the Laurentian margin.",
  "norm_text": "The Wheeler_Formation yields diverse trilobite assemblages including Asaphiscus wheeleri, Elrathia kingii, and Bolaspidella housensis. These taxa are characteristic of Cambrian_Stage_5 in the Laurentian margin.",
  "align_map": null
}
```

### B. Sample NER Test Entry

```json
{
  "text": "The Wheeler_Formation yields diverse trilobite assemblages including Asaphiscus wheeleri, Elrathia kingii, and Bolaspidella housensis.",
  "entities": [
    {"start": 4, "end": 21, "label": "STRAT", "text": "Wheeler_Formation"},
    {"start": 69, "end": 88, "label": "TAXON", "text": "Asaphiscus wheeleri"},
    {"start": 90, "end": 105, "label": "TAXON", "text": "Elrathia kingii"},
    {"start": 111, "end": 132, "label": "TAXON", "text": "Bolaspidella housensis"}
  ]
}
```

### C. Sample RE Test Entry

```json
{
  "text": "The Wheeler_Formation yields diverse trilobite assemblages including Asaphiscus wheeleri from Cambrian_Stage_5.",
  "entities": [
    {"id": "e1", "start": 4, "end": 21, "label": "STRAT", "text": "Wheeler_Formation"},
    {"id": "e2", "start": 69, "end": 88, "label": "TAXON", "text": "Asaphiscus wheeleri"},
    {"id": "e3", "start": 94, "end": 111, "label": "CHRONO", "text": "Cambrian_Stage_5"}
  ],
  "relations": [
    {"head": "e2", "tail": "e1", "label": "occurs_in"},
    {"head": "e1", "tail": "e3", "label": "assigned_to"}
  ]
}
```

---

**END OF P04 IMPLEMENTATION PLAN**
