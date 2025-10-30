# P07: Relation Extraction (RE) Preparation

**Date:** 2025-10-30
**Status:** 🔄 In Progress
**Type:** Plan
**Dependencies:** NER model (M2), DAPT model (P06)

---

## Overview

**Goal:** Prepare all components for Relation Extraction (RE) training - schema design, data generation, training script, and configuration.

**Context:** NER training is currently running on a separate machine. RE preparation proceeds in parallel to maintain project momentum.

---

## Objectives

1. Design comprehensive RE annotation schema
2. Implement auto-annotation pipeline from NER data
3. Create RE training script with entity marker approach
4. Configure training for 11GB GPU constraint
5. Generate 60K-160K training pairs

---

## Target Relations

### Relation Types

| Relation | Signature | Description |
|----------|-----------|-------------|
| **occurs_in** | TAXON → STRAT | Taxon found in stratigraphic unit |
| **found_at** | TAXON → LOC | Taxon discovered at locality |
| **assigned_to** | STRAT → CHRONO | Strat unit correlated to time interval |
| **part_of** | STRAT → STRAT | Strat unit is subdivision of another |
| **NO_RELATION** | ANY → ANY | No semantic relationship |

**Total Classes:** 5 (4 positive + 1 negative)

### Examples

**occurs_in:**
```
Text: "Olenellus gilberti occurs in the Wheeler Formation"
Triple: (Olenellus_gilberti, occurs_in, Wheeler_Formation)
```

**found_at:**
```
Text: "Olenellus wheeleri was found in the House Range, Utah"
Triple: (Olenellus_wheeleri, found_at, House_Range)
```

**assigned_to:**
```
Text: "The Wheeler Formation is assigned to Cambrian Stage 5"
Triple: (Wheeler_Formation, assigned_to, Stage_5)
```

**part_of:**
```
Text: "Wheeler Member of the Marjum Formation"
Triple: (Wheeler_Member, part_of, Marjum_Formation)
```

---

## Architecture

### Model Approach: Entity Marker Classification

**Input Format:**
```
"[SUBJ] Olenellus gilberti [/SUBJ] occurs in [OBJ] Wheeler Formation [/OBJ]"
```

**Pipeline:**
```
Marked Text
    ↓
Tokenizer (with special tokens: [SUBJ], [/SUBJ], [OBJ], [/OBJ])
    ↓
DeBERTa Encoder (from DAPT checkpoint)
    ↓
[CLS] representation
    ↓
Classification Head (5 classes)
    ↓
Relation Label (0-4)
```

**Why Entity Markers?**
- Simple and proven effective
- Clear subject/object identification
- Works well with pre-trained transformers
- No complex positional embeddings needed

---

## Data Generation Strategy

### Step 1: Entity Pair Extraction

**From NER Annotations:**
- Source: 73K NER annotated sentences
- Sentences with entities: 24K (32.7%)
- Average entities per sentence: 2.4
- Average valid pairs per sentence: 3-5

**Valid Pair Types:**

| Subject | Object | Valid Relations |
|---------|--------|-----------------|
| TAXON | STRAT | occurs_in |
| TAXON | LOC | found_at |
| STRAT | CHRONO | assigned_to |
| STRAT | STRAT | part_of |
| Other | Other | NO_RELATION only |

**Expected Candidates:** 40K-74K pairs

### Step 2: Pattern-Based Relation Assignment

**occurs_in patterns:**
```regex
{TAXON} (?:occurs?|found|collected) (?:in|from) (?:the )?{STRAT}
{STRAT} yields? {TAXON}
{TAXON} (?:of|from) (?:the )?{STRAT}
```

**found_at patterns:**
```regex
{TAXON} (?:from|at|in) {LOC}
{LOC} yields? {TAXON}
```

**assigned_to patterns:**
```regex
{STRAT} (?:assigned|correlated|dated) (?:to|with) {CHRONO}
{CHRONO} {STRAT}
```

**part_of patterns:**
```regex
{STRAT1} (?:Member|Bed) (?:of|within) (?:the )?{STRAT2}
{STRAT1} (?:part of|within) (?:the )?{STRAT2}
```

**Expected Accuracy:** 50-70% (auto-annotation)

### Step 3: Negative Sampling

**NO_RELATION Assignment:**
- Valid type pairs without pattern match
- Entities >15 tokens apart
- Random sampling to achieve 1:2 to 1:4 ratio

**Target Distribution:**
- Positive relations: 20K-40K
- Negative relations: 40K-120K
- Total: 60K-160K pairs

### Step 4: Data Splits

- **Train:** 80% (stratified by relation)
- **Dev:** 10%
- **Test:** 10%

---

## Implementation Plan

### Task 1: RE Annotation Schema

**File:** `docs/re_annotation_schema.md`

**Contents:**
- Relation definitions with examples
- Valid entity type pairs
- Annotation guidelines
- Edge cases and ambiguity resolution
- Data format specification
- Evaluation metrics

**Status:** ⏳ To be implemented

### Task 2: Data Generation Script

**File:** `scripts/generate_re_samples.py`

**Functionality:**
```python
# High-level flow
1. Load NER annotated sentences
2. Extract entity pairs (filter by valid types)
3. Apply relation patterns
4. Assign NO_RELATION to unmatched pairs
5. Negative sampling (balance dataset)
6. Split into train/dev/test
7. Save as JSONL with entity markers
```

**Output Format:**
```json
{
  "text": "Original sentence",
  "marked_text": "[SUBJ] entity1 [/SUBJ] ... [OBJ] entity2 [/OBJ]",
  "subject": {"type": "TAXON", "text": "Olenellus"},
  "object": {"type": "STRAT", "text": "Wheeler Formation"},
  "relation": "occurs_in",
  "label_id": 1,
  "metadata": {"doc_id": "...", "sent_id": "..."}
}
```

**Status:** ⏳ To be implemented

### Task 3: RE Training Script

**File:** `scripts/train_re.py`

**Key Features:**
1. Load JSONL → HuggingFace Dataset
2. Add special tokens to tokenizer
3. Process marked text
4. Load DAPT model + classification head
5. Class-weighted cross-entropy loss
6. Per-relation F1 evaluation
7. Best model selection

**Special Tokens:**
```python
special_tokens = ["[SUBJ]", "[/SUBJ]", "[OBJ]", "[/OBJ]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
```

**Evaluation Metrics:**
- Per-relation: Precision, Recall, F1
- Overall: Micro-F1, Macro-F1, Accuracy
- Confusion matrix

**Status:** ⏳ To be implemented

### Task 4: RE Configuration

**File:** `config/re_config.yaml`

**Key Settings:**
```yaml
# Model
model_name_or_path: "checkpoints/paleo-dapt-expanded"
tokenizer_path: "artifacts/tokenizer_v1"

# Data
train_file: "artifacts/re_data/train.jsonl"
dev_file: "artifacts/re_data/dev.jsonl"
test_file: "artifacts/re_data/test.jsonl"
max_seq_length: 384

# Labels
num_labels: 5
label_names: ["NO_RELATION", "occurs_in", "found_at", "assigned_to", "part_of"]

# Training (11GB VRAM)
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 1.0e-5  # Lower than NER
num_train_epochs: 8
fp16: true

# Class weighting (handle imbalance)
class_weights: [0.5, 2.0, 2.0, 2.0, 2.0]  # Downweight NO_RELATION
```

**Status:** ⏳ To be implemented

---

## Success Criteria

### Target Metrics (from CLAUDE.md)

**Overall:**
- Micro-F1 ≥ 0.75
- Macro-F1 ≥ 0.70
- Accuracy ≥ 0.85

**Per-Relation:**
- F1(occurs_in) ≥ 0.80
- F1(found_at) ≥ 0.70
- F1(assigned_to) ≥ 0.70
- F1(part_of) ≥ 0.70

**Quality:**
- NO_RELATION precision ≥ 0.80
- Per-relation recall ≥ 0.65

---

## Timeline

### Phase 1: Preparation (Current)
- Schema design: 2-3 hours
- Data generation script: 3-4 hours
- Training script: 4-6 hours
- Configuration: 2-3 hours
- **Total: 11-16 hours (~2 days)**

### Phase 2: Execution (After DAPT available)
- Generate data: 30-60 minutes
- Train model: 2-3 hours (8 epochs)
- Evaluate: 2-3 hours
- **Total: 5-7 hours (~1 day)**

**Overall P07:** 3-5 days

---

## Known Challenges

### 1. Class Imbalance
**Issue:** NO_RELATION can be 2-4× more frequent
**Mitigation:** Class weights, careful negative sampling, monitor per-class metrics

### 2. Pattern Coverage
**Issue:** Rule-based patterns miss paraphrases (50-70% recall expected)
**Mitigation:** Neural model learns from patterns, future manual annotation

### 3. Long-Distance Dependencies
**Issue:** Subject and object may be 10+ tokens apart
**Mitigation:** Transformer attention handles long dependencies

### 4. NER Error Propagation
**Issue:** NER errors affect RE (wrong entity types → wrong relations)
**Mitigation:** Improve NER first, RE may learn some type disambiguation

---

## Dependencies

### Required Before Training:
- ⏸️ DAPT model checkpoint (`checkpoints/paleo-dapt-expanded/`)
- ⏸️ NER annotations (already available: 73K sentences)

### Optional:
- NER model for inference on new text (blocked until NER training completes)

### Enables:
- M4: End-to-End Pipeline
- Knowledge graph construction
- Information extraction applications

---

## Deliverables

### Documentation
- `docs/re_annotation_schema.md` - Annotation guidelines

### Code
- `scripts/generate_re_samples.py` - Data generation
- `scripts/train_re.py` - Training script
- `config/re_config.yaml` - Configuration

### Data
- `artifacts/re_data/train.jsonl` - Training pairs
- `artifacts/re_data/dev.jsonl` - Dev pairs
- `artifacts/re_data/test.jsonl` - Test pairs

### Model
- `checkpoints/paleo-re-v1/` - Trained RE model

---

## Next Actions

### Immediate
1. ⏳ Write RE annotation schema document
2. ⏳ Implement `generate_re_samples.py`
3. ⏳ Implement `train_re.py`
4. ⏳ Create `re_config.yaml`

### After Schema Complete
1. Generate RE training data
2. Validate data statistics
3. Test training script with small sample

### After DAPT Available
1. Run full RE training
2. Evaluate on test set
3. Analyze per-relation performance
4. Document results in completion devlog

---

## Notes

- **Parallel development:** NER training on separate machine (M2 Phase 2)
- **Resource requirements:** Same as NER (RTX 2080 Ti, 11GB VRAM)
- **Design choice:** Entity markers over positional embeddings (simpler, proven)
- **Classification strategy:** 5-way classification (more efficient than binary per relation)

---

**Status:** In Progress
**Next Milestone:** M3 Phase 2 (RE Training)
**Last Updated:** 2025-10-30
