# P08: Gold Standard Dataset Creation

**Date:** 2025-10-31
**Status:** 📋 Planning
**Type:** Preparation
**Dependencies:** M2 (NER Complete), M3 (RE Partial)

---

## Overview

**Goal:** Create high-quality, manually annotated gold standard datasets for rigorous evaluation of NER and RE models, providing unbiased performance metrics independent of auto-annotation quality.

**Context:** Current models trained on auto-annotated data (60-70% expected quality) show exceptional performance (NER 99.18% F1, RE 82.95% Micro F1). Gold standard evaluation is essential to validate these metrics on truly clean data and measure real-world performance.

---

## Objectives

1. Define gold standard dataset requirements and scope
2. Design annotation guidelines and quality control procedures
3. Select optimal sampling strategy for representative coverage
4. Set up annotation tools and infrastructure
5. Plan annotation workflow and timeline
6. Establish inter-annotator agreement (IAA) protocols

---

## What is a Gold Standard Dataset?

### Definition

A **gold standard** (also called **reference standard** or **ground truth**) is a manually verified, high-quality dataset used as the authoritative benchmark for model evaluation.

| Aspect | Auto-Annotation | Gold Standard |
|--------|-----------------|---------------|
| **Creation Method** | Pattern/rule-based automatic | **Manual expert review** |
| **Accuracy** | 50-70% (estimated) | **95-100%** |
| **Speed** | Very fast (73K in 8 min) | Slow (100 sentences in 4-8 hours) |
| **Purpose** | **Training data** | **Evaluation data** |
| **Cost** | Low ($0) | High ($300-1000+) |
| **Bias** | Pattern-dependent | Minimal (if multi-annotator) |

**Key Principle:** Train on auto-annotation, evaluate on gold standard.

---

## Why Gold Standard is Needed

### Problem: Auto-Annotation Circularity

```
Auto-Annotation → Training Data
      ↓
  Model Training
      ↓
Evaluation on Auto-Annotated Test Set
      ↓
High F1 (99.18%) ← But is it real or circular?
```

**Issue:** Test set has same biases as training set (same pattern-based generation).

### Solution: Independent Gold Standard

```
Manual Annotation → Gold Standard Test
      ↓
Evaluation on Independent Data
      ↓
True F1 (90-95%?) ← Real performance estimate
```

**Value:**
- Unbiased performance measurement
- Detection of auto-annotation artifacts
- Generalization assessment
- Confidence in production deployment

---

## Gold Standard Scope

### For PaleoBERT

#### NER Gold Standard
- **Target Size:** 300-500 sentences
- **Entity Coverage:** 75-100 examples per type (TAXON, STRAT, CHRONO, LOC)
- **Purpose:** Validate 99.18% F1 claim, detect overfitting

#### RE Gold Standard
- **Target Size:** 200-300 relation pairs
- **Relation Coverage:** 50+ examples per relation (occurs_in, found_at, assigned_to, part_of)
- **Purpose:** Validate RE model, especially rare relations

---

## Sampling Strategy

### 1. Stratified Sampling (Recommended ✅)

**Principle:** Ensure proportional representation of all entity/relation types.

**NER Example:**
```python
gold_samples = {
    'TAXON': 100,    # 25% of 400 sentences
    'STRAT': 100,    # 25%
    'CHRONO': 100,   # 25%
    'LOC': 100,      # 25%
}
```

**RE Example:**
```python
gold_samples = {
    'occurs_in': 60,      # Most common
    'found_at': 60,
    'assigned_to': 40,    # Rare but critical
    'part_of': 40,
    'NO_RELATION': 100,   # Negative samples
}
```

### 2. Random Sampling

**Approach:**
```python
import random
test_data = load_jsonl('artifacts/ner_data/test.jsonl')
gold_standard = random.sample(test_data, 300)
```

**Pros:** Simple, unbiased distribution
**Cons:** May under-sample rare types

### 3. Hard Example Sampling

**Approach:** Focus on difficult cases
```python
# Model predictions with low confidence
hard_examples = [ex for ex in test_data
                 if model_confidence(ex) < 0.80]
```

**Pros:** Identifies model weaknesses
**Cons:** Not representative of overall performance

### 4. Diverse Sampling (Recommended for PaleoBERT ✅)

**Multi-source strategy:**
```python
sources = {
    'test_set': 150,           # Current auto-annotated test
    'train_set': 50,            # Check training data quality
    'new_papers_1950s': 25,    # Older papers (OCR challenges)
    'new_papers_2020s': 25,    # Recent papers (new terminology)
    'edge_cases': 50,           # Ambiguous examples
}
```

**Rationale:**
- Test set: Validate current metrics
- Train set: Detect training data issues
- New papers: Assess generalization
- Edge cases: Improve guidelines

---

## Annotation Guidelines

### Document Structure

**File:** `docs/gold_standard_annotation_guide.md`

**Sections:**
1. Annotation principles
2. Entity/relation definitions
3. Edge case resolution rules
4. Examples (positive and negative)
5. Quality checklist

### NER Annotation Rules

#### General Principles

1. **Preserve original text** (no normalization during annotation)
2. **Maximum span** (include full binomial names, not just genus)
3. **No nested entities** (choose most specific type)
4. **Consistent boundaries** (include/exclude determiners consistently)

#### Entity-Specific Rules

**TAXON:**
```markdown
✅ Include:
- "Olenellus gilberti" (full binomial)
- "Olenellus sp." (genus + uncertain species)
- "trilobites" (taxonomic group names)

❌ Exclude:
- "fauna", "species" (generic terms without taxonomic meaning)
- "the Olenellus" → tag "Olenellus" only (exclude determiner)

Edge Cases:
- "Olenellus and Paradoxides" → Tag each separately
- "Middle Cambrian trilobites" → Tag "trilobites" only (CHRONO + TAXON)
```

**STRAT:**
```markdown
✅ Include:
- "Wheeler Formation"
- "Wheeler Fm." (abbreviation)
- "Marjum Formation"

❌ Exclude:
- "formation" alone (generic geological term)
- "the Wheeler Formation" → tag "Wheeler Formation" only

Edge Cases:
- "Middle Wheeler Formation" → Tag "Wheeler Formation" (temporal modifier not part of name)
- "Wheeler Member of Marjum Formation" → Tag both separately (part_of relation)
```

**CHRONO:**
```markdown
✅ Include:
- "Cambrian Stage 5"
- "Jiangshanian"
- "Series 2" or "Series_2" (both forms)
- "Middle Cambrian"

❌ Exclude:
- "Cambrian" when used adjectivally ("Cambrian rocks" → no tag)

Edge Cases:
- "uppermost Lower Cambrian" → Tag "Lower Cambrian" (modifiers outside)
```

**LOC:**
```markdown
✅ Include:
- "House Range"
- "Utah"
- "Wheeler Amphitheater" (geological locality)

❌ Exclude:
- "western" in "western Utah" → Tag "Utah" only
- "North America" (too broad, case-by-case decision)

Edge Cases:
- Fossil locality vs modern place: Include both
- "Wheeler" (ambiguous) → Need context (Formation vs locality)
```

#### Ambiguity Resolution

**Rule 1: Nested Entities**
```
Text: "Middle Cambrian Wheeler Formation"
Annotation:
  - [B-CHRONO] Middle
  - [I-CHRONO] Cambrian
  - [B-STRAT] Wheeler
  - [I-STRAT] Formation
```

**Rule 2: Coordination**
```
Text: "Olenellus and Paradoxides"
Annotation:
  - [B-TAXON] Olenellus
  - [O] and
  - [B-TAXON] Paradoxides
```

**Rule 3: When Uncertain**
```
If unclear → Use [O] tag
(Conservative approach prevents false positives)
```

### RE Annotation Rules

#### Relation Definitions

**occurs_in (TAXON → STRAT):**
```markdown
Definition: Taxon is found within stratigraphic unit

Positive Examples:
- "Olenellus gilberti occurs in the Wheeler Formation"
- "The Wheeler Formation yields diverse trilobites"
- "trilobites from the Marjum Formation"

Negative Examples (NOT occurs_in):
- "Olenellus and Wheeler Formation" (mere co-occurrence)
- "studies of Olenellus in the Wheeler Formation" (research context)
```

**found_at (TAXON → LOC):**
```markdown
Definition: Taxon discovered at geographic locality

Positive Examples:
- "Olenellus from the House Range, Utah"
- "specimens collected in Wheeler Amphitheater"

Negative Examples:
- "Olenellus, widespread in western USA" (too vague)
```

**assigned_to (STRAT → CHRONO):**
```markdown
Definition: Stratigraphic unit correlated to chronostratigraphic interval

Positive Examples:
- "Wheeler Formation is assigned to Cambrian Stage 5"
- "Marjum Formation, Drumian"

Negative Examples:
- "Cambrian Wheeler Formation" (adjective use)
```

**part_of (STRAT → STRAT):**
```markdown
Definition: Stratigraphic unit is subdivision of another

Positive Examples:
- "Wheeler Member of the Marjum Formation"
- "the Spence Shale within the Langston Formation"

Negative Examples:
- "Wheeler and Marjum Formations" (siblings, not part-of)
```

---

## Annotation Tools

### Option 1: Label Studio (Recommended ✅)

**Installation:**
```bash
pip install label-studio
label-studio start
```

**Access:** http://localhost:8080

**Features:**
- Web-based UI
- BIO tagging support
- Multi-annotator collaboration
- Export to JSONL
- Built-in IAA calculation

**Setup for NER:**
```xml
<View>
  <Labels name="ner" toName="text">
    <Label value="TAXON" background="red"/>
    <Label value="STRAT" background="blue"/>
    <Label value="CHRONO" background="green"/>
    <Label value="LOC" background="orange"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

### Option 2: Prodigy

**Cost:** $390/annotator (commercial)

**Pros:** Active learning, production-ready
**Cons:** Expensive for small projects

### Option 3: Custom Script

**For simple projects:**
```python
# scripts/manual_annotation_tool.py

def annotate_sentence(text, tokens):
    """Simple terminal-based annotation."""
    print(f"\nText: {text}")
    print("Tokens:", " ".join(f"[{i}]{t}" for i, t in enumerate(tokens)))

    tags = []
    for i, token in enumerate(tokens):
        tag = input(f"[{i}] {token}: ").strip()
        tags.append(tag if tag else 'O')

    return tags

# Usage
with open('data/gold_standard/to_annotate.jsonl') as f:
    for line in f:
        example = json.loads(line)
        tags = annotate_sentence(example['text'], example['tokens'])
        example['gold_tags'] = tags
        save_annotation(example)
```

---

## Annotator Selection

### Single Annotator

**When:**
- Pilot study
- Budget constraints
- Highly specialized domain (only 1 expert available)

**Pros:**
- Fast
- Consistent (no inter-annotator disagreement)
- Low cost

**Cons:**
- Personal bias
- No reliability measure (IAA)
- Not suitable for publication

### Multiple Annotators (Recommended ✅)

**Recommended Setup for PaleoBERT:**
- **Annotator 1:** Geologist/Paleontologist (entity accuracy)
- **Annotator 2:** NLP Engineer (BIO consistency, technical QA)
- **Optional Annotator 3:** Tie-breaker for disagreements

**Workflow:**
1. Both annotate same data independently
2. Measure inter-annotator agreement (IAA)
3. Resolve disagreements through discussion
4. Finalize gold standard

---

## Inter-Annotator Agreement (IAA)

### Purpose

Measure annotation reliability and guideline quality.

**Target:** Cohen's Kappa ≥ 0.80 (almost perfect agreement)

### Cohen's Kappa (2 annotators)

```python
from sklearn.metrics import cohen_kappa_score

annotator_1 = ['B-TAXON', 'I-TAXON', 'O', 'B-STRAT', ...]
annotator_2 = ['B-TAXON', 'I-TAXON', 'O', 'B-STRAT', ...]

kappa = cohen_kappa_score(annotator_1, annotator_2)

# Interpretation:
# 0.81-1.00: Almost perfect ✅
# 0.61-0.80: Substantial ⚠️
# 0.41-0.60: Moderate ❌ (revise guidelines)
# 0.21-0.40: Fair ❌❌ (major issues)
# 0.00-0.20: Slight ❌❌❌ (restart)
```

### F1-based Agreement (Entity-level)

```python
def compute_entity_agreement(entities_1, entities_2):
    """
    Compute agreement as F1 score.
    Treat annotator 1 as gold, annotator 2 as prediction.
    """
    precision = len(entities_1 & entities_2) / len(entities_2)
    recall = len(entities_1 & entities_2) / len(entities_1)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# Target: F1 ≥ 0.95
```

### Adjudication (Resolving Disagreements)

**Method 1: Majority Vote (3+ annotators)**
```
Annotator 1: TAXON
Annotator 2: TAXON
Annotator 3: O
→ Final: TAXON (2/3)
```

**Method 2: Expert Arbitration (2 annotators)**
```
Annotator 1: TAXON
Annotator 2: STRAT
→ Domain expert reviews → Final: TAXON
```

**Method 3: Conservative Approach**
```
If disagreement → Use O tag (exclude ambiguous cases)
```

---

## Quality Control Workflow

### Phase 1: Pilot Annotation (20-30 sentences)

**Steps:**
1. Annotator training on guidelines
2. Annotate 20 sentences independently
3. Measure IAA
4. Discuss disagreements
5. **Revise guidelines** based on findings

**Decision:**
- Kappa ≥ 0.80 → Proceed to main annotation
- Kappa < 0.80 → Iterate (revise guidelines, re-annotate)

### Phase 2: Main Annotation

**Steps:**
1. Annotate remaining sentences (280-480)
2. Periodic IAA checks (every 100 sentences)
3. Adjudication meetings for disagreements
4. Final IAA calculation

### Phase 3: Validation

**Automated checks:**
```python
# scripts/validate_gold_standard.py

def validate_annotations(data):
    errors = []

    # 1. Format validation
    for ex in data:
        if len(ex['tokens']) != len(ex['ner_tags']):
            errors.append(f"Length mismatch: {ex['id']}")

    # 2. BIO consistency
    for tags in all_tags:
        for i, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if i == 0 or not tags[i-1].endswith(entity_type):
                    errors.append(f"Invalid BIO sequence at {i}: {tag}")

    # 3. Entity distribution check
    counts = count_entities(data)
    for entity_type in ['TAXON', 'STRAT', 'CHRONO', 'LOC']:
        if counts[entity_type] < 50:
            errors.append(f"Low coverage for {entity_type}: {counts[entity_type]}")

    # 4. Guideline compliance
    # Check for common errors (e.g., "sp." tagged as TAXON)

    return errors
```

---

## PaleoBERT Gold Standard Plan

### NER Gold Standard

**Target:** 400 sentences

**Sampling Strategy:**
```python
sources = {
    'test_set': 200,              # Validate current 99.18% F1
    'train_set': 50,               # Check training data quality
    'new_papers_cambrian': 100,   # Generalization (same period)
    'new_papers_ordovician': 50,  # Generalization (different period)
}

entity_distribution = {
    'TAXON': 100,
    'STRAT': 100,
    'CHRONO': 100,
    'LOC': 100,
}
```

**Annotators:**
- Geologist/Paleontologist (domain expertise)
- NLP Engineer (technical QA)

**Tools:** Label Studio

**Timeline:** 20-30 hours total
- Pilot: 4 hours
- Main annotation: 12-16 hours
- Adjudication: 4-6 hours
- QA: 2-4 hours

**Cost Estimate:** $500-800 (assuming $25/hour expert rate)

### RE Gold Standard

**Target:** 300 relation pairs

**Sampling Strategy:**
```python
relations = {
    'occurs_in': 70,
    'found_at': 70,
    'assigned_to': 50,     # Rare but critical
    'part_of': 50,          # Rare but critical
    'NO_RELATION': 60,
}
```

**Source:** Extract from NER gold standard sentences

**Timeline:** 12-16 hours

**Cost Estimate:** $300-400

---

## Expected Outcomes

### Scenario 1: High Agreement (F1 ≥ 95%)

**Interpretation:**
- Auto-annotation quality better than expected
- Model performance validated
- Current metrics reliable

**Action:** Proceed with production deployment

### Scenario 2: Moderate Agreement (F1 85-95%)

**Interpretation:**
- Auto-annotation adequate for training
- Actual performance slightly lower than measured
- Expected scenario ✅

**Action:**
- Report realistic metrics
- Deploy with appropriate confidence intervals

### Scenario 3: Low Agreement (F1 75-85%)

**Interpretation:**
- Significant auto-annotation bias
- Model overfit to pattern artifacts
- Generalization issues

**Action:**
- Retrain with gold standard data
- Improve auto-annotation patterns
- Consider active learning

### Scenario 4: Very Low Agreement (F1 < 75%)

**Interpretation:**
- Auto-annotation fundamentally flawed
- Model learned wrong patterns
- Major revision needed

**Action:**
- Full pipeline review
- Manual annotation required
- Reconsider approach

---

## Deliverables

### Documentation
- ✅ `docs/gold_standard_annotation_guide.md` - Complete guidelines
- ✅ `docs/gold_standard_iaa_report.md` - Inter-annotator agreement results

### Data
- ✅ `data/gold_standard/ner_gold.jsonl` - 400 gold NER sentences
- ✅ `data/gold_standard/re_gold.jsonl` - 300 gold RE pairs
- ✅ `data/gold_standard/iaa_raw.jsonl` - Raw multi-annotator data

### Scripts
- ✅ `scripts/sample_gold_standard.py` - Sampling strategy implementation
- ✅ `scripts/validate_gold_standard.py` - Quality validation
- ✅ `scripts/compute_iaa.py` - IAA calculation
- ✅ `scripts/evaluate_on_gold.py` - Model evaluation

### Results
- ✅ `results/ner_gold_evaluation.json` - NER performance on gold standard
- ✅ `results/re_gold_evaluation.json` - RE performance on gold standard

---

## Cost-Benefit Analysis

| Approach | Time | Cost | Confidence | Recommended |
|----------|------|------|------------|-------------|
| **No gold standard** | 0h | $0 | Low | ❌ Not for production |
| **50 sentences** | 4h | $100 | Moderate | ⚠️ Pilot only |
| **100 sentences** | 8h | $200 | Good | ⚠️ Minimum viable |
| **300 sentences** | 20h | $500 | High | ✅ **Recommended** |
| **500 sentences** | 35h | $875 | Very high | ✅ Publication-ready |
| **1000+ sentences** | 70h+ | $1750+ | Publication | 🏆 Research standard |

**PaleoBERT Recommendation:** 400 NER + 300 RE = $800 total

**ROI Justification:**
- Validates $2000+ worth of training compute
- Enables confident production deployment
- Provides publishable metrics
- Identifies improvement opportunities

---

## Implementation Checklist

### Phase 1: Preparation
- [ ] Write annotation guidelines document
- [ ] Set up Label Studio
- [ ] Prepare sample data for annotation
- [ ] Recruit annotators (2 people)
- [ ] Train annotators on guidelines

### Phase 2: Pilot
- [ ] Annotate 20-30 pilot sentences
- [ ] Compute IAA (target Kappa ≥ 0.80)
- [ ] Discuss disagreements
- [ ] Revise guidelines
- [ ] Re-measure IAA

### Phase 3: Main Annotation
- [ ] NER: Annotate 400 sentences
- [ ] RE: Annotate 300 pairs
- [ ] Periodic IAA checks
- [ ] Adjudication meetings

### Phase 4: Quality Control
- [ ] Run validation scripts
- [ ] Final IAA calculation
- [ ] Export gold standard JSONL
- [ ] Document methodology

### Phase 5: Evaluation
- [ ] Evaluate NER model on gold standard
- [ ] Evaluate RE model on gold standard
- [ ] Compare to auto-annotation metrics
- [ ] Document findings

---

## Alternative: Active Learning Approach

**If budget/time constrained:**

### Hybrid Strategy

1. **Start small:** 100 gold sentences
2. **Identify errors:** Find model's worst mistakes
3. **Targeted annotation:** Focus on error-prone cases
4. **Iterative improvement:** Retrain → Annotate → Repeat

**Cost:** $200 initial, $100 per iteration

**Outcome:** Better model with less annotation

---

## References

### Annotation Standards
- CoNLL-2003 NER shared task guidelines
- ACE (Automatic Content Extraction) relation annotation
- BRAT annotation tool documentation

### IAA Metrics
- Cohen's Kappa for inter-annotator agreement
- Fleiss' Kappa for 3+ annotators
- Krippendorff's Alpha for complex schemes

### Tools
- Label Studio: https://labelstud.io/
- Prodigy: https://prodi.gy/
- BRAT: http://brat.nlplab.org/

---

## Next Actions

### Immediate (This Phase)
1. ⏳ Draft `docs/gold_standard_annotation_guide.md`
2. ⏳ Set up Label Studio environment
3. ⏳ Implement `scripts/sample_gold_standard.py`
4. ⏳ Recruit annotators

### After Pilot Complete
1. Main NER annotation (400 sentences)
2. Main RE annotation (300 pairs)
3. Evaluate models on gold standard
4. Document true performance metrics

### After Evaluation
1. Compare gold vs auto-annotation metrics
2. Identify systematic errors
3. Plan model improvements
4. Prepare for production deployment

---

## Notes

- **Gold standard is investment, not cost** - Validates all downstream work
- **Start small** - Pilot with 100 sentences before committing to 400
- **Multi-annotator essential** - IAA provides reliability measure
- **Guidelines are living document** - Update based on edge cases found
- **Conservative when uncertain** - Better to exclude ambiguous cases than introduce noise

---

**Status:** Planning
**Next Milestone:** M4 Phase 1 (Gold Standard Creation)
**Blocked By:** None (can proceed in parallel with RE improvement)
**Last Updated:** 2025-10-31
