# Architecture Review and Phased Development Strategy

**Date:** 2025-10-28
**Session:** Project architecture review and strategic planning
**Status:** Planning - Strategic direction established

---

## Executive Summary

Comprehensive review of PaleoBERT's architecture, identifying BERT's inherent limitations for document-level reasoning, and establishing a pragmatic phased development strategy prioritizing NER over RE, with document-level extensions as future work.

**Key Decision:** Focus on NER as core deliverable (high ROI), treat RE as best-effort bonus (caption-level only), defer document-level reasoning to Phase 3+.

---

## 1. System Input/Output Analysis

### 1.1 Input Format

**JSONL format with dual text representation:**

```json
{
  "raw_text": "Asaphiscus wheeleri was found in the Wheeler Formation at the House Range locality. This taxon occurs in Cambrian Stage 10 strata.",
  "norm_text": "Asaphiscus wheeleri was found in the Wheeler Formation at the House_Range locality. This taxon occurs in Cambrian_Stage_10 strata.",
  "align_map": {"0": 0, "1": 1, "...": "..."},
  "pub_id": "paper_12345",
  "cap_id": "fig3_caption"
}
```

**Input sources:**
- Open-access paleontology papers (captions, methods, stratigraphy sections)
- Geological bulletins/reports
- Museum specimen notes (where permitted)

### 1.2 Output Format

**Structured JSON with entities, relations, and provenance:**

```json
{
  "pub_id": "paper_12345",
  "cap_id": "fig3_caption",
  "raw_text": "...",

  "entities": [
    {
      "id": "e1",
      "type": "TAXON",
      "surface_form": "Asaphiscus wheeleri",
      "canonical_id": "TAXON:Asaphiscus_wheeleri",
      "char_start": 0,
      "char_end": 19,
      "confidence": 0.97
    },
    {
      "id": "e2",
      "type": "STRAT",
      "surface_form": "Wheeler Formation",
      "canonical_id": "STRAT:Wheeler_Fm",
      "char_start": 33,
      "char_end": 50,
      "confidence": 0.95
    }
  ],

  "relations": [
    {
      "subject": "e1",
      "predicate": "occurs_in",
      "object": "e2",
      "confidence": 0.89,
      "evidence_span": [0, 80],
      "evidence_type": "explicit"
    }
  ],

  "provenance": {
    "model_version": "paleo-pipeline-v1",
    "ner_model": "paleo-ner-v1",
    "re_model": "paleo-re-v1",
    "extracted_at": "2025-10-28T12:00:00Z"
  }
}
```

### 1.3 Target Information

**Entities (4 types):**
- **TAXON**: Organisms (Asaphiscus, Olenellus, Trilobita)
- **STRAT**: Stratigraphic units (Wheeler Formation, Burgess Shale)
- **CHRONO**: Chronostratigraphic units (Cambrian Stage 10, Paibian, Series 2)
- **LOC**: Localities (House Range, Utah, Antarctica)

**Relations (4 types):**
- **occurs_in**: TAXON → STRAT (taxon found in formation)
- **found_at**: TAXON → LOC (taxon found at locality)
- **part_of**: STRAT → STRAT (stratigraphic hierarchy)
- **assigned_to**: STRAT → CHRONO (formation assigned to time period)

---

## 2. Tokenizer Architecture Deep Dive

### 2.1 Role of Domain Vocabulary

**Critical clarification:** Domain vocabulary files (taxa.txt, strat_units.txt, etc.) are NOT for weight initialization or training data. They are for **tokenizer vocabulary expansion**.

**Process:**

```python
# Step 1: Add tokens to tokenizer vocabulary
tokenizer.add_tokens(["Asaphiscus", "Paibian", "Wheeler_Formation", ...])
# Returns: number of tokens added (~500)

# Step 2: Resize model embedding layer
model.resize_token_embeddings(len(tokenizer))
# Creates new embedding vectors (initialized randomly)

# Step 3: DAPT learns embeddings for new tokens
# 100M tokens of paleontology text → new token embeddings learn meaning
```

### 2.2 Tokenization Behavior

**With domain vocabulary (PaleoBERT):**
```
"Olenellus" → ["Olenellus"]  (1 token) ✅
"Jiangshanian" → ["Jiangshanian"]  (1 token) ✅
"Cambrian_Stage_10" → ["Cambrian_Stage_10"]  (1 token) ✅
```

**Without domain vocabulary (base DeBERTa):**
```
"Olenellus" → ["O", "##len", "##ell", "##us"]  (4 tokens) ❌
"Jiangshanian" → ["Ji", "##ang", "##shan", "##ian"]  (4 tokens) ❌
```

**Novel terms (not in vocabulary):**
```
"Newtaxoniscus" → ["New", "##tax", "##on", "##is", "##cus"]  (5 subwords) ✅
# Still processable! No [UNK] token
# Subword components carry semantic information
```

### 2.3 Benefits of Domain Vocabulary

1. **Efficiency:** 75% reduction in sequence length for domain terms
2. **Semantic preservation:** "Cambrian" as single token vs fragmented ["Cam", "##brian"]
3. **Fragmentation rate reduction:** 95% → 0% for added terms

**Fragmentation rate metric:**
```python
def fragmentation_rate(tokenizer, terms):
    fragmented = 0
    for term in terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        if len(tokens) > 1:
            fragmented += 1
    return fragmented / len(terms) * 100

# Expected: 95% before → 0% after for domain terms
```

---

## 3. NER Capability for Novel Terms

### 3.1 Core Principle: Context-based Recognition

**Key insight:** Deep learning NER learns **contextual patterns**, not word memorization.

**Training examples:**
```
"Olenellus wheeleri occurs in the Marjum Formation"
  [TAXON]  [TAXON]              [STRAT] [STRAT]

"Asaphiscus bonnensis found in Wheeler Shale"
  [TAXON]   [TAXON]               [STRAT] [STRAT]
```

**Model learns:**
- ❌ NOT: "Olenellus" = TAXON (memorization)
- ✅ YES: "X occurs in Y" pattern → X=TAXON, Y=STRAT
- ✅ YES: Latin binomial pattern (2 capitalized words)
- ✅ YES: Morphological patterns ("-iscus", "-idae", "-morpha" → TAXON)
- ✅ YES: Positional/syntactic cues

### 3.2 Inference on Novel Terms

**Example: Never-seen taxon**
```python
Input: "Newtaxoniscus mysteriosus occurs in the Wheeler Formation"

# Inference process:
# 1. Detect "occurs in" pattern
# 2. Position before "occurs in" → likely TAXON
# 3. Two capitalized words → Latin binomial pattern
# 4. Subword "##is", "##cus" seen in training (Asaphiscus, etc.)
# 5. "Wheeler Formation" in vocabulary → STRAT confirmed

Output:
"Newtaxoniscus mysteriosus occurs in the Wheeler Formation"
  [TAXON]       [TAXON]               [STRAT] [STRAT]
  conf=0.87     conf=0.87             conf=0.95
```

### 3.3 Performance Expectations

```
┌─────────────────────────────┬────────────┬────────────┐
│         Term Type           │  Precision │   Recall   │
├─────────────────────────────┼────────────┼────────────┤
│ In vocabulary               │   ~95%     │   ~92%     │
│ (Olenellus, Wheeler_Fm)     │            │            │
├─────────────────────────────┼────────────┼────────────┤
│ Novel but similar pattern   │   ~85%     │   ~78%     │
│ (Newtaxoniscus - binomial)  │            │            │
├─────────────────────────────┼────────────┼────────────┤
│ Completely atypical         │   ~65%     │   ~55%     │
│ (Specimen ABC-123)          │            │            │
└─────────────────────────────┴────────────┴────────────┘
```

**Subword advantage:**
```
"Newtaxoniscus" → ["New", "##tax", "##on", "##is", "##cus"]
                      |       |       |       |       |
                    new   taxonomy organism form  Latin suffix

# Shared suffixes with training data:
"Asaphiscus" → ["Asaph", "##is", "##cus"]  ← overlapping subwords!
→ Similar embeddings → similar predictions
```

---

## 4. BERT Architecture Limitations

### 4.1 Fixed Context Window Constraint

**Fundamental limitation:**
```
max_seq_length = 512 tokens (or 384 for NER/RE)
                 ^^^
                 Cannot process longer contexts

# Typical paper structure:
Title:        20 tokens
Abstract:     200 tokens
Introduction: 500 tokens  ← already exceeds 512!
Methods:      1000 tokens
Results:      2000 tokens
Total:        5000+ tokens

→ BERT cannot see entire document at once
```

### 4.2 Self-Attention Complexity

```
Sequence length 2× → Computation 4× (O(n²))

512 tokens:  512² = 262,144 operations
2048 tokens: 2048² = 4,194,304 operations (16× more!)

→ VRAM explosion
→ Long document processing infeasible
```

### 4.3 Implications for Relation Extraction

**Current PaleoBERT design (OVERVIEW.md § 4.1):**
> "Build candidate pairs from NER outputs **within sentence/caption windows**"

**Critical constraint:** RE only works for entities in the **same sentence/caption**.

**Problematic scenario:**
```
┌─────────────────────────────────────────────┐
│ Title: Trilobites of the Marjum Formation  │  ← "Marjum Formation" (STRAT)
├─────────────────────────────────────────────┤
│ Section 3.2: Systematic Paleontology        │
│                                             │
│ Olenellus wheeleri Clark, 1924              │  ← "Olenellus wheeleri" (TAXON)
│                                             │
│ Description: Cephalon semi-circular with    │  ← No mention of Formation!
│ prominent genal spines...                   │
└─────────────────────────────────────────────┘

# NER results:
Sentence 1: [("Marjum Formation", "STRAT")]
Sentence 2: [("Olenellus wheeleri", "TAXON")]

# RE results:
No relation extracted ❌
# Entities not in same window → RE model never sees them together
```

### 4.4 Success Cases: Figure Captions

**Why captions work well:**
```
✅ "Figure 3. Olenellus wheeleri from the Marjum Formation,
              House Range, Utah. Cambrian Stage 10."

# All entities in ONE caption:
NER: [TAXON, STRAT, LOC, CHRONO]
RE: All relations extractable!
  - (Olenellus wheeleri, occurs_in, Marjum Formation)
  - (Olenellus wheeleri, found_at, House Range)
  - (Marjum Formation, located_at, Utah)
  - (Marjum Formation, assigned_to, Cambrian Stage 10)
```

---

## 5. Document-level Reasoning Requirements

### 5.1 The Gap

**Current architecture:**
```
BERT (sentence/caption level)
  ↓
✅ Explicit relations (same window)
❌ Implicit relations (cross-sentence)
❌ Document-level metadata inference
```

**What's missing:**
1. Primary formation/locality/age identification from document metadata
2. Implicit relation inference (entity in Results + formation in Title)
3. Cross-document knowledge integration

### 5.2 Proposed Extension Architecture

```
┌─────────────────────────────────────────────────┐
│         PaleoBERT Core (Current Scope)          │
│                                                 │
│  Raw Text → NER → RE (sentence/caption level)  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│      Document-level Module (Future Work)        │
│                                                 │
│  1. Metadata Extractor                         │
│     • Primary formation (from title/abstract)   │
│     • Primary locality (from title/abstract)    │
│     • Primary age (from stratigraphy section)   │
│                                                 │
│  2. Implicit Relation Inference                │
│     • Entity + Metadata → likely relations      │
│     • Confidence scoring (0.4 ~ 0.7 range)     │
│                                                 │
│  3. Knowledge Graph Integration                │
│     • Cross-document co-occurrence patterns     │
│     • Taxon-formation associations from corpus  │
│     • Confidence boosting from multiple sources │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│              Unified Output                     │
│                                                 │
│  Explicit relations (high confidence, 0.8-0.95) │
│  + Implicit relations (medium conf, 0.4-0.7)    │
│  + Provenance tracking (explicit vs inferred)   │
└─────────────────────────────────────────────────┘
```

### 5.3 Evidence Layering Example

```json
{
  "triple": ["Olenellus wheeleri", "occurs_in", "Marjum Formation"],
  "evidence": [
    {
      "type": "explicit",
      "confidence": 0.92,
      "source": "sentence_47",
      "text": "...occurs in the Marjum Formation..."
    },
    {
      "type": "document_metadata",
      "confidence": 0.58,
      "source": "paper_title",
      "text": "Trilobites of the Marjum Formation"
    },
    {
      "type": "knowledge_graph",
      "confidence": 0.85,
      "source": "cross_document_stats",
      "text": "Co-occurred in 47/50 papers"
    }
  ]
}
```

**User filtering options:**
- High precision: `evidence_type == "explicit" AND confidence > 0.80`
- High recall: `confidence > 0.50` (include all evidence types)

---

## 6. Phased Development Strategy

### 6.1 Revised Milestone Plan

**Original plan (OVERVIEW.md):**
```
M1: DAPT complete
M2: NER baseline (F1 ≥ 0.80~0.90)
M3: RE baseline (F1 ≥ 0.75)
M4: End-to-end pipeline
M5: Release
```

**Revised pragmatic plan:**

```
┌─────────────────────────────────────────────────┐
│ Phase 1: NER-Centric (CORE VALUE)              │
├─────────────────────────────────────────────────┤
│ M1: DAPT ✅                                     │
│     Goal: MLM perplexity improvement            │
│     Timeline: 20-30 hours training              │
│                                                 │
│ M2: NER ✅✅✅ (PRIMARY DELIVERABLE)             │
│     Goal: F1 ≥ 0.85 (realistic target)          │
│     Validation: 4 entity types × F1 ≥ 0.80      │
│     Deliverable: NER-only API/tool              │
│     Value: Sufficient for deployment!           │
│     Timeline: 2-3 weeks (data + training + eval)│
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Phase 2: RE Experiment (BEST EFFORT)           │
├─────────────────────────────────────────────────┤
│ M3: Caption-level RE ⚠️                        │
│     Goal: F1 ≥ 0.60 (lowered expectation)       │
│     Scope: Figure/table captions ONLY           │
│     Success criteria:                           │
│       - occurs_in F1 ≥ 0.70 (critical relation) │
│       - Precision ≥ 0.65 (avoid false positives)│
│     Failure condition: <0.60 after 3 epochs     │
│       → Abort and move to Phase 3               │
│     Timeline: 1-2 weeks experiment              │
│                                                 │
│ M4: Basic Pipeline Release ✅                   │
│     Components: NER (guaranteed) + RE (bonus)   │
│     Version: v1.0 (production-ready)            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Phase 3: Document-level (FUTURE RESEARCH)      │
├─────────────────────────────────────────────────┤
│ M5.1: Metadata Extraction                       │
│     Component: Document-level primary entity ID │
│     Timeline: 1-2 months (separate project)     │
│                                                 │
│ M5.2: Implicit Relation Inference               │
│     Component: Entity + metadata → relations    │
│     Timeline: 2-3 months (research phase)       │
│                                                 │
│ M5.3: Knowledge Graph Integration               │
│     Component: Cross-document reasoning         │
│     Timeline: 3-6 months (separate publication?)│
└─────────────────────────────────────────────────┘
```

### 6.2 Priority Justification

**Why NER first:**

| Capability | NER Only | NER + RE (caption) | NER + Doc-level |
|------------|----------|-------------------|----------------|
| Taxa extraction | ✅ | ✅ | ✅ |
| Formation extraction | ✅ | ✅ | ✅ |
| Locality extraction | ✅ | ✅ | ✅ |
| Age extraction | ✅ | ✅ | ✅ |
| **Direct relations** | ❌ | ✅ (50-70%) | ✅ (70-80%) |
| **Implicit relations** | ❌ | ❌ | ✅ (40-60%) |
| **Search indexing** | ✅ | ✅ | ✅ |
| **Auto-tagging** | ✅ | ✅ | ✅ |
| **Database population** | ✅ | ✅ | ✅ |

**Value proposition:**
- NER alone delivers 70% of use cases
- RE adds 20% (but 4× effort)
- Doc-level adds 10% (but 10× effort)

### 6.3 RE Success Rate by Context

```
┌─────────────────┬──────────┬─────────────────────┐
│   Context       │ RE F1    │  Reason             │
├─────────────────┼──────────┼─────────────────────┤
│ Figure captions │ 70-80%   │ Dense, explicit     │
│ Table captions  │ 65-75%   │ Structured format   │
│ Abstracts       │ 50-60%   │ Long, implicit      │
│ Methods         │ 30-40%   │ Descriptive, vague  │
│ Results         │ 40-50%   │ Cross-references    │
└─────────────────┴──────────┴─────────────────────┘
```

**Strategic focus:** Target high-success contexts first (captions), defer low-success contexts to Phase 3.

### 6.4 ROI Analysis

```
┌─────────────┬──────────┬──────────┬─────────────┐
│   Phase     │ Effort   │ Success  │  ROI        │
├─────────────┼──────────┼──────────┼─────────────┤
│ NER         │  ★★☆☆☆  │  85-90%  │  ★★★★★     │
│ (Core)      │  Medium  │  High    │  Excellent  │
├─────────────┼──────────┼──────────┼─────────────┤
│ RE          │  ★★★★☆  │  50-70%  │  ★★★☆☆     │
│ (Caption)   │  High    │  Medium  │  Moderate   │
├─────────────┼──────────┼──────────┼─────────────┤
│ Doc-level   │  ★★★★★  │  40-60%  │  ★★☆☆☆     │
│ Metadata    │  V.High  │  Medium  │  Low        │
├─────────────┼──────────┼──────────┼─────────────┤
│ Knowledge   │  ★★★★★  │  30-50%  │  ★☆☆☆☆     │
│ Graph       │  V.High  │  Low     │  Very Low   │
└─────────────┴──────────┴──────────┴─────────────┘
```

---

## 7. Practical Implementation Guidelines

### 7.1 NER Phase (M2) - Maximize Quality

**Data requirements:**
- 5,000-20,000 labeled sentences
- Stratified split by publication (avoid leakage)
- Active learning for difficult cases

**Quality checklist:**
- [ ] Entity type coverage balanced (TAXON/STRAT/CHRONO/LOC)
- [ ] Diverse publication sources (journals, bulletins, notes)
- [ ] Edge cases included (abbreviations, novel taxa, informal names)
- [ ] Raw text span validation (not just normalized text)

**Success metrics:**
```python
NER_SUCCESS_CRITERIA = {
    "TAXON_f1": 0.90,   # Most important
    "STRAT_f1": 0.80,
    "CHRONO_f1": 0.80,
    "LOC_f1": 0.80,
    "overall_f1": 0.85,
}
```

### 7.2 RE Phase (M3) - Pragmatic Scoping

**Scope limitation:**
```python
RE_SCOPE = {
    "contexts": ["figure_captions", "table_captions"],
    "max_distance": 384,  # tokens between entities
    "min_confidence": 0.60,
}
```

**Abort criteria:**
```python
# Decision point: after 3 training epochs
if re_f1 < 0.60 or precision < 0.65:
    log("RE performance below threshold")
    log("Recommendation: Proceed to Phase 3 (doc-level)")
    decision = "ABORT_RE_TRAINING"
```

**Hard negative mining:**
```python
# Essential for RE training
# Example: TAXON and STRAT in same caption but no relation
"Figure 3. Olenellus wheeleri specimen. Scale bar: 5mm.
 From the Wheeler Formation collection."

→ ("Olenellus wheeleri", "Wheeler Formation") in same caption
→ But relation is INDIRECT (collection context, not occurrence)
→ Label: NONE (hard negative)
```

### 7.3 Document-level Phase (M5) - Research Mode

**Metadata extraction approach:**
```python
def extract_document_metadata(paper):
    """
    Extract primary entities from high-weight sections
    """
    # Weight by section
    weights = {
        "title": 3.0,
        "abstract": 2.0,
        "introduction": 1.0,
        "body": 0.5,
    }

    # Aggregate entity mentions
    entity_scores = defaultdict(float)
    for section, weight in weights.items():
        entities = ner_model(paper.sections[section])
        for entity in entities:
            entity_scores[entity] += weight

    # Select primary entities
    return {
        "primary_formation": top_entity(entity_scores, type="STRAT"),
        "primary_locality": top_entity(entity_scores, type="LOC"),
        "primary_age": top_entity(entity_scores, type="CHRONO"),
    }
```

---

## 8. Key Decisions and Rationale

### Decision 1: NER as Minimum Viable Product
**Rationale:** NER alone provides 70% of user value with 20% of effort. Ensures deliverable even if RE fails.

### Decision 2: Lower RE expectations
**Rationale:** BERT's context window limitation makes document-level RE infeasible. Caption-level RE at F1=0.60 is realistic and still useful.

### Decision 3: Defer document-level to Phase 3
**Rationale:** Requires architectural additions beyond BERT (metadata extraction, inference layer, KG integration). Separate research project scope.

### Decision 4: Caption-centric strategy
**Rationale:** Captions naturally contain dense entity co-occurrences. Highest ROI for RE component.

### Decision 5: Explicit failure criteria for RE
**Rationale:** Avoid endless tuning. Clear abort conditions enable pivot to Phase 3 if needed.

---

## 9. Risk Assessment

### High-confidence deliverables
- ✅ Tokenizer with domain vocabulary (P01)
- ✅ DAPT checkpoint (M1)
- ✅ NER model at F1 ≥ 0.85 (M2)

### Medium-confidence deliverables
- ⚠️ RE model at F1 ≥ 0.60 (M3) - 60% success probability
- ⚠️ Caption-level pipeline (M4) - depends on M3

### Low-confidence extensions
- 🔮 Document metadata extraction (M5.1) - research phase
- 🔮 Implicit relation inference (M5.2) - untested approach
- 🔮 Knowledge graph integration (M5.3) - long-term project

---

## 10. Next Steps

### Immediate (This week)
1. Complete P01 tokenizer setup (in progress)
   - Create domain vocabulary files (taxa.txt, etc.)
   - Build tokenizer_v1
   - Validation script

### Short-term (Next 2-4 weeks)
2. Data collection for DAPT (M1)
3. Data annotation for NER (M2)
4. Implement core modules (normalization, data loading, models)

### Medium-term (2-3 months)
5. DAPT training (20-30 hours)
6. NER training and evaluation
7. RE experiment (best effort)
8. Pipeline integration and release

### Long-term (6+ months)
9. Document-level extension (if M3 fails or post-v1.0)
10. Knowledge graph integration (research project)

---

## References

- OVERVIEW.md § 4: Relation Extraction design
- OVERVIEW.md § 5: Integration & Inference pipeline
- devlog/20251027_P01_tokenizer_setup.md: Tokenizer implementation plan

---

## Revision History

- 2025-10-28: Initial version - Architecture review and strategy established
