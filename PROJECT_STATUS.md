# PaleoBERT Project Status

**Last Updated:** 2025-10-29
**Current Phase:** P06 Planning (Corpus Collection + DAPT Training)
**Overall Progress:** 60% (Foundation Complete, Training Phase Pending)

---

## Quick Status Overview

| Phase | Status | Progress | Notes |
|-------|--------|----------|-------|
| **P01: Tokenizer Setup** | ✅ Complete | 100% | 722 tokens, 0% fragmentation |
| **P02: Normalization** | ✅ Complete | 100% | 35/35 tests passing |
| **P03: DAPT Script** | ✅ Complete | 100% | Ready for training |
| **P04: Geyer 2019 PDF** | ✅ Complete | 100% | 9K tokens extracted |
| **P05: Trilobite Catalog** | ✅ Complete | 100% | 1,248 genera, 200 test examples |
| **P06: Corpus + DAPT** | 📋 Planning | 0% | **NEXT STEP** |
| **P07: NER Training** | ⏸️ Blocked | 0% | Depends on P06 |
| **P08: RE Training** | ⏸️ Blocked | 0% | Depends on P06 |
| **P09: Pipeline** | ⏸️ Blocked | 0% | Depends on P07+P08 |

**Legend:**
- ✅ Complete
- 🔄 In Progress
- 📋 Planning
- ⏸️ Blocked
- 🔴 Blocker

---

## Completed Milestones (P01-P05)

### ✅ P01: Tokenizer Setup

**Completion Date:** 2025-10-29

**Achievements:**
- Extended DeBERTa-v3-base tokenizer with 722 domain tokens
- Categories: 419 taxa, 140 formations, 34 chrono units, 129 localities
- 0% fragmentation rate (all terms single tokens)
- Tokenizer saved at `artifacts/tokenizer_v1/`

**Deliverables:**
- `scripts/build_tokenizer.py` - Tokenizer builder
- `scripts/validate_tokenizer.py` - Fragmentation validator
- `artifacts/vocab/*.txt` - Vocabulary files
- `artifacts/tokenizer_v1/` - Extended tokenizer

**Documentation:**
- `devlog/20251029_002_P01_tokenizer_completion.md`

---

### ✅ P02: Text Normalization

**Completion Date:** 2025-10-29

**Achievements:**
- Implemented dual-text normalization (raw ↔ normalized)
- Character-level alignment maps for span projection
- 35/35 tests passing
- Supports round-trip raw→norm→raw projection

**Deliverables:**
- `src/normalization.py` - Core normalization module (412 lines)
- `tests/test_normalization.py` - Comprehensive test suite
- `examples/demo_normalization.py` - Usage examples

**Normalization Patterns:**
- Chronostratigraphy: `Stage 10` → `Stage_10`
- Formations: `Wheeler Formation` → `Wheeler_Formation`
- Localities: `House Range` → `House_Range`
- Binomials: `Olenellus wheeleri` → `Olenellus_wheeleri`

**Documentation:**
- `devlog/20251029_003_P02_normalization_implementation_complete.md`

---

### ✅ P03: DAPT Training Script

**Completion Date:** 2025-10-29

**Achievements:**
- Complete DAPT training script with validation metrics
- Supports RTX 2080 Ti (11GB VRAM) with gradient checkpointing
- Custom callbacks for rare-token perplexity tracking
- YAML-based configuration system

**Deliverables:**
- `scripts/train_dapt.py` - Main training script
- `config/dapt_config.yaml` - Training configuration
- Rare-token perplexity callback
- Validation metric suite

**Configuration:**
- Batch size: 8 × 16 (gradient accumulation) = 128 effective
- Sequence length: 512
- Learning rate: 2e-4 with linear warmup
- FP16 + gradient checkpointing

**Documentation:**
- `devlog/20251029_P03_dapt_training_script.md`
- `devlog/20251029_004_P03_Phase2_validation_metrics_complete.md`

---

### ✅ P04: Geyer 2019 PDF Integration

**Completion Date:** 2025-10-29

**Achievements:**
- Extracted 9,219 tokens from IUGS Cambrian correlation chart
- Created 10 JSONL corpus entries
- Generated 3 gold-standard test examples (NER + RE)
- Established PDF extraction pipeline

**Deliverables:**
- `scripts/p04_extract_pdf_text.py` - PDF extraction
- `scripts/p04_generate_test_data.py` - Test data generation
- `data/corpus_norm/train_geyer2019.jsonl` - Corpus entries
- `data/ner/test_geyer2019.jsonl` - NER test set
- `data/re/test_geyer2019.jsonl` - RE test set

**Source:**
- Geyer, G. (2019). "A comprehensive Cambrian correlation chart"
- Episodes Vol. 42, No. 4, pp. 321-374

**Documentation:**
- `devlog/20251029_005_P04_geyer2019_integration_complete.md`

---

### ✅ P05: Trilobite Catalog Integration

**Completion Date:** 2025-10-29

**Achievements:**
- Parsed 2,839 total entries, extracted 1,248 Cambrian trilobites
- Vocabulary expansion: 337 → 722 tokens (+114%)
- Generated 200 test examples (100 NER + 100 RE)
- Created comprehensive metadata database

**Deliverables:**
- `scripts/p05_extract_trilobite_names.py` - PDF parser
- `scripts/p05_update_vocabulary.py` - Vocabulary merger
- `scripts/p05_generate_test_data.py` - Test generator
- `data/trilobite_metadata.json` - 1,248 genera metadata
- `data/ner/test_trilobite.jsonl` - 100 NER examples
- `data/re/test_trilobite.jsonl` - 100 RE examples

**Vocabulary Breakdown:**
- Taxa: 222 → 419 (+197, +89%)
- Formations: 40 → 140 (+100, +250%)
- Localities: 41 → 129 (+88, +215%)
- Chrono: 34 → 34 (no change)

**Source:**
- Jell, P.A. & Adrain, J.M. (2002). "Available Generic Names for Trilobites"
- Memoirs of the Queensland Museum 48(2): 331-553

**Documentation:**
- `devlog/20251029_006_P05_trilobite_catalog_execution_complete.md`

---

## Current Status: P06 Planning

### 📋 P06: Corpus Collection and DAPT Training

**Status:** Planning Complete, Awaiting Execution

**Current Blocker:** 🔴 **Insufficient corpus data**
- Have: ~9K tokens (Geyer 2019)
- Need: 40-50M tokens for full DAPT
- **Resolution Options:**
  1. **Plan A:** Collect full corpus (1-2 weeks)
  2. **Plan B:** Use synthetic corpus for pilot (~240K tokens)

**Recommended Approach:** Hybrid Strategy
1. ✅ Start with Plan B (synthetic + existing) for pilot DAPT
2. 🔄 Collect full corpus in parallel
3. ⏭️ Run full DAPT when corpus ready

**Timeline Estimate:**
- **Fast Path (Plan B):** 2-3 weeks (pilot DAPT)
- **Full Path (Plan A):** 4-5 weeks (full corpus + DAPT)
- **Hybrid:** 3-4 weeks (pilot → full)

**Documentation:**
- `devlog/20251029_P06_corpus_collection_and_dapt_training_plan.md`

---

## Pending Milestones (P07-P09)

### ⏸️ P07: NER Training

**Status:** Blocked (depends on P06 DAPT completion)

**Plan:**
- Fine-tune DAPT model for Named Entity Recognition
- Entity types: TAXON, STRAT, CHRONO, LOC
- Training data: 5k-20k sentences (need to collect/annotate)
- Test data: 103 examples ready (P04 + P05)

**Success Criteria:**
- F1(TAXON) ≥ 0.90
- F1(STRAT) ≥ 0.80
- F1(CHRONO) ≥ 0.80
- F1(LOC) ≥ 0.80

**Estimated Timeline:** 1-2 weeks (after P06)

---

### ⏸️ P08: RE Training

**Status:** Blocked (depends on P06 DAPT completion)

**Plan:**
- Fine-tune DAPT model for Relation Extraction
- Relations: occurs_in, found_at, part_of, assigned_to
- Training data: 5k-20k entity pairs (need to generate)
- Test data: 103 examples ready (P04 + P05)

**Success Criteria:**
- micro-F1 ≥ 0.75
- occurs_in F1 ≥ 0.80

**Estimated Timeline:** 1-2 weeks (after P06)

---

### ⏸️ P09: End-to-End Pipeline

**Status:** Blocked (depends on P07 + P08)

**Plan:**
- Integrate NER + RE models into inference pipeline
- Add entity linking and normalization
- JSON output with provenance
- Performance benchmarking

**Success Criteria:**
- JSON schema validity ≥ 98%
- Triple validity@1 ≥ 0.75
- Provenance character offset match ≥ 0.95

**Estimated Timeline:** 1 week (after P07+P08)

---

## Key Metrics

### Vocabulary Statistics

| Metric | Value |
|--------|-------|
| Total vocabulary tokens | 722 |
| Taxa | 419 |
| Stratigraphic units | 140 |
| Chronostratigraphic units | 34 |
| Localities | 129 |
| Fragmentation rate | 0% |

### Tokenizer Statistics

| Metric | Value |
|--------|-------|
| Base vocab size (DeBERTa-v3) | 128,001 |
| Domain tokens added | 698 (24 duplicates removed) |
| Final vocab size | 128,620 |
| Increase | +0.48% |

### Test Data Statistics

| Dataset | Examples | Entities | Relations |
|---------|----------|----------|-----------|
| NER (Geyer 2019) | 3 | 12 | - |
| NER (Trilobite) | 100 | 346 | - |
| RE (Geyer 2019) | 3 | 12 | 9 |
| RE (Trilobite) | 100 | 346 | 246 |
| **Total** | **206** | **716** | **255** |

### Corpus Statistics

| Source | Tokens | Status |
|--------|--------|--------|
| Geyer 2019 PDF | ~9K | ✅ Available |
| Trilobite metadata (extractable) | ~30K | ✅ Available |
| Synthetic (can generate) | ~200K | 🔄 Can generate |
| **Current Total** | **~240K** | Ready for pilot |
| **Target for Full DAPT** | **40-50M** | 🔴 Need to collect |

---

## File Structure

```
PaleoBERT/
├── README.md
├── PROJECT_STATUS.md                    # This file
├── OVERVIEW.md                          # Training design spec
├── CLAUDE.md                            # Project instructions
├── VOCABULARY_EXPANSION_PLAN.md
├── FAMILY_ARCHITECTURE.md
│
├── artifacts/
│   ├── vocab/                           # 722 terms (4 files)
│   └── tokenizer_v1/                    # Extended tokenizer
│
├── data/
│   ├── trilobite_entries.json           # 2,839 parsed entries
│   ├── trilobite_cambrian.json          # 1,248 Cambrian entries
│   ├── trilobite_metadata.json          # Metadata DB
│   ├── corpus_norm/
│   │   └── train_geyer2019.jsonl        # 10 entries, ~9K tokens
│   ├── ner/
│   │   ├── test_geyer2019.jsonl         # 3 examples
│   │   └── test_trilobite.jsonl         # 100 examples
│   └── re/
│       ├── test_geyer2019.jsonl         # 3 examples
│       └── test_trilobite.jsonl         # 100 examples
│
├── scripts/
│   ├── build_tokenizer.py               # ✅ P01
│   ├── validate_tokenizer.py            # ✅ P01
│   ├── train_dapt.py                    # ✅ P03
│   ├── p04_extract_pdf_text.py          # ✅ P04
│   ├── p04_generate_test_data.py        # ✅ P04
│   ├── p05_extract_trilobite_names.py   # ✅ P05
│   ├── p05_update_vocabulary.py         # ✅ P05
│   └── p05_generate_test_data.py        # ✅ P05
│
├── src/
│   └── normalization.py                 # ✅ P02 (412 lines)
│
├── tests/
│   └── test_normalization.py            # ✅ P02 (35 tests)
│
├── config/
│   └── dapt_config.yaml                 # ✅ P03
│
└── devlog/
    ├── 20251029_002_P01_tokenizer_completion.md
    ├── 20251029_003_P02_normalization_implementation_complete.md
    ├── 20251029_004_P03_Phase2_validation_metrics_complete.md
    ├── 20251029_005_P04_geyer2019_integration_complete.md
    ├── 20251029_006_P05_trilobite_catalog_execution_complete.md
    └── 20251029_P06_corpus_collection_and_dapt_training_plan.md
```

---

## Next Steps (Recommended)

### Immediate Action (Next Session)

**Option 1: Fast Path (Recommended)**

1. **Generate synthetic corpus** (1 day)
   ```bash
   python scripts/p06_generate_synthetic_corpus.py \
     --metadata data/trilobite_metadata.json \
     --output data/corpus_norm/synthetic_v1.jsonl \
     --num_samples 10000
   ```

2. **Merge with existing data** (1 hour)
   ```bash
   cat data/corpus_norm/train_geyer2019.jsonl \
       data/corpus_norm/synthetic_v1.jsonl \
       > data/corpus_norm/train_pilot.jsonl
   ```

3. **Run pilot DAPT** (2-3 days)
   ```bash
   python scripts/train_dapt.py \
     --config config/dapt_pilot_config.yaml
   ```

4. **Evaluate and decide** (1 day)
   - If good: Proceed to P07 (NER)
   - If poor: Collect more corpus, iterate

**Option 2: Full Corpus Path**

1. **Start corpus collection** (1-2 weeks)
   - Identify 300-500 open-access PDFs
   - Legal review
   - Download and extract

2. **Preprocess corpus** (2-3 days)
   - Clean and normalize
   - Validate format

3. **Run full DAPT** (28 hours GPU)
   - Train on 40-50M tokens
   - 3 epochs

4. **Proceed to P07**

### Medium-term (After P06)

5. **P07: NER Training** (1-2 weeks)
   - Collect/annotate NER training data
   - Fine-tune DAPT model
   - Evaluate on test sets

6. **P08: RE Training** (1-2 weeks)
   - Generate RE training pairs
   - Fine-tune DAPT model
   - Evaluate on test sets

7. **P09: Pipeline Integration** (1 week)
   - Build inference pipeline
   - End-to-end testing
   - Performance benchmarking

---

## Risk Assessment

### Current Risks

**🔴 HIGH: Corpus Collection Blocker**
- **Issue:** Need 40-50M tokens, have ~9K
- **Impact:** Cannot proceed with full DAPT
- **Mitigation:** Use synthetic corpus for pilot (Plan B)

**🟡 MEDIUM: Training Time**
- **Issue:** 28-hour DAPT training requires GPU availability
- **Impact:** Timeline delays if GPU not available
- **Mitigation:** Use cloud GPU if needed, or train pilot first (smaller corpus)

**🟢 LOW: Technical Issues**
- **Issue:** Training script bugs, VRAM overflow
- **Impact:** Minor delays (1-2 days)
- **Mitigation:** Already tested training script (P03), monitoring in place

---

## Success Criteria (Overall Project)

### Milestone 1: DAPT (P06)
- ✅ Held-out MLM perplexity ≤ baseline - 10%
- ✅ Rare-token perplexity ≤ baseline - 20%
- ✅ Fragmentation rate < 10%

### Milestone 2: NER (P07)
- ✅ F1(TAXON) ≥ 0.90
- ✅ F1(STRAT) ≥ 0.80
- ✅ F1(CHRONO) ≥ 0.80
- ✅ F1(LOC) ≥ 0.80

### Milestone 3: RE (P08)
- ✅ micro-F1 ≥ 0.75
- ✅ occurs_in F1 ≥ 0.80

### Milestone 4: Pipeline (P09)
- ✅ JSON schema validity ≥ 98%
- ✅ Triple validity@1 ≥ 0.75
- ✅ Provenance character offset match ≥ 0.95

---

## Team Notes

**Hardware Requirements:**
- GPU: 1× RTX 2080 Ti (11GB VRAM) or equivalent
- RAM: 64-128 GB
- Storage: 50 GB (corpus + checkpoints)

**Key Dependencies:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PyMuPDF, sentencepiece, protobuf

**Contact:**
- Project Documentation: `CLAUDE.md`
- Technical Details: `OVERVIEW.md`
- Devlog: `devlog/` directory

---

**Last Updated:** 2025-10-29

**Status:** Foundation complete (P01-P05), ready for corpus collection and DAPT (P06)

**Recommendation:** ✅ Start with Plan B (synthetic corpus pilot) to unblock downstream work
