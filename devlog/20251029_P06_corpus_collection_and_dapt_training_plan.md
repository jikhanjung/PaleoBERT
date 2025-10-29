# P06: Corpus Collection and DAPT Training Plan

**Date:** 2025-10-29
**Milestone:** P06 (Corpus Collection + DAPT Training)
**Status:** 📋 PLANNING
**Dependencies:** P01-P05 completed

---

## Executive Summary

This document outlines the plan for **Phase 6: Corpus Collection and Domain-Adaptive Pretraining (DAPT)**. Currently, we have completed vocabulary expansion (722 tokens) and tokenizer setup, but we only have ~9K tokens of corpus data (from Geyer 2019 PDF). The next critical step is to collect a comprehensive Cambrian paleontology corpus (target: 40-50M tokens) and perform DAPT training.

**Current Status:**
- ✅ Tokenizer built with 722 domain tokens (0% fragmentation)
- ✅ Normalization pipeline ready (P02)
- ✅ DAPT training script available (`scripts/train_dapt.py`)
- ⚠️ **BLOCKER:** Insufficient corpus data (9K tokens vs. target 40-50M)
- ✅ Test data ready (103 NER + 103 RE examples)

**Critical Path:**
```
P06.1: Corpus Collection (1-2 weeks)
  └─> P06.2: Corpus Preprocessing (2-3 days)
      └─> P06.3: DAPT Training (20-30 hours GPU time)
          └─> P06.4: DAPT Evaluation
              └─> P07: NER Training
```

---

## Phase Breakdown

### Phase 6.1: Corpus Collection (CRITICAL)

**Objective:** Collect 40-50M tokens of Cambrian paleontology literature

**Timeline:** 1-2 weeks (data collection + legal review)

**Target:** 40-50M tokens (~3-5 GB cleaned text)

#### Strategy 1: Open-Access PDF Collection

**Sources:**

1. **Open-Access Journals:**
   - PaleoBios (University of California)
   - Palaeontology (Wiley) - open access articles
   - Journal of Paleontology - open access
   - Alcheringa (Taylor & Francis) - open access
   - Acta Palaeontologica Polonica

2. **Geological Survey Publications:**
   - USGS Bulletins (Cambrian sections)
   - Geological Survey of Canada
   - British Geological Survey
   - Chinese geological surveys

3. **Museum Publications:**
   - Smithsonian Contributions to Paleobiology
   - Royal Ontario Museum publications
   - Queensland Museum Memoirs (e.g., Jell & Adrain 2002)

4. **Thesis/Dissertation Repositories:**
   - ProQuest Dissertations (open access)
   - University repositories

**Collection Method:**

```python
# Automated PDF collection script
python scripts/p06_collect_pdfs.py \
  --sources config/pdf_sources.yaml \
  --output data/pdfs_raw/ \
  --min_year 1950 \
  --keywords "Cambrian,trilobite,olenellus,burgess shale" \
  --max_pdfs 500
```

**Expected Yield:**
- 300-500 PDFs
- ~30-50M tokens (after cleaning)
- Mix of monographs, papers, and bulletins

#### Strategy 2: Web Scraping (Legal Content)

**Sources:**
- GeoScienceWorld (open access articles)
- BioOne (open access)
- PaleoBios online archive

**Method:**
- Use BeautifulSoup/Scrapy for HTML extraction
- Convert HTML → plain text
- Apply P02 normalization

#### Strategy 3: Existing Corpus Augmentation

**Sources:**
- Paleobiology Database (PBDB) - specimen descriptions
- Macrostrat - formation descriptions
- Geobiodiversity Database (GBDB)

**Method:**
- API queries for Cambrian records
- Extract textual descriptions
- Aggregate into corpus

**Legal Considerations:**
- ✅ Only collect open-access or public domain materials
- ✅ Check copyright status of each source
- ✅ Store only permitted content
- ✅ Keep metadata (DOI, license, URL) for provenance

#### Deliverables

```
data/
├── pdfs_raw/               # Raw PDFs (not committed to git)
├── text_extracted/         # Extracted text files
│   ├── source_metadata.json
│   └── [pub_id].txt
├── corpus_raw/             # Pre-normalization corpus
│   └── cambrian_corpus_v1_raw.jsonl
└── corpus_norm/            # Post-normalization corpus
    └── cambrian_corpus_v1_norm.jsonl
```

**Success Criteria:**
- ✅ 40-50M tokens collected
- ✅ All sources documented with licenses
- ✅ JSONL format with pub_id/cap_id metadata
- ✅ Diverse sources (not just one journal)

---

### Phase 6.2: Corpus Preprocessing

**Objective:** Clean, normalize, and prepare corpus for DAPT training

**Timeline:** 2-3 days

**Dependencies:** Phase 6.1 complete

#### Task 2.1: Text Extraction

**Script:** `scripts/p06_extract_corpus.py`

**Process:**
1. Extract text from PDFs (PyMuPDF)
2. Clean OCR errors (ligatures, dehyphenation)
3. Split into paragraphs/captions
4. Filter noise (tables, references, headers/footers)

**Quality Filters:**
```python
# Minimum paragraph length
min_chars = 100

# Alpha character ratio
min_alpha_ratio = 0.5

# Remove duplicate paragraphs
dedup = True

# Remove boilerplate (copyright notices, etc.)
remove_boilerplate = True
```

#### Task 2.2: Normalization

**Script:** Apply P02 normalization to all text

```bash
python scripts/p06_normalize_corpus.py \
  --input data/corpus_raw/cambrian_corpus_v1_raw.jsonl \
  --output data/corpus_norm/cambrian_corpus_v1_norm.jsonl \
  --create_align_maps false
```

**Normalization Patterns:**
- `Stage 10` → `Stage_10`
- `Wheeler Formation` → `Wheeler_Formation`
- Taxonomic binomials: `Olenellus wheeleri` → `Olenellus_wheeleri`
- See P02 for full pattern list

#### Task 2.3: Data Validation

**Checks:**
```python
# 1. JSONL format validity
validate_jsonl(corpus_file)

# 2. Token count
total_tokens = count_tokens(corpus_file)
assert total_tokens >= 40_000_000, "Insufficient tokens"

# 3. Vocabulary coverage
domain_term_coverage = check_vocab_coverage(corpus_file, vocab_files)
assert domain_term_coverage >= 0.7, "Low domain coverage"

# 4. Duplicate detection
duplicate_rate = check_duplicates(corpus_file)
assert duplicate_rate < 0.05, "High duplicate rate"
```

#### Task 2.4: Train/Validation Split

**Strategy:**
- 98% training, 2% validation
- Stratify by publication to avoid data leakage
- Keep document boundaries intact

```python
python scripts/p06_split_corpus.py \
  --input data/corpus_norm/cambrian_corpus_v1_norm.jsonl \
  --train_output data/corpus_norm/train.jsonl \
  --val_output data/corpus_norm/val.jsonl \
  --split_ratio 0.98 \
  --stratify_by pub_id
```

**Output:**
```
data/corpus_norm/
├── train.jsonl           # 98% of data (39-49M tokens)
├── val.jsonl             # 2% of data (0.8-1M tokens)
└── corpus_stats.json     # Statistics
```

#### Deliverables

- ✅ Clean, normalized corpus in JSONL format
- ✅ Train/validation split
- ✅ Corpus statistics report
- ✅ Vocabulary coverage analysis

---

### Phase 6.3: DAPT Training

**Objective:** Perform Domain-Adaptive Pretraining on Cambrian corpus

**Timeline:** 20-30 hours GPU time (single RTX 2080 Ti)

**Dependencies:** Phase 6.2 complete

#### Configuration

**Hardware Requirements:**
- GPU: 1× RTX 2080 Ti (11GB VRAM) or equivalent
- RAM: 64-128 GB
- Storage: 50 GB (corpus + checkpoints)

**Training Configuration:**

```yaml
# config/dapt_config.yaml

model:
  base_model: "microsoft/deberta-v3-base"
  tokenizer: "artifacts/tokenizer_v1"
  config:
    attention_probs_dropout_prob: 0.1
    hidden_dropout_prob: 0.1

data:
  train_file: "data/corpus_norm/train.jsonl"
  val_file: "data/corpus_norm/val.jsonl"
  max_seq_length: 512
  mlm_probability: 0.15
  whole_word_masking: true

training:
  output_dir: "checkpoints/paleo-dapt-v1"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 16  # Effective batch = 128
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_steps: 10000
  lr_scheduler_type: "linear"
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4

  # Logging
  logging_steps: 100
  eval_steps: 2000
  save_steps: 10000
  save_total_limit: 5

  # Early stopping
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false

evaluation:
  eval_strategy: "steps"
  per_device_eval_batch_size: 16

seed: 42
```

#### Training Command

```bash
# Basic training
python scripts/train_dapt.py \
  --config config/dapt_config.yaml

# With custom overrides
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --learning_rate 3e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32

# Resume from checkpoint
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --resume_from_checkpoint checkpoints/paleo-dapt-v1/checkpoint-50000
```

#### Training Metrics to Monitor

**1. MLM Loss:**
- Track training and validation loss
- Target: Validation loss < 2.5 (lower is better)

**2. Perplexity:**
- Standard perplexity on validation set
- Target: Perplexity < 15

**3. Rare-Token Perplexity:**
- Perplexity measured on newly added domain tokens
- Target: 20-30% improvement vs. base model

**4. Learning Rate Schedule:**
- Monitor LR decay over time
- Ensure warmup completes smoothly

**5. Throughput:**
- Iterations per second
- Expected: 3.5-5.5 it/s on RTX 2080 Ti

#### Expected Timeline

```
Corpus size:        40M tokens
Epochs:             3
Total tokens:       120M
Batch size:         8 × 16 (GA) = 128
Seq length:         512
Tokens per batch:   128 × 512 = 65,536
Total steps:        ~100,000 steps
Time per step:      ~1 second (with GA)
Total time:         ~28 hours

Breakdown:
  Training:         ~25 hours
  Validation:       ~2 hours
  Checkpointing:    ~1 hour
```

#### Checkpointing Strategy

**Save Frequency:**
- Every 10,000 steps
- Keep top 5 checkpoints by validation loss

**Checkpoint Contents:**
```
checkpoints/paleo-dapt-v1/
├── checkpoint-10000/
│   ├── pytorch_model.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-20000/
│   └── ...
└── best/                    # Best checkpoint by val loss
    └── ...
```

#### Deliverables

- ✅ Trained DAPT model at `checkpoints/paleo-dapt-v1/best/`
- ✅ Training logs and metrics
- ✅ Validation perplexity report
- ✅ Rare-token perplexity improvement analysis

---

### Phase 6.4: DAPT Evaluation

**Objective:** Validate DAPT model performance before NER/RE fine-tuning

**Timeline:** 1-2 days

**Dependencies:** Phase 6.3 complete

#### Evaluation Tasks

**1. MLM Perplexity on Held-Out Set**

**Script:** `scripts/evaluate_dapt.py`

```bash
python scripts/evaluate_dapt.py \
  --model checkpoints/paleo-dapt-v1/best \
  --tokenizer artifacts/tokenizer_v1 \
  --eval_data data/corpus_norm/val.jsonl \
  --output reports/dapt_eval.json
```

**Metrics:**
- Overall perplexity
- Rare-token perplexity (domain vocabulary)
- Perplexity by entity type (TAXON, STRAT, CHRONO, LOC)

**Success Criteria:**
- Overall perplexity ≤ 15
- Rare-token perplexity improvement ≥ 20% vs. base DeBERTa
- Low variance across entity types

**2. Fragmentation Rate Analysis**

**Check if domain terms remain single tokens after training:**

```bash
python scripts/validate_tokenizer.py \
  --tokenizer checkpoints/paleo-dapt-v1/best \
  --vocab_dir artifacts/vocab
```

**Success Criteria:**
- Fragmentation rate ≤ 5%
- All added tokens remain in vocabulary

**3. Sample Text Generation**

**Qualitative check: Does the model "understand" domain context?**

```python
# Masked token prediction
text = "The <mask> yields abundant Olenellus from Stage_10."
# Expected predictions: Wheeler_Formation, Marjum_Formation, etc.

text = "Asaphiscus wheeleri occurs in the <mask>."
# Expected predictions: Wheeler_Formation, Marjum_Formation, etc.
```

**4. Embedding Analysis**

**Check if domain terms have meaningful embeddings:**

```python
# Nearest neighbors for domain terms
query = "Olenellus"
# Expected neighbors: Paradoxides, Elrathia, other trilobite genera

query = "Wheeler_Formation"
# Expected neighbors: Marjum_Formation, Weeks_Formation, other formations
```

#### Comparison to Base Model

**Baseline:** microsoft/deberta-v3-base (no DAPT)

| Metric | Base DeBERTa | PaleoBERT-DAPT | Improvement |
|--------|--------------|----------------|-------------|
| Overall PPL | ~18 | **<15** | **>15%** |
| Rare-token PPL | ~35 | **<28** | **>20%** |
| Fragmentation | 15% | **<5%** | **-10%** |

#### Deliverables

- ✅ DAPT evaluation report (`reports/dapt_eval.json`)
- ✅ Perplexity comparison (base vs. DAPT)
- ✅ Sample predictions showcase
- ✅ Embedding visualization (optional)
- ✅ Decision: Proceed to P07 (NER) or iterate on DAPT

---

## Alternative: Incremental Corpus Strategy (Fast Path)

**If full corpus collection is blocked or time-limited:**

### Plan B: Start with Existing + Synthetic Data

**Current Available Data:**
- Geyer 2019 PDF: ~9K tokens
- Trilobite catalog text: ~30K tokens (can be extracted from entries)
- Synthetic captions: Generate from trilobite metadata

**Synthetic Caption Generation:**

```python
# Generate 10k synthetic captions from trilobite metadata
python scripts/p06_generate_synthetic_corpus.py \
  --metadata data/trilobite_metadata.json \
  --output data/corpus_norm/synthetic_v1.jsonl \
  --num_samples 10000
```

**Expected yield:** ~200K tokens (synthetic)

**Total:** ~240K tokens (enough for minimal DAPT experiment)

**Trade-offs:**
- ✅ Fast (no corpus collection needed)
- ✅ Can start DAPT immediately
- ⚠️ Limited diversity (synthetic data bias)
- ⚠️ May not generalize well to real literature

**Use Case:** Proof-of-concept / pilot study before full corpus collection

---

## Success Metrics (P06 Overall)

### Phase 6.1: Corpus Collection

| Metric | Target | Status |
|--------|--------|--------|
| Total tokens | 40-50M | 🔴 Pending |
| Number of sources | 300-500 PDFs | 🔴 Pending |
| Geographic diversity | All major continents | 🔴 Pending |
| Temporal coverage | 1950-present | 🔴 Pending |
| Legal compliance | 100% open-access/permitted | 🔴 Pending |

### Phase 6.2: Preprocessing

| Metric | Target | Status |
|--------|--------|--------|
| Clean JSONL format | 100% valid | 🔴 Pending |
| Normalization applied | All patterns | 🔴 Pending |
| Vocabulary coverage | ≥70% | 🔴 Pending |
| Duplicate rate | <5% | 🔴 Pending |

### Phase 6.3: DAPT Training

| Metric | Target | Status |
|--------|--------|--------|
| Training completed | 3 epochs | 🔴 Pending |
| Validation loss | <2.5 | 🔴 Pending |
| Convergence | No divergence | 🔴 Pending |
| Checkpoints saved | ≥5 | 🔴 Pending |

### Phase 6.4: Evaluation

| Metric | Target | Status |
|--------|--------|--------|
| Overall perplexity | <15 | 🔴 Pending |
| Rare-token PPL improvement | ≥20% | 🔴 Pending |
| Fragmentation rate | <5% | 🔴 Pending |

---

## Timeline and Milestones

### Optimistic Timeline (Full Corpus)

```
Week 1-2:  Corpus Collection (P06.1)
  Day 1-3:   Identify sources, setup scrapers
  Day 4-10:  Download PDFs, extract text
  Day 11-14: Legal review, finalize sources

Week 3:    Corpus Preprocessing (P06.2)
  Day 15-16: Text extraction and cleaning
  Day 17:    Normalization
  Day 18-19: Validation and splitting
  Day 20-21: Buffer/QA

Week 4:    DAPT Training (P06.3)
  Day 22-23: Setup training environment
  Day 24-25: Start DAPT training (28h runtime)
  Day 26:    Monitor and checkpoint

Week 5:    DAPT Evaluation (P06.4)
  Day 27-28: Run evaluation suite
  Day 29:    Analysis and reporting
  Day 30:    Decision: Proceed to P07 or iterate

Total: 4-5 weeks
```

### Fast Path Timeline (Plan B)

```
Week 1:    Synthetic Corpus (P06.1-alt)
  Day 1-2:   Generate synthetic captions (10K)
  Day 3:     Merge with existing data (~240K tokens)
  Day 4-5:   Preprocessing and validation

Week 2:    DAPT Training (P06.3)
  Day 6-7:   Setup training (smaller corpus = faster)
  Day 8:     DAPT training (~8h for 240K tokens)
  Day 9:     Evaluation

Week 3:    Iterate or Proceed
  Day 10-11: Evaluate results
  Day 12-14: If needed: Collect more data, otherwise proceed to P07

Total: 2-3 weeks
```

---

## Recommended Approach

### Strategy: Hybrid Approach

**Phase 1 (Immediate - 1 week):**
1. ✅ Use Plan B (synthetic + existing data) for **pilot DAPT**
2. ✅ Validate pipeline end-to-end with small corpus
3. ✅ Identify any issues with training script
4. ✅ Establish baseline metrics

**Phase 2 (Parallel - 2-3 weeks):**
1. 🔄 Collect full corpus (40-50M tokens) in background
2. 🔄 Legal review and licensing
3. 🔄 Preprocess in batches as collected

**Phase 3 (After corpus ready - 1 week):**
1. ⏭️ Run full DAPT with complete corpus
2. ⏭️ Evaluate and compare to pilot
3. ⏭️ Proceed to P07 (NER training)

**Rationale:**
- Unblocks downstream work (P07, P08) while corpus collection proceeds
- Validates training pipeline early
- Reduces risk of blocked timeline
- Allows iterative improvement

---

## Dependencies and Blockers

### Current Blockers

🔴 **CRITICAL:** Insufficient corpus data
- Have: ~9K tokens (Geyer 2019)
- Need: 40-50M tokens (DAPT target)
- **Resolution:** Corpus collection (P06.1) or synthetic generation (Plan B)

### Prerequisites (Completed)

✅ P01: Tokenizer with 722 domain tokens
✅ P02: Normalization module
✅ P03: DAPT training script
✅ P04: PDF extraction pipeline
✅ P05: Vocabulary expansion and metadata

### Downstream Dependencies

⏭️ **P07:** NER training (depends on DAPT checkpoint)
⏭️ **P08:** RE training (depends on DAPT checkpoint)
⏭️ **P09:** Pipeline integration (depends on DAPT + NER + RE)

---

## Risk Assessment

### High Risk

**1. Corpus Collection Delay**
- **Risk:** Legal review takes longer than expected
- **Impact:** Delays entire timeline by 2-4 weeks
- **Mitigation:** Use Plan B (synthetic) to unblock

**2. Insufficient Corpus Diversity**
- **Risk:** Collected corpus is too narrow (e.g., only one journal)
- **Impact:** Poor generalization to diverse literature
- **Mitigation:** Explicitly target diverse sources (geography, time, topic)

**3. Training Divergence**
- **Risk:** DAPT training loss diverges or plateaus early
- **Impact:** Poor model quality, need to restart
- **Mitigation:** Start with smaller corpus (Plan B), validate hyperparameters

### Medium Risk

**4. Hardware Availability**
- **Risk:** GPU not available for 28-hour training run
- **Impact:** Training delays or needs to be split
- **Mitigation:** Reserve GPU time, use cloud if needed (Google Colab Pro, AWS)

**5. Copyright Issues**
- **Risk:** Accidentally include copyrighted material
- **Impact:** Legal liability, need to retrain
- **Mitigation:** Strict legal review, maintain source metadata

### Low Risk

**6. Technical Issues**
- **Risk:** Training script bugs, VRAM overflow
- **Impact:** Minor delays (1-2 days)
- **Mitigation:** Test with Plan B first, monitor VRAM usage

---

## Next Immediate Actions

### This Session

1. ✅ Create P06 plan document (this file)
2. ⏭️ **DECISION POINT:** Choose strategy (Full Corpus vs. Plan B)

### Recommended: Start with Plan B (Fast Path)

**Why:**
- Unblocks downstream work (P07, P08)
- Validates pipeline end-to-end
- Low risk, fast results
- Can iterate with full corpus later

**Actions (Next Session):**

1. **Generate Synthetic Corpus (Day 1):**
   ```bash
   python scripts/p06_generate_synthetic_corpus.py \
     --metadata data/trilobite_metadata.json \
     --templates config/caption_templates.txt \
     --output data/corpus_norm/synthetic_v1.jsonl \
     --num_samples 10000
   ```

2. **Merge with Existing Data (Day 1):**
   ```bash
   cat data/corpus_norm/train_geyer2019.jsonl \
       data/corpus_norm/synthetic_v1.jsonl \
       > data/corpus_norm/train_pilot.jsonl
   ```

3. **Validate Corpus (Day 1):**
   ```bash
   python scripts/validate_corpus.py \
     --input data/corpus_norm/train_pilot.jsonl
   ```

4. **Run Pilot DAPT (Day 2-3):**
   ```bash
   python scripts/train_dapt.py \
     --config config/dapt_pilot_config.yaml \
     --output_dir checkpoints/paleo-dapt-pilot
   ```

5. **Evaluate Pilot (Day 4):**
   ```bash
   python scripts/evaluate_dapt.py \
     --model checkpoints/paleo-dapt-pilot/best \
     --eval_data data/corpus_norm/val.jsonl
   ```

6. **Proceed to P07 or Iterate (Day 5):**
   - If pilot results good: ✅ Start P07 (NER)
   - If pilot results poor: 🔄 Collect more corpus, iterate

---

## References

### Related Documents

- **OVERVIEW.md** - Overall training design spec
- **VOCABULARY_EXPANSION_PLAN.md** - Vocabulary expansion strategy (completed in P05)
- **P03:** `devlog/20251029_P03_dapt_training_script.md` - DAPT script implementation
- **P05:** `devlog/20251029_006_P05_trilobite_catalog_execution_complete.md` - Vocabulary expansion results

### External Resources

- **HuggingFace Transformers:** https://huggingface.co/docs/transformers/
- **DeBERTa Paper:** https://arxiv.org/abs/2006.03654
- **Domain-Adaptive Pretraining:** Gururangan et al. (2020) - "Don't Stop Pretraining"
- **Paleobiology Database API:** https://paleobiodb.org/data1.2/

---

## Appendix: Script Templates

### A. Synthetic Corpus Generation Script

**File:** `scripts/p06_generate_synthetic_corpus.py`

**Purpose:** Generate synthetic captions from trilobite metadata

**Input:**
- `data/trilobite_metadata.json`
- `config/caption_templates.txt`

**Output:**
- `data/corpus_norm/synthetic_v1.jsonl`

**Templates:**
```
{taxon} occurs in the {formation} at {locality}.
The {formation} yields abundant {taxon} from {chrono}.
{taxon} from {locality} is characteristic of {chrono}.
Specimens of {taxon} were collected from the {formation}.
```

### B. Corpus Validation Script

**File:** `scripts/validate_corpus.py`

**Purpose:** Validate JSONL corpus format and statistics

**Checks:**
- JSONL format validity
- Token count
- Vocabulary coverage
- Duplicate detection
- Metadata completeness

### C. Pilot DAPT Config

**File:** `config/dapt_pilot_config.yaml`

**Differences from full config:**
```yaml
data:
  train_file: "data/corpus_norm/train_pilot.jsonl"  # Smaller corpus

training:
  num_train_epochs: 5  # More epochs for smaller data
  per_device_train_batch_size: 16  # Larger batch for stability
  gradient_accumulation_steps: 8  # Effective batch = 128
  save_steps: 500  # More frequent saves
  eval_steps: 500
```

---

**Status:** 📋 PLANNING COMPLETE

**Next Milestone:** P06 Execution (Corpus Collection or Pilot DAPT)

**Critical Decision:** Choose Fast Path (Plan B) or Full Corpus Path

**Recommendation:** ✅ **Start with Plan B** to unblock downstream work, then iterate with full corpus

**Date:** 2025-10-29
