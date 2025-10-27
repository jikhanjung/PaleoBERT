# PaleoBERT (DeBERTa‑DAPT) — Training Design Spec

**Scope:** End‑to‑end plan to build **PaleoBERT** using **DeBERTa‑v3‑base** as the backbone with **Domain‑Adaptive Pretraining (DAPT)**, followed by **NER** and **RE** fine‑tuning, and culminating in an integrated release.  
**Audience:** ML engineer / researcher operating on a single‑GPU workstation (e.g., RTX 2080 Ti, 11 GB).  
**Output artifacts:** DeBERTa tokenizer w/ added tokens, DAPT checkpoint, NER/RE heads, inference pipeline, evaluation reports, and a versioned release.

---

## 0) Assumptions & Goals

- **Primary goal:** Extract structured paleontology facts from captions/paragraphs: entities (**taxa, strat units, chrono units, localities**) and relations (**taxon→strat**, **taxon→locality**, **strat→chrono**, etc.).
- **Hardware baseline:** 1× **RTX 2080 Ti (11 GB VRAM)**, 64–128 GB RAM, fast NVMe.
- **Backbone:** `microsoft/deberta-v3-base` (Hugging Face ID).
- **Tokenizer:** fast tokenizer; **added tokens** for domain vocabulary (taxa/strat/chrono/localities).
- **Text normalization:** internal normalized view (e.g., `Stage_10`, `Series_2`), with **align maps** to original raw text for round‑trip recovery.
- **Reproducibility:** Seeded runs, DVC/Git‑LFS for checkpoints & corpora manifests, explicit versioning.

---

## 1) Data Design

### 1.1 Corpora (DAPT)
- **Sources (examples):**
  - Open‑access paleontology papers (captions + methods + locality/strat sections).
  - Geological bulletins/reports (formation/member descriptions).
  - Museum specimen notes (where legally permitted).
- **Target size:** **~100M tokens** (≈ 8–12 GB cleaned text). Scales to 200M if time allows.
- **Segmentation:** sentence‑ or caption‑level; keep **document boundaries** (for shuffling) and **pub_id/cap_id** metadata.
- **Cleaning:**
  - OCR fixes (ligatures, dehyphenation at line breaks).
  - Unicode normalization (`NFKC`), collapse multiple spaces.
  - Strip references/tables when noisy; retain figure/caption blocks.
- **Normalization (optional but recommended):**
  - `Stage 10 / Stage-10 / Stage X` → `Stage_10` (Roman→Arabic).
  - `Series 2` → `Series_2`.
  - `Cambrian Stage 10` → `Cambrian_Stage_10` (optionally).
  - Keep **raw_text** + **norm_text** + **align_map** (char‑level index mapping).

### 1.2 Added Tokens
- Build lists: `taxa.txt`, `strat_units.txt`, `chrono_units.txt`, `localities.txt`.
- **Examples:** `Asaphiscus`, `Olenellus`, `Paibian`, `Jiangshanian`, `Shackleton`, `Limestone`, `Formation`, `Dyer`, `Member`, `Holyoake`, `Transantarctic`, `Series_2`, `Stage_10`, etc.
- Add via `tokenizer.add_tokens(list)`; then `model.resize_token_embeddings(len(tokenizer))` before DAPT.

### 1.3 Labeled Data (Fine‑tuning)
- **NER**: 5k–20k sentences with BIO tags for `TAXON`, `STRAT`, `CHRONO`, `LOC`.
  - Bootstrap from rules + manual corrections (active learning on low‑confidence spans).
- **RE**: 5k–20k **entity‑paired** samples from NER outputs, with labels:
  - `occurs_in (TAXON→STRAT)`, `found_at (TAXON→LOC)`, `part_of (STRAT→STRAT)`, `assigned_to (STRAT→CHRONO)`.
  - Negative sampling: random unlinked pairs; hard negatives from near‑miss contexts.

---

## 2) Tokenizer & Pre‑training Config

### 2.1 Tokenizer
```python
from transformers import AutoTokenizer, AddedToken
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)

custom_tokens = taxa + strat_units + chrono_units + localities  # lists of strings
tok.add_tokens(custom_tokens)  # returns number of tokens added
tok.save_pretrained("artifacts/tokenizer_v1/")
```

### 2.2 DAPT Objective & Setup
- **Objective:** Masked Language Modeling (MLM).  
- **Masking:** 15% token mask; whole‑word‑masking enabled for added tokens where possible.
- **Batching (2080 Ti):**
  - `max_seq_length=512`, `per_device_train_batch_size=8`.
  - `gradient_accumulation_steps=16` (effective batch=128).
  - `fp16=True` (Apex or native amp).
  - Gradient checkpointing **ON**.
- **Optimization:**
  - `AdamW` / `AdamW Torch` with `lr=2e-4`, `betas=(0.9, 0.98)`, `weight_decay=0.01`.
  - Learning rate schedule: **linear warmup** 10k steps → linear decay.
- **Training length:**
  - Tokens: **100M** (3–4 epochs over corpus shards).
  - Steps: **~100k** (log every 100, eval every 2k, save every 10k).
- **Throughput expectation (2080 Ti):** ~3.5–5.5 it/s at seq512 + GA; wall‑clock **20–30 h**.
- **Checkpointing:**
  - Save: model, optimizer, LR scheduler, tokenizer snapshot(`tokenizer_v1`).
  - Keep top‑K by **validation MLM loss** & **rare‑token perplexity** (see below).

### 2.3 DAPT Validation
- **Metrics:**
  - Standard **MLM loss**/**perplexity** on held‑out domain set.
  - **Rare‑token PPL** for newly added tokens (masked on occurrences of added tokens).
  - **Fragmentation rate** (pre vs post): % of domain terms split into >1 subword.
- **Early stopping:** plateau on MLM loss for 5 evals or rare‑token PPL improvement < 1% for 3 evals.

---

## 3) NER Fine‑tuning

### 3.1 Task & Labels
- BIO tagging with entity types: `TAXON`, `STRAT`, `CHRONO`, `LOC`.
- Use **normalized text** during training; project spans back to **raw text** via align maps for evaluation and storage.

### 3.2 Model
- DeBERTa‑v3‑base (from **best DAPT ckpt**) + token‑classification head.
- Class weights optional for class imbalance.

### 3.3 Hyperparameters (2080 Ti)
- `max_seq_length=256/384/512` (grid).
- `batch_size=16` (GA to fit VRAM), `fp16=True`.
- `lr=2e-5` (grid: 1e‑5, 2e‑5, 3e‑5), `epochs=5–10` with early stopping (dev F1).
- Dropout 0.1 on classifier; weight_decay 0.01.

### 3.4 Data Strategy
- **Active learning:** pick low‑confidence spans from rule‑based model to annotate first.
- **Augment:** minor orthographic variants (hyphen/space, roman↔arabic numerals) in normalized view.
- **Stratified split:** by **publication** (to avoid leakage), e.g., 80/10/10 train/dev/test.

### 3.5 Evaluation
- **Metrics:** micro/macro **F1** per entity type; **span‑level** exact match.
- **Robustness:** evaluate on **raw text** spans after projection; track JSON validity rate (schema‑conformant outputs).

---

## 4) Relation Extraction (RE) Fine‑tuning

### 4.1 Task Setup
- Build candidate pairs from NER outputs within sentence/caption windows.
- **Labels:** multi‑class (or multi‑label) among `{occurs_in, found_at, part_of, assigned_to, NONE}`.

### 4.2 Model Variants
1) **CLS classifier** over `[CLS]` of the sentence + entity position embeddings.  
2) **Span‑pair classifier**: concatenate span representations (start/end pooling) → MLP.  
3) (Option) **Biaffine** scorer for pairwise relation scores.

### 4.3 Hyperparameters
- `max_seq_length=256/384`, `batch_size=16` (GA), `lr=1e-5~2e-5`, `epochs=5–10`.
- Negative sampling ratio 1:2~1:4; incorporate **hard negatives** from near‑miss contexts.
- Class‑balanced focal loss (optional) if positives are sparse.

### 4.4 Evaluation
- **Metrics:** Precision/Recall/F1 per relation; **micro‑F1** overall.
- **Ablations:** sentence‑level vs caption‑level; with/without added tokens; with/without DAPT.
- **Error analysis:** confusion between `occurs_in` vs `found_at`, and `part_of` hierarchy errors.

---

## 5) Integration & Inference

### 5.1 Pipeline
1) Preprocess (normalize text + align map).
2) **NER** on normalized text → spans & labels.
3) Project spans back to raw text via align map.
4) **Linking/Normalization**: map surface forms to canonical IDs (taxon, strat, chrono, loc).
5) **RE** over candidate pairs; produce triples with confidences.
6) **JSON** assembly (mentions, triples, evidence) following the PaleoAI contract.

### 5.2 Throughput (2080 Ti)
- NER (seq384, bs=32 effective): ~120–180 docs/min (caption‑sized).
- RE: ~200–300 candidate pairs/s (depending on windowing).
- Full pipeline (caption‑centric): ~1k–2k captions/min offline with batching.

### 5.3 Confidence & Provenance
- Retain per‑entity and per‑triple **confidence** scores.
- Store **provenance**: `pub_id`, `cap_id`, `char_offsets(raw/norm)`, snippet text.

---

## 6) Versioning & Release

### 6.1 Version IDs
- `paleo-deberta-tokenizer-v1`
- `paleo-deberta-dapt-v1` (ckpt SHA & DVC hash)
- `paleo-ner-v1`, `paleo-re-v1`
- `paleo-pipeline-v1`

### 6.2 Model Card (template)
- **Intended use:** paleontology entity/relationship extraction.
- **Training data:** overview + licenses (no raw copyrighted text stored).
- **Added tokens:** counts per category (no proprietary lists unless permitted).
- **Limitations:** OCR noise sensitivity; ambiguous locality disambiguation.
- **Ethics & Safety:** academic research context, not medical or legal advice.

### 6.3 Deliverables
- `artifacts/tokenizer_v1/` (added tokens + config)
- `checkpoints/paleo-dapt-v1/` (best .bin + optimizer states)
- `checkpoints/paleo-ner-v1/`, `checkpoints/paleo-re-v1/`
- `scripts/` (train_dapt.py, train_ner.py, train_re.py, infer_pipeline.py)
- `reports/` (mlm_eval.md, ner_eval.json, re_eval.json, ablation.md)
- `README.md`, `MODEL_CARD.md`, `LICENSE`

---

## 7) Validation Plan (Milestones)

| Milestone | Gate Criteria |
|-----------|----------------|
| **M1: DAPT complete** | Held‑out MLM ppl ≤ baseline−10%; rare‑token ppl ≤ baseline−20% |
| **M2: NER baseline** | F1(TAXON) ≥ 0.90; F1(STRAT) ≥ 0.80; F1(CHRONO) ≥ 0.80; F1(LOC) ≥ 0.80 |
| **M3: RE baseline** | micro‑F1 ≥ 0.75; `occurs_in` F1 ≥ 0.80 |
| **M4: End‑to‑end** | JSON validity ≥ 98%; triple‑validity@1 ≥ 0.75; provenance match ≥ 0.95 |
| **M5: Release** | Reproducible run scripts; model card; checksum; fixed seeds |

---

## 8) Risks & Mitigations

- **Data licensing:** Keep only IDs/snippets allowed; store document metadata, not full text when restricted.
- **OCR noise:** Use cleanup filters; track an OCR‑quality flag to exclude worst pages from DAPT.
- **Imbalanced labels:** Class‑weighted loss and focal loss; targeted sampling.
- **Catastrophic forgetting:** Lower LR for DAPT; monitor general English MLM loss if needed.
- **Tokenizer drift:** Freeze `tokenizer_v1` post‑DAPT; any future tokens → `tokenizer_v2` with compatibility notes.

---

## 9) Example CLI Interfaces

```bash
# 1) DAPT
python scripts/train_dapt.py \
  --model microsoft/deberta-v3-base \
  --tokenizer artifacts/tokenizer_v1 \
  --data data/corpus_norm/*.jsonl \
  --seq_len 512 --batch 8 --grad_accum 16 \
  --lr 2e-4 --epochs 3 --fp16 --grad_checkpoint \
  --save_dir checkpoints/paleo-dapt-v1

# 2) NER
python scripts/train_ner.py \
  --model checkpoints/paleo-dapt-v1/best.pt \
  --data data/ner/*.jsonl --seq_len 384 \
  --lr 2e-5 --epochs 8 --batch 16 --fp16 \
  --save_dir checkpoints/paleo-ner-v1

# 3) RE
python scripts/train_re.py \
  --model checkpoints/paleo-dapt-v1/best.pt \
  --data data/re/*.jsonl --seq_len 384 \
  --lr 1e-5 --epochs 8 --batch 16 --fp16 \
  --save_dir checkpoints/paleo-re-v1

# 4) Inference
python scripts/infer_pipeline.py \
  --model-ner checkpoints/paleo-ner-v1/best.pt \
  --model-re  checkpoints/paleo-re-v1/best.pt \
  --tokenizer artifacts/tokenizer_v1 \
  --in data/new_docs/*.jsonl \
  --out outputs/jsonl/
```

---

## 10) Appendix: Added‑Token Tips

- Use **underscores** to bind multi‑token units (`Series_2`, `Stage_10`, `Cambrian_Stage_10`).
- Keep a **two‑way map** for raw↔normalized strings; store **align maps** for span projections.
- Limit token explosion: enumerate realistic ranges (e.g., `Stage_1..Stage_10`, `Series_2..Series_4`).
- Evaluate **fragmentation** pre/post to confirm benefit.
- Re‑run DAPT if added‑token list significantly grows → bump tokenizer version (`v2`).


