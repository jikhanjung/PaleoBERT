# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaleoBERT is a domain-adaptive NLP system for extracting structured paleontology information from scientific text. It uses DeBERTa-v3-base with Domain-Adaptive Pretraining (DAPT), followed by Named Entity Recognition (NER) and Relation Extraction (RE) fine-tuning.

**Target entities:** TAXON (organisms), STRAT (stratigraphic units), CHRONO (chronostratigraphic units), LOC (localities)
**Target relations:** occurs_in (taxon→strat), found_at (taxon→loc), part_of (strat→strat), assigned_to (strat→chrono)

**Hardware constraint:** Single GPU (RTX 2080 Ti, 11GB VRAM)

## Training Commands

### 1. Domain-Adaptive Pretraining (DAPT)
```bash
python scripts/train_dapt.py \
  --model microsoft/deberta-v3-base \
  --tokenizer artifacts/tokenizer_v1 \
  --data data/corpus_norm/*.jsonl \
  --seq_len 512 --batch 8 --grad_accum 16 \
  --lr 2e-4 --epochs 3 --fp16 --grad_checkpoint \
  --save_dir checkpoints/paleo-dapt-v1
```

### 2. NER Training
```bash
python scripts/train_ner.py \
  --model checkpoints/paleo-dapt-v1/best.pt \
  --data data/ner/*.jsonl --seq_len 384 \
  --lr 2e-5 --epochs 8 --batch 16 --fp16 \
  --save_dir checkpoints/paleo-ner-v1
```

### 3. RE Training
```bash
python scripts/train_re.py \
  --model checkpoints/paleo-dapt-v1/best.pt \
  --data data/re/*.jsonl --seq_len 384 \
  --lr 1e-5 --epochs 8 --batch 16 --fp16 \
  --save_dir checkpoints/paleo-re-v1
```

### 4. Inference Pipeline
```bash
python scripts/infer_pipeline.py \
  --model-ner checkpoints/paleo-ner-v1/best.pt \
  --model-re checkpoints/paleo-re-v1/best.pt \
  --tokenizer artifacts/tokenizer_v1 \
  --in data/new_docs/*.jsonl \
  --out outputs/jsonl/
```

## Architecture

### Text Normalization & Alignment
The system maintains two parallel text representations:
- **raw_text**: Original input text
- **norm_text**: Normalized view with domain-specific transformations (e.g., `Stage 10` → `Stage_10`, `Series 2` → `Series_2`)
- **align_map**: Character-level index mapping between raw and normalized text for round-trip span projection

Training and inference operate on normalized text, but final outputs are projected back to raw text offsets.

### Tokenizer with Added Tokens
Custom tokens are added to DeBERTa's tokenizer from four categories:
- `taxa.txt`: Taxonomic names (e.g., Asaphiscus, Olenellus)
- `strat_units.txt`: Stratigraphic units (e.g., Formation, Member, Limestone)
- `chrono_units.txt`: Chronostratigraphic units (e.g., Paibian, Jiangshanian, Series_2, Stage_10)
- `localities.txt`: Geographic localities

Use underscores to bind multi-token units (e.g., `Cambrian_Stage_10`). After adding tokens, always call `model.resize_token_embeddings(len(tokenizer))` before training.

### Pipeline Architecture
1. **Preprocess**: Normalize text and create align_map
2. **NER**: Extract entity spans on norm_text
3. **Span Projection**: Map entity spans back to raw_text via align_map
4. **Entity Linking**: Normalize surface forms to canonical IDs
5. **RE**: Extract relations from candidate entity pairs
6. **JSON Assembly**: Produce structured output with mentions, triples, confidence scores, and provenance

### VRAM Management for 11GB GPU
- Use `fp16=True` for all training
- Enable gradient checkpointing via `--grad_checkpoint`
- DAPT: seq_len=512, batch=8, grad_accum=16 (effective batch=128)
- NER/RE: seq_len=384, batch=16 with gradient accumulation as needed
- Monitor VRAM with `nvidia-smi` during training

## Training Validation Criteria

### DAPT (M1)
- Held-out MLM perplexity: ≤ baseline - 10%
- Rare-token perplexity: ≤ baseline - 20%
- Track fragmentation rate for added tokens

### NER (M2)
- F1(TAXON) ≥ 0.90
- F1(STRAT) ≥ 0.80
- F1(CHRONO) ≥ 0.80
- F1(LOC) ≥ 0.80
- Span-level exact match on raw text

### RE (M3)
- micro-F1 ≥ 0.75
- occurs_in F1 ≥ 0.80

### End-to-End Pipeline (M4)
- JSON schema validity ≥ 98%
- Triple validity@1 ≥ 0.75
- Provenance character offset match ≥ 0.95

## Data Considerations

### Corpus Preparation (DAPT)
- Target: ~100M tokens (8-12 GB cleaned text)
- Sources: Open-access paleontology papers, geological bulletins, museum notes (where permitted)
- Keep document boundaries and metadata (pub_id, cap_id) for proper shuffling
- OCR cleanup: ligatures, dehyphenation, NFKC normalization

### Labeled Data (Fine-tuning)
- NER: 5k-20k sentences with BIO tags
- RE: 5k-20k entity-paired samples with negative sampling (1:2 to 1:4 ratio)
- Stratified split by publication to avoid leakage (80/10/10)
- Use active learning on low-confidence spans for efficient annotation

## Versioning

Models and artifacts use explicit version IDs:
- `paleo-deberta-tokenizer-v1`
- `paleo-deberta-dapt-v1`
- `paleo-ner-v1`, `paleo-re-v1`
- `paleo-pipeline-v1`

Tokenizer versioning is frozen post-DAPT. Significant changes to added tokens require a new tokenizer version (v2) with compatibility notes.

## Key Risks

- **Data licensing**: Store only permitted IDs/snippets; avoid full copyrighted text
- **OCR noise**: Track quality flags and exclude worst pages from DAPT
- **Catastrophic forgetting**: Use lower LR for DAPT; monitor general English MLM if needed
- **Tokenizer drift**: Freeze tokenizer_v1 after DAPT; document any future changes
