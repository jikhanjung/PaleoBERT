# PaleoBERT Training Scripts

This directory contains training scripts for the PaleoBERT-Cambrian v1.0 model.

## Available Scripts

### 1. `train_dapt.py` - Domain-Adaptive Pretraining

Performs Masked Language Modeling (MLM) on Cambrian paleontology literature to adapt DeBERTa-v3-base to the domain.

**Status:** ✅ Phase 1 & 2 Complete (Core training + validation metrics implemented)

**Prerequisites:**
- P01: Extended tokenizer at `artifacts/tokenizer_v1/` (run `build_tokenizer.py` first)
- P02: Normalized corpus in JSONL format
- 11GB VRAM GPU (RTX 2080 Ti or equivalent)
- Dependencies installed: `pip install -r requirements.txt`

**Usage:**

```bash
# Basic training with default configuration
python scripts/train_dapt.py --config config/dapt_config.yaml

# Override configuration values
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --max_steps 50000

# Resume from checkpoint
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --resume_from_checkpoint checkpoints/paleo-dapt-v1/checkpoint-10000

# Quick test with sample data (10 steps)
python scripts/train_dapt.py \
  --config config/dapt_config_test.yaml \
  --max_steps 10
```

**Configuration Files:**
- `config/dapt_config.yaml` - Production configuration (40-50M tokens)
- `config/dapt_config_test.yaml` - Testing configuration (sample data)

**Key Features:**
- ✅ JSONL corpus loader with normalized text support
- ✅ MLM data collator with whole-word masking
- ✅ 11GB VRAM optimization (fp16 + gradient checkpointing)
- ✅ Checkpoint save/resume support
- ✅ TensorBoard logging
- ✅ **Domain-specific metrics** - Phase 2 ✨ NEW
  - Rare-token perplexity (measures domain vocabulary learning)
  - Fragmentation rate (tracks tokenization quality)
- ✅ **Early stopping** - Phase 2 ✨ NEW
  - Configurable patience and improvement threshold
  - Monitors eval_loss or rare-token PPL
- ✅ **Custom evaluation callbacks** - Phase 2 ✨ NEW
  - Automated domain-specific validation
  - TensorBoard integration

**Expected Performance:**
- Throughput: 3.5-5.5 it/s @ seq512, batch=8, GA=16
- Training time: 20-30 hours for 40-50M tokens
- VRAM usage: ~10GB with fp16 enabled

**Output:**
- Checkpoints saved to `checkpoints/paleo-dapt-v1/`
- TensorBoard logs in `checkpoints/paleo-dapt-v1/logs/`
- Best model automatically selected based on eval_loss

---

### 2. `build_tokenizer.py` - Build Extended Tokenizer (P01)

Builds extended DeBERTa tokenizer with Cambrian domain vocabulary.

**Status:** ✅ Complete (requires user execution)

**Usage:**
```bash
python scripts/build_tokenizer.py
```

See P01 devlog for details.

---

### 3. `validate_tokenizer.py` - Validate Tokenizer (P01)

Validates tokenizer fragmentation rates.

**Status:** ✅ Complete

**Usage:**
```bash
python scripts/validate_tokenizer.py
```

---

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PaleoBERT Training Pipeline                  │
└─────────────────────────────────────────────────────────────────┘

P01: Build Tokenizer
  └─> artifacts/tokenizer_v1/

P02: Normalize Corpus
  └─> data/corpus_norm/*.jsonl

P03: DAPT Training ← YOU ARE HERE
  └─> checkpoints/paleo-dapt-v1/best.pt

P04: NER Training (Coming Soon)
  └─> checkpoints/paleo-ner-v1/best.pt

P05: RE Training (Coming Soon)
  └─> checkpoints/paleo-re-v1/best.pt

P06: Inference Pipeline (Coming Soon)
  └─> Final JSON output
```

---

## Troubleshooting

### VRAM Out of Memory

If you encounter VRAM OOM errors:

1. **Reduce batch size:**
   ```bash
   python scripts/train_dapt.py \
     --config config/dapt_config.yaml \
     --batch_size 4
   ```

2. **Increase gradient accumulation to maintain effective batch size:**
   ```yaml
   # In config/dapt_config.yaml
   per_device_train_batch_size: 4  # Reduced from 8
   gradient_accumulation_steps: 32  # Increased from 16
   # Effective batch size still 128 (4 * 32)
   ```

3. **Reduce sequence length:**
   ```yaml
   max_seq_length: 384  # Reduced from 512
   ```

4. **Disable gradient checkpointing (increases VRAM usage but faster):**
   ```yaml
   gradient_checkpointing: false
   ```

### Slow Data Loading

If training is slow due to data loading:

1. **Increase number of workers:**
   ```yaml
   preprocessing_num_workers: 16  # Increased from 8
   ```

2. **Use faster storage (NVMe SSD)**

3. **Cache tokenized examples** (feature coming in Phase 2)

### Tokenizer Not Found

```
FileNotFoundError: Tokenizer not found at artifacts/tokenizer_v1/
```

**Solution:** Run P01 tokenizer build script first:
```bash
python scripts/build_tokenizer.py
```

### Corpus Not Found

```
ValueError: No examples loaded! Check your data_files paths.
```

**Solution:** Ensure corpus files exist at the specified paths in config:
```bash
ls data/corpus_norm/train_*.jsonl
ls data/corpus_norm/eval_*.jsonl
```

For testing, use sample data:
```bash
python scripts/train_dapt.py --config config/dapt_config_test.yaml
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/paleo-dapt-v1/logs
```

Open browser to `http://localhost:6006`

### Key Metrics to Watch

1. **train_loss**: Should decrease steadily
2. **eval_loss**: Should track train_loss (not diverge)
3. **learning_rate**: Should follow warmup → linear decay schedule
4. **grad_norm**: Should stay < 1.0 (with max_grad_norm=1.0)
5. **samples/second**: Should be stable around 3.5-5.5 it/s

### Red Flags

- ❌ **train_loss not decreasing**: Check learning rate
- ❌ **eval_loss >> train_loss**: Overfitting (reduce training steps)
- ❌ **samples/second < 2**: Data loading bottleneck
- ❌ **VRAM OOM**: Reduce batch size or seq_length

---

## Development Status

### Phase 1: Core Training Loop ✅ COMPLETE

- ✅ Dataset loader (CambrianCorpusDataset)
- ✅ MLM data collator (CambrianMLMCollator)
- ✅ Model initialization
- ✅ Training loop with HuggingFace Trainer
- ✅ Configuration files (YAML)
- ✅ Sample test corpus
- ✅ Documentation

### Phase 2: Validation & Metrics ✅ COMPLETE

- ✅ Rare-token perplexity metric (RareTokenMetrics class)
- ✅ Custom evaluation callback (DAPTEvaluationCallback)
- ✅ Early stopping logic (DAPTEarlyStoppingCallback)
- ✅ Fragmentation rate tracking
- ✅ Integration with main training loop
- ✅ Test suite (tests/test_dapt_phase2.py)

### Phase 3: Testing & Refinement (Coming Soon)

- ⚠️ Unit tests
- ⚠️ Integration tests
- ⚠️ Performance profiling
- ⚠️ Additional documentation

---

## References

- **OVERVIEW.md**: Complete training specification
- **devlog/20251029_P03_dapt_training_script.md**: Implementation plan
- **DeBERTa Paper**: He et al., 2021 (ICLR)
- **DAPT Paper**: Gururangan et al., 2020 (ACL)

---

## Contact

For questions or issues, please refer to:
- Project documentation in `docs/`
- Devlog in `devlog/`
- CLAUDE.md for AI assistant guidance
