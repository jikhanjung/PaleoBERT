# P03 DAPT Training Script - Implementation Plan

**Date:** 2025-10-29
**Milestone:** P03 (Domain-Adaptive Pretraining)
**Status:** 📋 PLANNING
**Priority:** HIGH (Critical path blocker)
**Estimated Duration:** 2-3 days implementation

---

## Executive Summary

Implement the Domain-Adaptive Pretraining (DAPT) training script for PaleoBERT-Cambrian v1.0. This script will perform Masked Language Modeling (MLM) on Cambrian-specific paleontology literature to adapt DeBERTa-v3-base to the domain vocabulary and linguistic patterns.

**Critical Success Factor:** Must fit within 11GB VRAM constraint (RTX 2080 Ti) while maintaining training efficiency.

---

## Objectives

### Primary Goal
Create a production-ready DAPT training script that:
1. Loads normalized Cambrian corpus (from P02)
2. Performs MLM training on DeBERTa-v3-base with extended tokenizer
3. Validates domain adaptation via MLM perplexity and rare-token metrics
4. Saves checkpoints for downstream NER/RE fine-tuning

### Success Criteria
- ✅ Script runs within 11GB VRAM constraint
- ✅ Processes 100M tokens in ~20-30 hours (3.5-5.5 it/s)
- ✅ Held-out MLM perplexity: ≤ baseline - 10%
- ✅ Rare-token perplexity: ≤ baseline - 20%
- ✅ Checkpointing and resumption working correctly
- ✅ Validation metrics tracked (TensorBoard/WandB)

---

## Technical Requirements (from OVERVIEW.md)

### Hardware Constraints
- **GPU:** RTX 2080 Ti (11GB VRAM)
- **RAM:** 64-128GB system memory
- **Storage:** NVMe for fast I/O

### Corpus Requirements
- **Size:** 40-50M tokens (3-5 GB cleaned text)
- **Focus:** Cambrian Period literature (541-485 Ma)
- **Format:** JSONL with normalized text + metadata
- **Preprocessing:** Applied via P02 normalization module

### Model Configuration
- **Base Model:** `microsoft/deberta-v3-base` (184M params)
- **Tokenizer:** Extended with ~400 Cambrian domain tokens
- **Embeddings:** Resized to accommodate added tokens

### Training Hyperparameters (11GB VRAM Optimized)

```python
TRAINING_CONFIG = {
    # Batch configuration
    "max_seq_length": 512,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 16,  # Effective batch = 128
    "per_device_eval_batch_size": 16,

    # Memory optimization
    "fp16": True,  # Mixed precision training
    "gradient_checkpointing": True,

    # Optimization
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-6,
    "max_grad_norm": 1.0,

    # Learning rate schedule
    "lr_scheduler_type": "linear",
    "warmup_steps": 10000,

    # Training length
    "num_train_epochs": 3,  # 3-4 epochs over corpus
    "max_steps": 100000,  # ~100k steps expected

    # Logging & checkpointing
    "logging_steps": 100,
    "eval_steps": 2000,
    "save_steps": 10000,
    "save_total_limit": 3,  # Keep top-3 checkpoints

    # MLM configuration
    "mlm_probability": 0.15,  # 15% masking rate
    "whole_word_masking": True,  # For multi-token terms
}
```

### Validation Metrics

1. **Standard MLM Loss/Perplexity**
   - Measured on held-out Cambrian corpus
   - Target: ≤ baseline - 10%

2. **Rare-Token Perplexity**
   - Focused on newly added domain tokens
   - Target: ≤ baseline - 20%

3. **Fragmentation Rate**
   - % of domain terms split into >1 subword
   - Track pre-DAPT vs post-DAPT

4. **Throughput Monitoring**
   - Iterations/second
   - Samples/second
   - GPU utilization

---

## Implementation Plan

### Phase 1: Core Training Loop (Day 1, 6-8 hours)

#### 1.1 Dataset Preparation Module

**File:** `scripts/train_dapt.py` (DataLoader section)

**Tasks:**
- Implement JSONL corpus reader
- Handle normalized text extraction
- Tokenization with padding/truncation
- Dynamic batching for efficiency
- Document boundary preservation

**Code Structure:**
```python
class CambrianCorpusDataset(Dataset):
    """
    Load Cambrian corpus from JSONL files.

    Format:
        {"doc_id": "...", "pub_id": "...", "cap_id": "...",
         "raw_text": "...", "norm_text": "...", "align_map": {...}}
    """
    def __init__(self, data_files, tokenizer, max_length=512):
        self.examples = []
        for file in data_files:
            self.examples.extend(self._load_jsonl(file))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_jsonl(self, filepath):
        """Load and parse JSONL corpus."""
        pass

    def __getitem__(self, idx):
        """Return tokenized example."""
        pass
```

**Validation:**
- Test with sample JSONL files
- Verify tokenization preserves domain terms
- Check memory footprint

#### 1.2 MLM Data Collator

**Tasks:**
- Implement whole-word masking
- 15% masking rate
- Special handling for domain tokens
- Prepare labels for MLM objective

**Code Structure:**
```python
class CambrianMLMCollator(DataCollatorForLanguageModeling):
    """
    MLM collator with whole-word masking for domain terms.
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
        self.domain_token_ids = self._get_domain_token_ids()

    def _get_domain_token_ids(self):
        """Identify added domain tokens for special masking."""
        pass

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """Apply whole-word masking with domain token awareness."""
        pass
```

#### 1.3 Model Initialization

**Tasks:**
- Load `microsoft/deberta-v3-base`
- Load extended tokenizer from `artifacts/tokenizer_v1/`
- Resize token embeddings
- Enable gradient checkpointing
- Move to GPU with fp16

**Code Structure:**
```python
def initialize_model_and_tokenizer(args):
    """
    Load base model and extended tokenizer.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer
```

#### 1.4 Training Loop Skeleton

**Tasks:**
- HuggingFace Trainer configuration
- Training arguments setup
- Optimizer and scheduler initialization
- Basic training loop

**Code Structure:**
```python
def main(args):
    # Initialize
    model, tokenizer = initialize_model_and_tokenizer(args)
    train_dataset = CambrianCorpusDataset(args.train_files, tokenizer)
    eval_dataset = CambrianCorpusDataset(args.eval_files, tokenizer)
    data_collator = CambrianMLMCollator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # ... (full config from TRAINING_CONFIG)
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save final checkpoint
    trainer.save_model(args.output_dir)
```

**Day 1 Deliverables:**
- ✅ Basic training script executable
- ✅ Dataset loading working
- ✅ MLM training loop functional
- ✅ Can run on sample data (even if not converging yet)

---

### Phase 2: Validation & Metrics (Day 2, 6-8 hours)

#### 2.1 Rare-Token Perplexity Metric

**File:** `scripts/train_dapt.py` (Metrics section)

**Tasks:**
- Implement rare-token perplexity calculation
- Track perplexity for newly added tokens only
- Compare vs baseline DeBERTa perplexity

**Code Structure:**
```python
class RareTokenMetrics:
    """
    Calculate perplexity for domain-specific added tokens.
    """
    def __init__(self, tokenizer, domain_tokens):
        self.tokenizer = tokenizer
        self.domain_token_ids = self._get_domain_token_ids(domain_tokens)

    def compute_rare_token_ppl(self, model, eval_dataset):
        """
        Compute perplexity on sequences containing domain tokens.
        """
        # Filter examples with domain tokens
        # Calculate loss only on domain token positions
        # Return perplexity
        pass

    def compute_fragmentation_rate(self, test_terms):
        """
        Measure % of domain terms split into multiple tokens.
        """
        fragmented = 0
        for term in test_terms:
            tokens = self.tokenizer.encode(term, add_special_tokens=False)
            if len(tokens) > 1:
                fragmented += 1
        return fragmented / len(test_terms)
```

#### 2.2 Custom Evaluation Callback

**Tasks:**
- Integrate rare-token perplexity into evaluation
- Log to TensorBoard/WandB
- Track fragmentation rate over time

**Code Structure:**
```python
class DAPTEvaluationCallback(TrainerCallback):
    """
    Custom callback for DAPT-specific metrics.
    """
    def __init__(self, rare_token_metrics, test_terms):
        self.rare_token_metrics = rare_token_metrics
        self.test_terms = test_terms

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        """
        Compute additional metrics after each evaluation.
        """
        # Rare-token perplexity
        rare_ppl = self.rare_token_metrics.compute_rare_token_ppl(
            model, kwargs['eval_dataloader']
        )
        metrics['rare_token_perplexity'] = rare_ppl

        # Fragmentation rate
        frag_rate = self.rare_token_metrics.compute_fragmentation_rate(
            self.test_terms
        )
        metrics['fragmentation_rate'] = frag_rate

        # Log to console and tensorboard
        logger.info(f"Rare-token PPL: {rare_ppl:.4f}")
        logger.info(f"Fragmentation rate: {frag_rate:.2%}")

        return metrics
```

#### 2.3 Early Stopping Logic

**Tasks:**
- Plateau detection on MLM loss
- Rare-token PPL improvement tracking
- Save best checkpoint

**Criteria:**
- Stop if MLM loss plateaus for 5 evaluations
- Or rare-token PPL improvement < 1% for 3 evaluations

**Code Structure:**
```python
class DAPTEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping for DAPT based on MLM and rare-token metrics.
    """
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get('eval_loss')

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            control.should_training_stop = True

        return control
```

#### 2.4 Checkpoint Management

**Tasks:**
- Save model, optimizer, scheduler state
- Save tokenizer snapshot
- Keep top-K checkpoints by validation metrics
- Resume from checkpoint support

**Code Structure:**
```python
def save_checkpoint(trainer, output_dir, step, metrics):
    """
    Save full checkpoint with metrics.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")

    # Save model
    trainer.save_model(checkpoint_dir)

    # Save tokenizer
    trainer.tokenizer.save_pretrained(checkpoint_dir)

    # Save metrics
    with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training state
    trainer.state.save_to_json(
        os.path.join(checkpoint_dir, "trainer_state.json")
    )
```

**Day 2 Deliverables:**
- ✅ Rare-token perplexity metric implemented
- ✅ Custom evaluation callback working
- ✅ Early stopping functional
- ✅ Checkpoint management complete
- ✅ Metrics logged to TensorBoard/WandB

---

### Phase 3: Configuration, Testing & Documentation (Day 3, 4-6 hours)

#### 3.1 Configuration File

**File:** `config/dapt_config.yaml`

**Tasks:**
- Externalize all hyperparameters
- Support multiple configurations (dev, prod)
- Validation of config parameters

**Structure:**
```yaml
# DAPT Training Configuration for PaleoBERT-Cambrian v1.0

model:
  name_or_path: "microsoft/deberta-v3-base"
  tokenizer_path: "artifacts/tokenizer_v1"
  gradient_checkpointing: true

data:
  train_files:
    - "data/corpus_norm/train_*.jsonl"
  eval_files:
    - "data/corpus_norm/eval_*.jsonl"
  max_seq_length: 512
  preprocessing_num_workers: 8

training:
  output_dir: "checkpoints/paleo-dapt-v1"
  overwrite_output_dir: true

  # Batch configuration (11GB VRAM optimized)
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 16

  # Memory optimization
  fp16: true

  # Optimization
  learning_rate: 2.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.98
  adam_epsilon: 1.0e-6
  max_grad_norm: 1.0

  # Schedule
  lr_scheduler_type: "linear"
  warmup_steps: 10000

  # Training length
  num_train_epochs: 3
  max_steps: 100000

  # Checkpointing
  logging_steps: 100
  eval_steps: 2000
  save_steps: 10000
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"

  # MLM
  mlm_probability: 0.15

validation:
  # Domain-specific validation
  rare_token_eval: true
  fragmentation_eval: true
  test_terms_file: "artifacts/vocab/all_terms.txt"

  # Early stopping
  early_stopping_patience: 5
  early_stopping_threshold: 0.01

wandb:
  enabled: false
  project: "paleobert-cambrian"
  name: "dapt-v1"
  tags: ["dapt", "cambrian", "deberta-v3-base"]

seed: 42
```

#### 3.2 Command-Line Interface

**Tasks:**
- Argparse configuration
- Config file override support
- Sensible defaults

**Usage:**
```bash
# Basic usage
python scripts/train_dapt.py \
  --config config/dapt_config.yaml

# Override config values
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 32

# Resume from checkpoint
python scripts/train_dapt.py \
  --config config/dapt_config.yaml \
  --resume_from_checkpoint checkpoints/paleo-dapt-v1/checkpoint-50000
```

#### 3.3 Testing

**File:** `tests/test_dapt_training.py`

**Test Cases:**
1. **test_dataset_loading**: JSONL corpus loads correctly
2. **test_tokenization**: Domain terms preserved
3. **test_mlm_collator**: Masking applied correctly
4. **test_batch_creation**: Batches fit in VRAM
5. **test_forward_pass**: Single training step succeeds
6. **test_rare_token_metric**: Metric calculation correct
7. **test_checkpoint_save_load**: Checkpoint I/O works
8. **test_resume_training**: Resume from checkpoint

**Implementation:**
```python
import unittest
import torch
from scripts.train_dapt import (
    CambrianCorpusDataset,
    CambrianMLMCollator,
    initialize_model_and_tokenizer,
)

class TestDAPTTraining(unittest.TestCase):
    def setUp(self):
        self.test_corpus = "tests/data/sample_corpus.jsonl"
        self.tokenizer_path = "artifacts/tokenizer_v1"

    def test_dataset_loading(self):
        """Test corpus loading from JSONL."""
        dataset = CambrianCorpusDataset([self.test_corpus], None)
        self.assertGreater(len(dataset), 0)

    def test_mlm_collator(self):
        """Test MLM masking."""
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        collator = CambrianMLMCollator(tokenizer)

        # Create sample batch
        batch = [...]
        masked_batch = collator(batch)

        # Verify ~15% tokens masked
        mask_rate = (masked_batch['labels'] != -100).float().mean()
        self.assertAlmostEqual(mask_rate, 0.15, delta=0.05)

    def test_forward_pass(self):
        """Test single forward pass fits in VRAM."""
        model, tokenizer = initialize_model_and_tokenizer(args)
        model = model.cuda()

        # Create batch
        batch = create_test_batch(tokenizer)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(**batch)

        self.assertIsNotNone(outputs.loss)

        # Check VRAM usage
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        self.assertLess(vram_used, 11.0)  # Under 11GB
```

#### 3.4 Documentation

**Files:**
- `scripts/README.md` (update with DAPT usage)
- `docs/DAPT_TRAINING.md` (detailed guide)

**Content:**
- Usage instructions
- Configuration options
- Expected output and metrics
- Troubleshooting guide
- Performance tuning tips

**Day 3 Deliverables:**
- ✅ Configuration file created
- ✅ CLI interface complete
- ✅ Test suite passing
- ✅ Documentation written
- ✅ Ready for prototype run

---

## File Structure

```
scripts/
├── train_dapt.py                    # Main training script (500-700 lines)
│   ├── CambrianCorpusDataset        # JSONL corpus loader
│   ├── CambrianMLMCollator          # Whole-word MLM collator
│   ├── RareTokenMetrics             # Domain-specific metrics
│   ├── DAPTEvaluationCallback       # Custom evaluation
│   ├── DAPTEarlyStoppingCallback    # Early stopping
│   └── main()                       # Entry point

config/
└── dapt_config.yaml                 # Training configuration

tests/
└── test_dapt_training.py            # Unit tests

docs/
└── DAPT_TRAINING.md                 # Training guide

tests/data/
└── sample_corpus.jsonl              # Test data (10-20 examples)
```

---

## Expected Training Timeline

### Prototype Run (10-20M tokens)
- **Purpose:** Validate pipeline
- **Duration:** 5-10 hours
- **Steps:** ~20-30k
- **Output:** Proof of concept checkpoint

### Full Run (40-50M tokens)
- **Purpose:** Production v1.0 model
- **Duration:** 20-30 hours
- **Steps:** ~100k
- **Output:** Best checkpoint for NER/RE

### Monitoring During Training

**Key Metrics to Watch:**
1. **Training Loss:** Should decrease steadily
2. **Eval Loss:** Should track training loss
3. **Rare-Token PPL:** Should improve faster than general PPL
4. **Fragmentation Rate:** Should remain ~0%
5. **GPU Utilization:** Should stay >90%
6. **Throughput:** 3.5-5.5 it/s target

**Red Flags:**
- ❌ VRAM OOM errors → Reduce batch size or seq_length
- ❌ Loss not decreasing → Check learning rate
- ❌ Throughput <2 it/s → Check data loading bottleneck
- ❌ Fragmentation rate >5% → Check tokenizer

---

## Validation Criteria

### Functional Validation
- [ ] Script executes without errors
- [ ] Fits within 11GB VRAM constraint
- [ ] Checkpoints save and load correctly
- [ ] Resume from checkpoint works
- [ ] All metrics logged properly

### Performance Validation
- [ ] Throughput: 3.5-5.5 it/s @ seq512, batch=8, GA=16
- [ ] GPU utilization: >90%
- [ ] Data loading not bottleneck (<10% wait time)

### Quality Validation
- [ ] MLM perplexity improves over training
- [ ] Held-out perplexity: ≤ baseline - 10%
- [ ] Rare-token perplexity: ≤ baseline - 20%
- [ ] Fragmentation rate: <1% for domain terms
- [ ] No catastrophic forgetting (general English MLM maintained)

---

## Integration Points

### Upstream Dependencies
- ✅ **P01 (Tokenizer):** `artifacts/tokenizer_v1/` must exist
- ✅ **P02 (Normalization):** Corpus must be normalized

### Downstream Consumers
- **P04 (NER Training):** Will load best DAPT checkpoint
- **P05 (RE Training):** Will load best DAPT checkpoint
- **P06 (Inference Pipeline):** Will use DAPT model for predictions

---

## Risk Assessment & Mitigation

### Risk 1: VRAM Overflow
**Probability:** Medium
**Impact:** High (blocks training)

**Mitigation:**
- Start with conservative settings (batch=4, GA=32)
- Test with small corpus first
- Profile VRAM usage with nvidia-smi
- Fallback: Reduce seq_length to 384 or 256

### Risk 2: Data Loading Bottleneck
**Probability:** Medium
**Impact:** Medium (slow training)

**Mitigation:**
- Use multiple data loading workers
- Prefetch batches
- Cache tokenized examples
- Use fast JSONL reader (orjson)

### Risk 3: Corpus Quality Issues
**Probability:** Low-Medium
**Impact:** High (poor model quality)

**Mitigation:**
- Validate corpus preprocessing (P02)
- Check for encoding issues
- Verify domain term coverage
- Monitor loss curves for anomalies

### Risk 4: Insufficient Corpus Size
**Probability:** Low
**Impact:** Medium (suboptimal performance)

**Mitigation:**
- Start with prototype (10-20M tokens)
- Validate improvement vs baseline
- Expand corpus if needed (50M → 100M)

### Risk 5: Catastrophic Forgetting
**Probability:** Low
**Impact:** Medium (poor generalization)

**Mitigation:**
- Use moderate learning rate (2e-4)
- Track general English MLM on held-out set
- Early stopping if general performance degrades
- Consider mixed corpus (90% domain, 10% general)

---

## Success Metrics Summary

| Metric | Target | Critical? |
|--------|--------|-----------|
| VRAM Usage | ≤11GB | ✅ Yes |
| Throughput | 3.5-5.5 it/s | ⚠️ Important |
| Held-out MLM PPL | ≤ baseline -10% | ✅ Yes |
| Rare-token PPL | ≤ baseline -20% | ✅ Yes |
| Fragmentation Rate | <1% | ✅ Yes |
| Training Time | 20-30h (full) | ⚠️ Important |
| GPU Utilization | >90% | ⚠️ Important |
| Checkpoint Valid | Resume works | ✅ Yes |

---

## Next Steps After P03 Completion

1. **Prototype Run**
   - Collect 10-20M token corpus
   - Run DAPT for 5-10 hours
   - Validate metrics

2. **Full Run**
   - Expand to 40-50M tokens
   - Run full DAPT (20-30h)
   - Select best checkpoint

3. **Evaluation**
   - Compare vs baseline DeBERTa
   - Analyze rare-token improvements
   - Document results

4. **Proceed to P04 (NER)**
   - Load DAPT checkpoint
   - Implement NER training script
   - Fine-tune on labeled data

---

## References

- **OVERVIEW.md:** Section 2 (Tokenizer & Pre-training Config)
- **P01 Devlog:** `devlog/20251029_002_P01_tokenizer_completion.md`
- **P02 Devlog:** `devlog/20251029_003_P02_normalization_implementation_complete.md`
- **DeBERTa Paper:** He et al., 2021 (ICLR)
- **DAPT Paper:** Gururangan et al., 2020 (ACL)

---

## Appendix A: Training Script Template

```python
#!/usr/bin/env python
"""
DAPT Training Script for PaleoBERT-Cambrian v1.0

Usage:
    python scripts/train_dapt.py --config config/dapt_config.yaml
"""

import os
import sys
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DAPTConfig:
    """DAPT training configuration."""
    # Model
    model_name_or_path: str
    tokenizer_path: str
    gradient_checkpointing: bool = True

    # Data
    train_files: List[str]
    eval_files: List[str]
    max_seq_length: int = 512

    # Training
    output_dir: str
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    fp16: bool = True

    # MLM
    mlm_probability: float = 0.15

    # Validation
    rare_token_eval: bool = True
    test_terms_file: Optional[str] = None


class CambrianCorpusDataset:
    """Load Cambrian corpus from JSONL."""
    # Implementation
    pass


class CambrianMLMCollator(DataCollatorForLanguageModeling):
    """MLM collator with whole-word masking."""
    # Implementation
    pass


class RareTokenMetrics:
    """Compute rare-token perplexity."""
    # Implementation
    pass


class DAPTEvaluationCallback(TrainerCallback):
    """Custom evaluation callback."""
    # Implementation
    pass


def load_config(config_path: str) -> DAPTConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DAPTConfig(**config_dict)


def initialize_model_and_tokenizer(config: DAPTConfig):
    """Load model and tokenizer."""
    # Implementation
    pass


def main():
    """Main training entry point."""
    # Parse arguments
    # Load config
    # Initialize model
    # Setup datasets
    # Create trainer
    # Train
    # Save final checkpoint
    pass


if __name__ == "__main__":
    main()
```

---

## Appendix B: Quick Start Guide

```bash
# Step 1: Prepare environment
pip install -r requirements.txt

# Step 2: Verify tokenizer exists
ls artifacts/tokenizer_v1/

# Step 3: Prepare sample corpus (for testing)
python scripts/prepare_sample_corpus.py \
  --output tests/data/sample_corpus.jsonl \
  --size 1000

# Step 4: Test training script (dry run)
python scripts/train_dapt.py \
  --config config/dapt_config_test.yaml \
  --max_steps 100

# Step 5: Run prototype training (10-20M tokens)
python scripts/train_dapt.py \
  --config config/dapt_config_prototype.yaml

# Step 6: Monitor training
tensorboard --logdir checkpoints/paleo-dapt-v1/runs

# Step 7: Evaluate best checkpoint
python scripts/evaluate_dapt.py \
  --checkpoint checkpoints/paleo-dapt-v1/checkpoint-best \
  --test_data data/corpus_norm/test.jsonl

# Step 8: Run full training (40-50M tokens)
python scripts/train_dapt.py \
  --config config/dapt_config.yaml
```

---

## Status: 📋 PLANNING COMPLETE

This implementation plan is ready for execution. Estimated timeline: 2-3 days for full implementation and testing.

**Next Action:** Begin Phase 1 implementation (Core Training Loop).
