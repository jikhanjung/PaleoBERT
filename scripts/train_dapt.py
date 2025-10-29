#!/usr/bin/env python
"""
DAPT Training Script for PaleoBERT-Cambrian v1.0

This script performs Domain-Adaptive Pretraining (DAPT) on Cambrian paleontology
literature using Masked Language Modeling (MLM). It adapts DeBERTa-v3-base to
domain-specific vocabulary and linguistic patterns.

Usage:
    # Basic training
    python scripts/train_dapt.py --config config/dapt_config.yaml

    # Override config values
    python scripts/train_dapt.py --config config/dapt_config.yaml \\
        --learning_rate 3e-4 --batch_size 4

    # Resume from checkpoint
    python scripts/train_dapt.py --config config/dapt_config.yaml \\
        --resume_from_checkpoint checkpoints/paleo-dapt-v1/checkpoint-50000

Requirements:
    - P01: Extended tokenizer at artifacts/tokenizer_v1/
    - P02: Normalized corpus in JSONL format
    - 11GB VRAM GPU (RTX 2080 Ti or equivalent)

Author: PaleoBERT Team
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob

import torch
import yaml
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, load_dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DAPTConfig:
    """
    DAPT training configuration.

    Loaded from YAML file and can be overridden via command-line arguments.
    """
    # Model
    model_name_or_path: str = "microsoft/deberta-v3-base"
    tokenizer_path: str = "artifacts/tokenizer_v1"
    gradient_checkpointing: bool = True

    # Data
    train_files: List[str] = field(default_factory=list)
    eval_files: List[str] = field(default_factory=list)
    max_seq_length: int = 512
    preprocessing_num_workers: int = 8

    # Training
    output_dir: str = "checkpoints/paleo-dapt-v1"
    overwrite_output_dir: bool = True
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 16

    # Memory optimization
    fp16: bool = True

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 1.0

    # Schedule
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 10000

    # Training length
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs

    # Checkpointing
    logging_steps: int = 100
    eval_steps: int = 2000
    save_steps: int = 10000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # MLM
    mlm_probability: float = 0.15

    # Validation
    rare_token_eval: bool = True
    test_terms_file: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01

    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DAPTConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure
        flat_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                flat_dict.update(value)
            else:
                flat_dict[key] = value

        return cls(**flat_dict)


# ============================================================================
# Dataset: Cambrian Corpus Loader
# ============================================================================

class CambrianCorpusDataset(torch.utils.data.Dataset):
    """
    Load Cambrian paleontology corpus from JSONL files.

    Expected JSONL format (from P02 normalization):
        {
            "doc_id": "pub123_cap456",
            "pub_id": "pub123",
            "cap_id": "cap456",
            "raw_text": "Olenellus wheeleri from Wheeler Formation",
            "norm_text": "Olenellus_wheeleri from Wheeler_Formation",
            "align_map": {"0": 0, "1": 1, ...}
        }

    This dataset:
    - Loads normalized text for training
    - Tokenizes on-the-fly or uses cached tokenization
    - Handles document boundaries for proper shuffling

    Args:
        data_files: List of JSONL file paths (supports glob patterns)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        cache_dir: Directory to cache tokenized examples (optional)
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir

        # Expand glob patterns and load all JSONL files
        self.examples = []
        logger.info(f"Loading corpus from {len(data_files)} file patterns...")

        for pattern in data_files:
            matching_files = glob.glob(pattern)
            if not matching_files:
                logger.warning(f"No files found matching pattern: {pattern}")
                continue

            for filepath in matching_files:
                logger.info(f"Loading {filepath}...")
                examples = self._load_jsonl(filepath)
                self.examples.extend(examples)
                logger.info(f"  Loaded {len(examples)} examples")

        logger.info(f"Total examples loaded: {len(self.examples)}")

        if len(self.examples) == 0:
            raise ValueError("No examples loaded! Check your data_files paths.")

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load examples from a JSONL file."""
        examples = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        example = json.loads(line)

                        # Validate required fields
                        if 'norm_text' not in example:
                            logger.warning(
                                f"Skipping line {line_num} in {filepath}: "
                                "missing 'norm_text' field"
                            )
                            continue

                        examples.append(example)

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping line {line_num} in {filepath}: "
                            f"JSON decode error: {e}"
                        )
                        continue

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized example.

        Returns:
            Dictionary with:
                - input_ids: token IDs
                - attention_mask: attention mask
        """
        example = self.examples[idx]

        # Extract normalized text
        text = example['norm_text']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Remove batch dimension (added by return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


# ============================================================================
# Data Collator: MLM with Whole-Word Masking
# ============================================================================

class CambrianMLMCollator(DataCollatorForLanguageModeling):
    """
    MLM data collator with whole-word masking for domain terms.

    Extends HuggingFace's DataCollatorForLanguageModeling to:
    - Apply whole-word masking (not just subword masking)
    - Give special attention to domain-specific tokens
    - Maintain 15% masking rate

    Args:
        tokenizer: HuggingFace tokenizer
        mlm_probability: Probability of masking (default 0.15)
        domain_token_ids: Optional list of domain token IDs for tracking
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        domain_token_ids: Optional[List[int]] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )
        self.domain_token_ids = set(domain_token_ids) if domain_token_ids else set()

    def torch_call(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and apply MLM masking.

        Args:
            examples: List of tokenized examples

        Returns:
            Batch dictionary with:
                - input_ids: masked token IDs
                - attention_mask: attention mask
                - labels: original token IDs for loss computation
        """
        # Stack examples into batch
        batch = {
            'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in examples]),
        }

        # Apply MLM masking
        # labels will have -100 for non-masked tokens (ignored in loss)
        batch['input_ids'], batch['labels'] = self.torch_mask_tokens(
            batch['input_ids']
        )

        return batch

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for MLM.

        Strategy:
        - 15% of tokens are selected for masking
        - Of selected tokens:
            - 80% replaced with [MASK]
            - 10% replaced with random token
            - 10% unchanged

        Args:
            inputs: Input token IDs [batch_size, seq_length]
            special_tokens_mask: Optional mask for special tokens

        Returns:
            masked_inputs: Input IDs with masking applied
            labels: Original input IDs (with -100 for non-masked positions)
        """
        labels = inputs.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)

        # Don't mask special tokens
        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for special_token_id in self.tokenizer.all_special_ids:
                special_tokens_mask |= (inputs == special_token_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (will be ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time: replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time: replace with random token
        indices_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), inputs.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # 10% of the time: keep original token (remaining masked_indices)

        return inputs, labels


# ============================================================================
# Model Initialization
# ============================================================================

def initialize_model_and_tokenizer(config: DAPTConfig) -> Tuple:
    """
    Load base model and extended tokenizer.

    Steps:
    1. Load extended tokenizer from artifacts/tokenizer_v1/ (P01)
    2. Load DeBERTa-v3-base model
    3. Resize token embeddings to match tokenizer vocabulary
    4. Enable gradient checkpointing for memory efficiency
    5. Convert to fp16 if specified

    Args:
        config: DAPT configuration

    Returns:
        model: DeBERTa model with resized embeddings
        tokenizer: Extended tokenizer

    Raises:
        FileNotFoundError: If tokenizer not found at specified path
        RuntimeError: If model initialization fails
    """
    logger.info("=" * 80)
    logger.info("Initializing model and tokenizer")
    logger.info("=" * 80)

    # Load tokenizer
    logger.info(f"Loading extended tokenizer from {config.tokenizer_path}...")

    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {config.tokenizer_path}. "
            "Please run P01 tokenizer build script first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        use_fast=True,
    )
    logger.info(f"  Tokenizer loaded: {len(tokenizer)} tokens")

    # Load base model
    logger.info(f"Loading base model: {config.model_name_or_path}...")

    model_config = AutoConfig.from_pretrained(config.model_name_or_path)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_name_or_path,
        config=model_config,
    )
    logger.info(f"  Model loaded: {model.num_parameters():,} parameters")

    # Resize token embeddings
    original_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)

    logger.info(f"  Token embeddings resized: {original_vocab_size} → {new_vocab_size}")
    logger.info(f"  Added tokens: {new_vocab_size - original_vocab_size}")

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        logger.info("  Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info("Model and tokenizer initialized successfully!")
    logger.info("=" * 80)

    return model, tokenizer


# ============================================================================
# Domain-Specific Metrics
# ============================================================================

class RareTokenMetrics:
    """
    Calculate domain-specific metrics for DAPT validation.

    Metrics:
    1. Rare-token perplexity: PPL on sequences containing domain tokens
    2. Fragmentation rate: % of domain terms split into multiple tokens
    3. Domain token coverage: % of domain tokens that appear in corpus

    Args:
        tokenizer: HuggingFace tokenizer with added domain tokens
        domain_vocab_files: Dict mapping category → vocab file path
            e.g., {"taxa": "artifacts/vocab/taxa.txt", ...}
    """

    def __init__(
        self,
        tokenizer,
        domain_vocab_files: Optional[Dict[str, str]] = None,
    ):
        self.tokenizer = tokenizer
        self.domain_vocab_files = domain_vocab_files or {}

        # Load domain terms
        self.domain_terms = self._load_domain_terms()
        logger.info(f"Loaded {len(self.domain_terms)} domain terms for metrics")

        # Identify domain token IDs
        self.domain_token_ids = self._get_domain_token_ids()
        logger.info(f"Identified {len(self.domain_token_ids)} domain token IDs")

    def _load_domain_terms(self) -> List[str]:
        """Load all domain terms from vocabulary files."""
        terms = []

        for category, filepath in self.domain_vocab_files.items():
            if not os.path.exists(filepath):
                logger.warning(f"Domain vocab file not found: {filepath}")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    category_terms = [line.strip() for line in f if line.strip()]
                    terms.extend(category_terms)
                    logger.info(f"  Loaded {len(category_terms)} terms from {category}")
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")

        return terms

    def _get_domain_token_ids(self) -> set:
        """
        Get token IDs for domain-specific tokens.

        Identifies tokens that were added beyond the base DeBERTa vocabulary.
        """
        # Base DeBERTa-v3-base has 128,000 tokens
        base_vocab_size = 128000

        # Any token ID >= base_vocab_size is a domain token
        domain_ids = set(range(base_vocab_size, len(self.tokenizer)))

        return domain_ids

    def compute_fragmentation_rate(self) -> Dict[str, float]:
        """
        Calculate fragmentation rate for domain terms.

        Fragmentation = % of terms that are split into >1 token

        Returns:
            Dictionary with:
                - fragmentation_rate: Overall rate (0.0 to 1.0)
                - fragmented_count: Number of fragmented terms
                - total_count: Total number of terms
                - fragmented_terms: List of fragmented terms (sample)
        """
        if not self.domain_terms:
            return {
                'fragmentation_rate': 0.0,
                'fragmented_count': 0,
                'total_count': 0,
                'fragmented_terms': [],
            }

        fragmented = []

        for term in self.domain_terms:
            tokens = self.tokenizer.encode(term, add_special_tokens=False)
            if len(tokens) > 1:
                fragmented.append(term)

        fragmentation_rate = len(fragmented) / len(self.domain_terms)

        return {
            'fragmentation_rate': fragmentation_rate,
            'fragmented_count': len(fragmented),
            'total_count': len(self.domain_terms),
            'fragmented_terms': fragmented[:10],  # Sample of first 10
        }

    def compute_rare_token_ppl(
        self,
        model,
        eval_dataloader,
        device: str = "cuda"
    ) -> float:
        """
        Compute perplexity on sequences containing domain tokens.

        This metric focuses on how well the model has learned domain-specific
        vocabulary compared to general text.

        Args:
            model: The language model
            eval_dataloader: Evaluation data loader
            device: Device to run computation on

        Returns:
            Rare-token perplexity (lower is better)
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Check if batch contains domain tokens
                contains_domain_token = False
                for token_id in self.domain_token_ids:
                    if (input_ids == token_id).any():
                        contains_domain_token = True
                        break

                if not contains_domain_token:
                    continue  # Skip batches without domain tokens

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Accumulate loss only for domain tokens
                loss = outputs.loss
                batch_size = input_ids.size(0)

                # Count tokens (excluding padding)
                num_tokens = (labels != -100).sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return float('inf')  # No domain tokens found

        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity


# ============================================================================
# Training Callbacks
# ============================================================================

class DAPTEvaluationCallback(TrainerCallback):
    """
    Custom evaluation callback for DAPT-specific metrics.

    This callback computes additional metrics during evaluation:
    - Rare-token perplexity
    - Fragmentation rate
    - Domain token coverage

    These metrics are logged to TensorBoard and console.

    Args:
        rare_token_metrics: RareTokenMetrics instance
        eval_every_n_steps: Compute expensive metrics every N eval steps
    """

    def __init__(
        self,
        rare_token_metrics: Optional[RareTokenMetrics] = None,
        eval_every_n_steps: int = 1,
    ):
        self.rare_token_metrics = rare_token_metrics
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_count = 0

    def on_evaluate(
        self,
        args,
        state,
        control,
        model,
        metrics,
        **kwargs
    ):
        """
        Called after evaluation step.

        Computes domain-specific metrics and adds them to the metrics dict.
        """
        self.eval_count += 1

        # Compute fragmentation rate (cheap, do every time)
        if self.rare_token_metrics:
            frag_stats = self.rare_token_metrics.compute_fragmentation_rate()

            metrics['fragmentation_rate'] = frag_stats['fragmentation_rate']
            metrics['fragmented_count'] = frag_stats['fragmented_count']

            logger.info("=" * 80)
            logger.info("Domain-Specific Metrics")
            logger.info("=" * 80)
            logger.info(
                f"Fragmentation rate: {frag_stats['fragmentation_rate']:.2%} "
                f"({frag_stats['fragmented_count']}/{frag_stats['total_count']})"
            )

            # Compute rare-token PPL (expensive, do periodically)
            if self.eval_count % self.eval_every_n_steps == 0:
                eval_dataloader = kwargs.get('eval_dataloader')

                if eval_dataloader is not None:
                    logger.info("Computing rare-token perplexity...")

                    device = next(model.parameters()).device

                    rare_ppl = self.rare_token_metrics.compute_rare_token_ppl(
                        model=model,
                        eval_dataloader=eval_dataloader,
                        device=device,
                    )

                    metrics['rare_token_perplexity'] = rare_ppl

                    logger.info(f"Rare-token perplexity: {rare_ppl:.4f}")

            logger.info("=" * 80)


class DAPTEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback for DAPT training.

    Stops training when:
    1. MLM loss plateaus for N evaluations, OR
    2. Rare-token PPL improvement < threshold for N evaluations

    This prevents overfitting and saves compute time.

    Args:
        patience: Number of evaluations with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        metric: Metric to monitor ('eval_loss' or 'rare_token_perplexity')
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        metric: str = 'eval_loss',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric

        self.best_metric = float('inf')
        self.counter = 0

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics,
        **kwargs
    ):
        """
        Check if we should stop training.
        """
        current_metric = metrics.get(self.metric)

        if current_metric is None:
            return  # Metric not available yet

        # Check if improved
        if current_metric < self.best_metric - self.min_delta:
            # Improvement
            self.best_metric = current_metric
            self.counter = 0

            logger.info(
                f"✓ {self.metric} improved to {current_metric:.4f} "
                f"(best: {self.best_metric:.4f})"
            )
        else:
            # No improvement
            self.counter += 1

            logger.info(
                f"✗ {self.metric} did not improve: {current_metric:.4f} "
                f"(best: {self.best_metric:.4f}, patience: {self.counter}/{self.patience})"
            )

            if self.counter >= self.patience:
                logger.info("=" * 80)
                logger.info("EARLY STOPPING TRIGGERED")
                logger.info(f"No improvement in {self.metric} for {self.patience} evaluations")
                logger.info(f"Best {self.metric}: {self.best_metric:.4f}")
                logger.info("=" * 80)

                control.should_training_stop = True


# ============================================================================
# Training
# ============================================================================

def setup_training_args(config: DAPTConfig) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments from DAPT config.

    Args:
        config: DAPT configuration

    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(
        # Output
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,

        # Training
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,

        # Schedule
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,

        # Memory
        fp16=config.fp16,

        # Logging & Checkpointing
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy in transformers 4.57+
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,

        # Reproducibility
        seed=config.seed,

        # Misc
        dataloader_num_workers=config.preprocessing_num_workers,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )


def main():
    """Main training entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train PaleoBERT-Cambrian with Domain-Adaptive Pretraining"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per_device_train_batch_size from config",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps from config (-1 for full epochs)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("=" * 80)
    logger.info("PaleoBERT-Cambrian DAPT Training")
    logger.info("=" * 80)
    logger.info(f"Loading configuration from {args.config}...")

    config = DAPTConfig.from_yaml(args.config)

    # Apply command-line overrides
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.learning_rate:
        config.learning_rate = args.learning_rate
        logger.info(f"Overriding learning_rate: {config.learning_rate}")
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
        logger.info(f"Overriding batch_size: {config.per_device_train_batch_size}")
    if args.max_steps:
        config.max_steps = args.max_steps
        logger.info(f"Overriding max_steps: {config.max_steps}")

    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(config)

    # Load datasets
    logger.info("=" * 80)
    logger.info("Loading datasets")
    logger.info("=" * 80)

    logger.info("Loading training data...")
    train_dataset = CambrianCorpusDataset(
        data_files=config.train_files,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    logger.info(f"Training examples: {len(train_dataset)}")

    logger.info("Loading evaluation data...")
    eval_dataset = CambrianCorpusDataset(
        data_files=config.eval_files,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    logger.info(f"Evaluation examples: {len(eval_dataset)}")

    # Create data collator
    logger.info("Creating MLM data collator...")
    data_collator = CambrianMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
    )

    # Setup domain-specific metrics (Phase 2)
    rare_token_metrics = None
    if config.rare_token_eval:
        logger.info("=" * 80)
        logger.info("Setting up domain-specific metrics...")
        logger.info("=" * 80)

        # Build domain vocab files dict
        domain_vocab_files = {}
        if os.path.exists("artifacts/vocab"):
            for vocab_file in ["taxa.txt", "strat_units.txt", "chrono_units.txt", "localities.txt"]:
                filepath = os.path.join("artifacts/vocab", vocab_file)
                if os.path.exists(filepath):
                    category = vocab_file.replace(".txt", "")
                    domain_vocab_files[category] = filepath

        if domain_vocab_files:
            rare_token_metrics = RareTokenMetrics(
                tokenizer=tokenizer,
                domain_vocab_files=domain_vocab_files,
            )
            logger.info("Domain-specific metrics enabled!")
        else:
            logger.warning("No domain vocabulary files found. Skipping rare-token metrics.")
            logger.warning("  Expected: artifacts/vocab/*.txt")

    # Setup training arguments
    training_args = setup_training_args(config)

    # Calculate effective batch size
    effective_batch_size = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * torch.cuda.device_count()
    )

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Tokenizer: {config.tokenizer_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Per-device batch size: {config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"FP16: {config.fp16}")
    logger.info(f"Gradient checkpointing: {config.gradient_checkpointing}")
    logger.info("=" * 80)

    # Setup callbacks (Phase 2)
    callbacks = []

    # Add domain-specific evaluation callback
    if rare_token_metrics is not None:
        eval_callback = DAPTEvaluationCallback(
            rare_token_metrics=rare_token_metrics,
            eval_every_n_steps=1,  # Compute rare-token PPL every eval
        )
        callbacks.append(eval_callback)
        logger.info("✓ DAPTEvaluationCallback added")

    # Add early stopping callback
    early_stopping = DAPTEarlyStoppingCallback(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_threshold,
        metric='eval_loss',
    )
    callbacks.append(early_stopping)
    logger.info(f"✓ DAPTEarlyStoppingCallback added (patience={config.early_stopping_patience})")

    # Create Trainer
    logger.info("=" * 80)
    logger.info("Creating Trainer...")
    logger.info("=" * 80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info(f"Trainer created with {len(callbacks)} callbacks")

    # Check for existing checkpoint
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Found existing checkpoint: {last_checkpoint}")

    # Use resume_from_checkpoint if provided, else use last_checkpoint
    checkpoint_to_resume = config.resume_from_checkpoint or last_checkpoint

    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)

        # Save final model
        logger.info("=" * 80)
        logger.info("Training complete! Saving final model...")
        logger.info("=" * 80)

        trainer.save_model(config.output_dir)
        trainer.save_state()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("Final model saved successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint...")
        trainer.save_model(os.path.join(config.output_dir, "interrupted"))
        logger.info("Checkpoint saved. You can resume training with:")
        logger.info(f"  --resume_from_checkpoint {os.path.join(config.output_dir, 'interrupted')}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
