#!/usr/bin/env python
"""
Train Relation Extraction (RE) model for PaleoBERT-Cambrian.

Fine-tunes DAPT model on relation classification task using entity markers.

Usage:
    python scripts/train_re.py --config config/re_config.yaml

Outputs:
    - checkpoints/paleo-re-v1/ - Trained model and checkpoints
    - Training logs and evaluation metrics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_re_data(filepath: str) -> List[Dict]:
    """Load RE data from JSONL."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(data: List[Dict], tokenizer, max_length: int) -> Dataset:
    """
    Prepare RE dataset for training.

    Args:
        data: List of RE examples with marked_text and label_id
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset
    """
    # Extract fields
    texts = [ex['marked_text'] for ex in data]
    labels = [ex['label_id'] for ex in data]

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    # Create dataset
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels),
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def compute_metrics(eval_pred, label_names: List[str]):
    """
    Compute evaluation metrics for RE.

    Args:
        eval_pred: Predictions from model
        label_names: List of label names

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Overall metrics
    accuracy = accuracy_score(labels, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(len(label_names))
    )

    # Micro and macro averages
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )

    # Build metrics dict
    metrics = {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }

    # Add per-relation metrics
    for i, label_name in enumerate(label_names):
        metrics[f'precision_{label_name}'] = precision[i]
        metrics[f'recall_{label_name}'] = recall[i]
        metrics[f'f1_{label_name}'] = f1[i]
        metrics[f'support_{label_name}'] = int(support[i])

    return metrics


class WeightedLossTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with class weights."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            # Apply class weights
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/re_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Check if model exists
    model_path = config['model_name_or_path']
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please ensure DAPT model is available before training RE model.")
        sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {config['tokenizer_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    # Add special tokens
    special_tokens = config.get('special_tokens', [])
    if special_tokens:
        logger.info(f"Adding special tokens: {special_tokens}")
        num_added = tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        logger.info(f"Added {num_added} special tokens")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=config['num_labels'],
        ignore_mismatched_sizes=config.get('ignore_mismatched_sizes', True)
    )

    # Resize token embeddings for new special tokens
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")

    # Load data
    logger.info("Loading RE data...")
    train_data = load_re_data(config['train_file'])
    dev_data = load_re_data(config['dev_file'])
    test_data = load_re_data(config.get('test_file', config['dev_file']))

    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Dev: {len(dev_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples")

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, config['max_seq_length'])
    dev_dataset = prepare_dataset(dev_data, tokenizer, config['max_seq_length'])
    test_dataset = prepare_dataset(test_data, tokenizer, config['max_seq_length'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        overwrite_output_dir=config.get('overwrite_output_dir', True),

        # Batch sizes
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],

        # Optimization
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        adam_beta1=config['adam_beta1'],
        adam_beta2=config['adam_beta2'],
        adam_epsilon=config['adam_epsilon'],
        max_grad_norm=config['max_grad_norm'],

        # Schedule
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],

        # Training length
        num_train_epochs=config['num_train_epochs'],
        max_steps=config.get('max_steps', -1),

        # Memory
        fp16=config['fp16'],
        gradient_checkpointing=config.get('gradient_checkpointing', False),

        # Logging
        logging_steps=config['logging_steps'],
        logging_strategy=config['logging_strategy'],

        # Evaluation
        eval_strategy=config['eval_strategy'],
        eval_steps=config.get('eval_steps'),

        # Checkpointing
        save_strategy=config['save_strategy'],
        save_steps=config.get('save_steps'),
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],

        # Misc
        seed=config['seed'],
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        label_smoothing_factor=config.get('label_smoothing_factor', 0.0),
        push_to_hub=config.get('push_to_hub', False),
    )

    # Metric computation function
    label_names = config['label_names']

    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, label_names)

    # Initialize trainer
    class_weights = config.get('class_weights')

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_wrapper,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=config.get('resume_from_checkpoint'))

    # Save final model
    logger.info(f"Saving final model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)

    # Print metrics
    logger.info("\n" + "=" * 80)
    logger.info("Test Set Results")
    logger.info("=" * 80)

    metrics = test_results.metrics

    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Micro F1: {metrics['test_f1_micro']:.4f}")
    logger.info(f"  Macro F1: {metrics['test_f1_macro']:.4f}")

    logger.info(f"\nPer-Relation Metrics:")
    for label_name in label_names:
        p = metrics.get(f'test_precision_{label_name}', 0)
        r = metrics.get(f'test_recall_{label_name}', 0)
        f = metrics.get(f'test_f1_{label_name}', 0)
        s = metrics.get(f'test_support_{label_name}', 0)
        logger.info(f"  {label_name:15s} - P: {p:.4f}, R: {r:.4f}, F1: {f:.4f} (n={s})")

    # Confusion matrix
    predictions = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    cm = confusion_matrix(labels, predictions)

    logger.info(f"\nConfusion Matrix:")
    logger.info("Rows: True labels, Columns: Predicted labels")
    logger.info(f"Label order: {label_names}")
    logger.info("\n" + str(cm))

    # Save metrics
    metrics_file = Path(config['output_dir']) / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_file}")

    # Check success criteria
    logger.info("\n" + "=" * 80)
    logger.info("Success Criteria Check (from CLAUDE.md)")
    logger.info("=" * 80)

    target_metrics = config.get('target_metrics', {})

    checks = [
        ("Micro F1 ≥ 0.75", metrics['test_f1_micro'], target_metrics.get('micro_f1', 0.75)),
        ("Macro F1 ≥ 0.70", metrics['test_f1_macro'], target_metrics.get('macro_f1', 0.70)),
        ("Accuracy ≥ 0.85", metrics['test_accuracy'], target_metrics.get('accuracy', 0.85)),
        ("F1(occurs_in) ≥ 0.80", metrics.get('test_f1_occurs_in', 0), target_metrics.get('f1_occurs_in', 0.80)),
        ("F1(found_at) ≥ 0.70", metrics.get('test_f1_found_at', 0), target_metrics.get('f1_found_at', 0.70)),
        ("F1(assigned_to) ≥ 0.70", metrics.get('test_f1_assigned_to', 0), target_metrics.get('f1_assigned_to', 0.70)),
        ("F1(part_of) ≥ 0.70", metrics.get('test_f1_part_of', 0), target_metrics.get('f1_part_of', 0.70)),
    ]

    for criterion, actual, target in checks:
        status = "✓" if actual >= target else "✗"
        logger.info(f"  {status} {criterion}: {actual:.4f}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
