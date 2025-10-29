#!/usr/bin/env python3
"""
NER Training Script for PaleoBERT-Cambrian

Fine-tunes the DAPT model for Named Entity Recognition.
Target entities: TAXON, STRAT, CHRONO, LOC

Usage:
    python scripts/train_ner.py --config config/ner_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_ner_data(file_path: str) -> List[Dict]:
    """Load NER data from JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_dataset(data: List[Dict], label2id: Dict[str, int]) -> Dataset:
    """Convert NER data to HuggingFace Dataset."""
    examples = {
        "tokens": [],
        "ner_tags": [],
        "id": [],
    }

    for i, item in enumerate(data):
        examples["tokens"].append(item["tokens"])
        # Convert string labels to IDs
        ner_tag_ids = [label2id.get(tag, label2id["O"]) for tag in item["ner_tags"]]
        examples["ner_tags"].append(ner_tag_ids)
        examples["id"].append(str(i))

    return Dataset.from_dict(examples)


def tokenize_and_align_labels(examples, tokenizer, label_pad_token_id, max_length):
    """Tokenize inputs and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding=False,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get label -100 (ignored in loss)
                label_ids.append(label_pad_token_id)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # Subsequent subwords get label -100
                # (Only first subword gets labeled)
                label_ids.append(label_pad_token_id)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list, return_entity_level_metrics=False):
    """Compute NER metrics using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

    if return_entity_level_metrics:
        # Get per-entity type metrics
        report = classification_report(true_labels, true_predictions, output_dict=True)
        # Extract F1 for each entity type
        for entity_type in ["TAXON", "STRAT", "CHRONO", "LOC"]:
            if entity_type in report:
                results[f"f1_{entity_type}"] = report[entity_type]["f1-score"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    logger.info("="*80)
    logger.info("PaleoBERT-Cambrian NER Training")
    logger.info("="*80)
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create label mappings
    label_list = config["labels"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Labels: {label_list}")

    # Load data
    logger.info("="*80)
    logger.info("Loading datasets")
    logger.info("="*80)
    logger.info(f"Train file: {config['train_file']}")
    logger.info(f"Dev file: {config['dev_file']}")
    logger.info(f"Test file: {config['test_file']}")

    train_data = load_ner_data(config["train_file"])
    dev_data = load_ner_data(config["dev_file"])
    test_data = load_ner_data(config["test_file"])

    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Dev:   {len(dev_data)} examples")
    logger.info(f"  Test:  {len(test_data)} examples")

    # Convert to datasets
    train_dataset = create_dataset(train_data, label2id)
    dev_dataset = create_dataset(dev_data, label2id)
    test_dataset = create_dataset(test_data, label2id)

    # Load tokenizer
    logger.info("="*80)
    logger.info("Loading tokenizer and model")
    logger.info("="*80)
    logger.info(f"Tokenizer: {config['tokenizer_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

    # Load model
    logger.info(f"Model: {config['model_name_or_path']}")
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name_or_path"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", True),
    )

    logger.info(f"  Model loaded: {model.num_parameters():,} parameters")

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    label_pad_token_id = -100

    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, label_pad_token_id, config["max_seq_length"]
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, label_pad_token_id, config["max_seq_length"]
        ),
        batched=True,
        remove_columns=dev_dataset.column_names,
    )

    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, label_pad_token_id, config["max_seq_length"]
        ),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=config["max_seq_length"],
    )

    # Training arguments
    logger.info("="*80)
    logger.info("Setting up training")
    logger.info("="*80)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=config["overwrite_output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta1"],
        adam_beta2=config["adam_beta2"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"],
        logging_strategy=config.get("logging_strategy", "steps"),
        logging_steps=config.get("logging_steps", 100),
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        seed=config["seed"],
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        label_smoothing_factor=config.get("label_smoothing_factor", 0.0),
        report_to=["tensorboard"],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(
            p, label_list, config.get("return_entity_level_metrics", False)
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))

    # Evaluate on test set
    logger.info("="*80)
    logger.info("Evaluating on test set...")
    logger.info("="*80)
    test_results = trainer.predict(tokenized_test)
    logger.info(f"Test results: {test_results.metrics}")

    # Save final model
    logger.info("="*80)
    logger.info("Saving final model...")
    logger.info("="*80)
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    logger.info("Training complete!")
    logger.info(f"Model saved to: {config['output_dir']}")

    # Print summary
    logger.info("="*80)
    logger.info("Training Summary")
    logger.info("="*80)
    logger.info(f"Train examples: {len(train_data)}")
    logger.info(f"Best validation F1: {trainer.state.best_metric:.4f}")
    logger.info(f"Test F1: {test_results.metrics.get('test_f1', 0):.4f}")
    logger.info(f"Test Precision: {test_results.metrics.get('test_precision', 0):.4f}")
    logger.info(f"Test Recall: {test_results.metrics.get('test_recall', 0):.4f}")

    # Entity-level metrics if available
    for entity_type in ["TAXON", "STRAT", "CHRONO", "LOC"]:
        key = f"test_f1_{entity_type}"
        if key in test_results.metrics:
            logger.info(f"Test F1 ({entity_type}): {test_results.metrics[key]:.4f}")

    logger.info("="*80)


if __name__ == "__main__":
    main()
