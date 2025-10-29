#!/usr/bin/env python
"""
Test script for DAPT Phase 2 components.

Tests:
1. RareTokenMetrics - fragmentation rate calculation
2. DAPTEvaluationCallback - metrics computation
3. DAPTEarlyStoppingCallback - stopping logic

Usage:
    python tests/test_dapt_phase2.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, MagicMock
import tempfile

# Note: These imports will fail without dependencies installed
# This is expected and documented in requirements.txt
try:
    import torch
    from transformers import AutoTokenizer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("WARNING: PyTorch/Transformers not installed. Skipping dependency-based tests.")

# Import Phase 2 components
# These imports will work even without dependencies for basic testing
try:
    from scripts.train_dapt import (
        RareTokenMetrics,
        DAPTEvaluationCallback,
        DAPTEarlyStoppingCallback,
    )
    COMPONENTS_AVAILABLE = True
except Exception as e:
    COMPONENTS_AVAILABLE = False
    print(f"WARNING: Could not import Phase 2 components: {e}")


class TestRareTokenMetricsBasic(unittest.TestCase):
    """Test RareTokenMetrics without actual tokenizer."""

    @unittest.skipIf(not COMPONENTS_AVAILABLE, "Phase 2 components not available")
    def test_init_without_vocab(self):
        """Test RareTokenMetrics initialization without vocab files."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=128400)

        metrics = RareTokenMetrics(
            tokenizer=mock_tokenizer,
            domain_vocab_files={}
        )

        self.assertEqual(len(metrics.domain_terms), 0)
        self.assertGreater(len(metrics.domain_token_ids), 0)

    @unittest.skipIf(not COMPONENTS_AVAILABLE, "Phase 2 components not available")
    def test_domain_token_id_range(self):
        """Test that domain token IDs are correctly identified."""
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=128400)

        metrics = RareTokenMetrics(tokenizer=mock_tokenizer)

        # Domain tokens should be 128000-128399 (400 added tokens)
        self.assertEqual(len(metrics.domain_token_ids), 400)
        self.assertEqual(min(metrics.domain_token_ids), 128000)
        self.assertEqual(max(metrics.domain_token_ids), 128399)


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Dependencies not installed")
class TestRareTokenMetricsWithTokenizer(unittest.TestCase):
    """Test RareTokenMetrics with sample vocab files."""

    def setUp(self):
        """Create temporary vocab files for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Create sample vocab file
        self.taxa_file = os.path.join(self.temp_dir, "taxa.txt")
        with open(self.taxa_file, 'w') as f:
            f.write("Olenellus\n")
            f.write("Asaphiscus\n")
            f.write("Elrathia\n")

        self.strat_file = os.path.join(self.temp_dir, "strat_units.txt")
        with open(self.strat_file, 'w') as f:
            f.write("Wheeler_Formation\n")
            f.write("Marjum_Formation\n")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_domain_terms(self):
        """Test loading domain terms from vocab files."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=128400)

        metrics = RareTokenMetrics(
            tokenizer=mock_tokenizer,
            domain_vocab_files={
                'taxa': self.taxa_file,
                'strat_units': self.strat_file,
            }
        )

        # Should have loaded 5 terms total
        self.assertEqual(len(metrics.domain_terms), 5)
        self.assertIn("Olenellus", metrics.domain_terms)
        self.assertIn("Wheeler_Formation", metrics.domain_terms)

    def test_fragmentation_rate_calculation(self):
        """Test fragmentation rate calculation."""
        # Mock tokenizer that fragments some terms
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=128400)

        def mock_encode(text, add_special_tokens=False):
            # Simulate: single-token for short terms, multi-token for long
            if len(text) < 10:
                return [1]  # Single token
            else:
                return [1, 2, 3]  # Fragmented

        mock_tokenizer.encode = mock_encode

        metrics = RareTokenMetrics(
            tokenizer=mock_tokenizer,
            domain_vocab_files={'taxa': self.taxa_file}
        )

        frag_stats = metrics.compute_fragmentation_rate()

        # All 3 taxa are < 10 chars, so fragmentation_rate = 0
        self.assertEqual(frag_stats['fragmentation_rate'], 0.0)
        self.assertEqual(frag_stats['fragmented_count'], 0)
        self.assertEqual(frag_stats['total_count'], 3)


@unittest.skipIf(not COMPONENTS_AVAILABLE, "Phase 2 components not available")
class TestDAPTCallbacks(unittest.TestCase):
    """Test DAPT callbacks."""

    def test_evaluation_callback_init(self):
        """Test DAPTEvaluationCallback initialization."""
        callback = DAPTEvaluationCallback(
            rare_token_metrics=None,
            eval_every_n_steps=2
        )

        self.assertIsNone(callback.rare_token_metrics)
        self.assertEqual(callback.eval_every_n_steps, 2)
        self.assertEqual(callback.eval_count, 0)

    def test_early_stopping_init(self):
        """Test DAPTEarlyStoppingCallback initialization."""
        callback = DAPTEarlyStoppingCallback(
            patience=3,
            min_delta=0.05,
            metric='eval_loss'
        )

        self.assertEqual(callback.patience, 3)
        self.assertEqual(callback.min_delta, 0.05)
        self.assertEqual(callback.metric, 'eval_loss')
        self.assertEqual(callback.best_metric, float('inf'))
        self.assertEqual(callback.counter, 0)

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        callback = DAPTEarlyStoppingCallback(
            patience=2,
            min_delta=0.01
        )

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # First evaluation: loss = 2.0
        metrics = {'eval_loss': 2.0}
        callback.on_evaluate(None, None, control, metrics)

        self.assertEqual(callback.best_metric, 2.0)
        self.assertEqual(callback.counter, 0)
        self.assertFalse(control.should_training_stop)

        # Second evaluation: loss = 1.5 (improved)
        metrics = {'eval_loss': 1.5}
        callback.on_evaluate(None, None, control, metrics)

        self.assertEqual(callback.best_metric, 1.5)
        self.assertEqual(callback.counter, 0)
        self.assertFalse(control.should_training_stop)

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after patience."""
        callback = DAPTEarlyStoppingCallback(
            patience=2,
            min_delta=0.01
        )

        control = Mock()
        control.should_training_stop = False

        # First: loss = 2.0
        metrics = {'eval_loss': 2.0}
        callback.on_evaluate(None, None, control, metrics)

        # Second: loss = 2.0 (no improvement)
        callback.on_evaluate(None, None, control, metrics)
        self.assertEqual(callback.counter, 1)
        self.assertFalse(control.should_training_stop)

        # Third: loss = 2.0 (still no improvement)
        callback.on_evaluate(None, None, control, metrics)
        self.assertEqual(callback.counter, 2)
        self.assertTrue(control.should_training_stop)  # Should stop now


def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("Testing DAPT Phase 2 Components")
    print("=" * 80)

    if not COMPONENTS_AVAILABLE:
        print("ERROR: Phase 2 components could not be imported.")
        print("Make sure scripts/train_dapt.py exists and is syntactically correct.")
        return 1

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRareTokenMetricsBasic))

    if DEPENDENCIES_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestRareTokenMetricsWithTokenizer))
    else:
        print("\nNOTE: Skipping dependency-based tests.")
        print("Install dependencies with: pip install -r requirements.txt")

    suite.addTests(loader.loadTestsFromTestCase(TestDAPTCallbacks))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
