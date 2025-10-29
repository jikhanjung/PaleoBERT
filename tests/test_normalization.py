"""
Unit tests for text normalization module.

Tests cover:
1. Basic normalization patterns (formations, stages, binomials, localities)
2. Character-level alignment map correctness
3. Span projection (raw ↔ norm)
4. Round-trip consistency
5. Edge cases (no normalization, multiple entities, complex patterns)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from src.normalization import (
    normalize_text,
    project_span,
    create_inverse_map,
    validate_normalization,
    get_normalization_stats,
)


class TestBasicNormalization(unittest.TestCase):
    """Test basic normalization patterns."""

    def test_formation_basic(self):
        """Test basic formation normalization."""
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Wheeler_Formation")
        self.assertEqual(len(align), len(raw))
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_formation_multi_word(self):
        """Test multi-word formation (e.g., 'Burgess Shale')."""
        raw = "Burgess Shale"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Burgess_Shale")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_formation_with_modifier(self):
        """Test formation with positional modifier."""
        raw = "Upper Wheeler Formation"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Upper_Wheeler_Formation")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_cambrian_stage(self):
        """Test Cambrian stage normalization."""
        raw = "Cambrian Stage 10"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Cambrian_Stage_10")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_stage_number(self):
        """Test stage number without 'Cambrian' prefix."""
        raw = "Stage 10"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Stage_10")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_series_number(self):
        """Test series number."""
        raw = "Series 2"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Series_2")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_binomial_nomenclature(self):
        """Test binomial (genus + species) normalization."""
        raw = "Olenellus wheeleri"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Olenellus_wheeleri")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_locality_range(self):
        """Test geographic locality (e.g., 'House Range')."""
        raw = "House Range"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "House_Range")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_locality_three_word(self):
        """Test three-word locality (e.g., 'Yoho National Park')."""
        raw = "Yoho National Park"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Yoho_National_Park")
        self.assertTrue(validate_normalization(raw, norm, align))


class TestComplexNormalization(unittest.TestCase):
    """Test complex scenarios with multiple entities."""

    def test_multiple_entities(self):
        """Test text with multiple entities."""
        raw = "Olenellus from Wheeler Formation in House Range"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Olenellus from Wheeler_Formation in House_Range")
        self.assertTrue(validate_normalization(raw, norm, align))

        # Check individual entities
        self.assertIn("Olenellus", norm)
        self.assertIn("Wheeler_Formation", norm)
        self.assertIn("House_Range", norm)

    def test_sentence_with_stage(self):
        """Test sentence with chronostratigraphic unit."""
        raw = "Trilobites occur in Cambrian Stage 10 deposits"
        norm, align = normalize_text(raw)

        self.assertIn("Cambrian_Stage_10", norm)
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_full_sentence(self):
        """Test realistic paleontology sentence."""
        raw = "Olenellus wheeleri occurs in Wheeler Formation, House Range"
        norm, align = normalize_text(raw)

        expected = "Olenellus_wheeleri occurs in Wheeler_Formation, House_Range"
        self.assertEqual(norm, expected)
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_no_normalization_needed(self):
        """Test text with no domain terms."""
        raw = "This is normal text without special terms."
        norm, align = normalize_text(raw)

        self.assertEqual(norm, raw)
        self.assertTrue(validate_normalization(raw, norm, align))

        # Alignment should be identity mapping
        for i in range(len(raw)):
            self.assertEqual(align[i], i)


class TestAlignmentMap(unittest.TestCase):
    """Test alignment map correctness."""

    def test_alignment_map_completeness(self):
        """Test that alignment map covers all characters."""
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # Every character in raw text should be mapped
        self.assertEqual(len(align), len(raw))

        # All indices should be present
        for i in range(len(raw)):
            self.assertIn(i, align)

    def test_alignment_map_validity(self):
        """Test that alignment map indices are valid."""
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)

        for raw_idx, norm_idx in align.items():
            # Raw index should be in range
            self.assertGreaterEqual(raw_idx, 0)
            self.assertLess(raw_idx, len(raw))

            # Norm index should be in range
            self.assertGreaterEqual(norm_idx, 0)
            self.assertLess(norm_idx, len(norm))

    def test_alignment_map_monotonic(self):
        """Test that alignment map is monotonically increasing."""
        raw = "Olenellus wheeleri"
        norm, align = normalize_text(raw)

        # Check that mapping is monotonic (never decreases)
        prev_norm_idx = -1
        for raw_idx in sorted(align.keys()):
            norm_idx = align[raw_idx]
            self.assertGreaterEqual(norm_idx, prev_norm_idx)
            prev_norm_idx = norm_idx


class TestInverseMap(unittest.TestCase):
    """Test inverse alignment map creation."""

    def test_inverse_map_basic(self):
        """Test basic inverse map creation."""
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)

        inverse = create_inverse_map(align)

        # Inverse map should be non-empty
        self.assertGreater(len(inverse), 0)

        # Check a few key positions
        self.assertIn(0, inverse)  # First character

    def test_inverse_map_roundtrip(self):
        """Test that forward + inverse mapping is consistent."""
        raw = "Olenellus wheeleri"
        norm, align = normalize_text(raw)

        inverse = create_inverse_map(align)

        # For characters that have 1:1 mapping, round-trip should work
        # (excluding the space that becomes underscore)
        for raw_idx in range(len(raw)):
            if raw[raw_idx] != ' ':  # Skip spaces (become underscores)
                norm_idx = align[raw_idx]
                if norm_idx in inverse:
                    recovered_raw_idx = inverse[norm_idx]
                    # Should recover the same or earlier position
                    self.assertLessEqual(recovered_raw_idx, raw_idx)


class TestSpanProjection(unittest.TestCase):
    """Test span projection between raw and normalized text."""

    def test_span_projection_forward(self):
        """Test raw → norm span projection."""
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # Project taxon span (raw)
        raw_span = (0, 18)  # "Olenellus wheeleri"
        norm_span = project_span(raw_span, align, "raw_to_norm")

        # Extract from normalized text
        entity = norm[norm_span[0]:norm_span[1]]
        self.assertEqual(entity, "Olenellus_wheeleri")

    def test_span_projection_inverse(self):
        """Test norm → raw span projection."""
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # NER extracts from normalized text
        norm_span = (24, 41)  # "Wheeler_Formation"
        raw_span = project_span(norm_span, align, "norm_to_raw")

        # Extract from raw text
        entity = raw[raw_span[0]:raw_span[1]]
        self.assertEqual(entity, "Wheeler Formation")

    def test_span_projection_roundtrip(self):
        """Test round-trip span projection (raw → norm → raw)."""
        raw = "Olenellus wheeleri occurs in Wheeler Formation"
        norm, align = normalize_text(raw)

        # Original span in raw text
        original_span = (0, 18)  # "Olenellus wheeleri"

        # raw → norm
        norm_span = project_span(original_span, align, "raw_to_norm")

        # norm → raw
        recovered_span = project_span(norm_span, align, "norm_to_raw")

        # Should recover original span
        self.assertEqual(recovered_span, original_span)

    def test_multiple_span_projections(self):
        """Test projection of multiple spans."""
        raw = "Olenellus from Wheeler Formation in House Range"
        norm, align = normalize_text(raw)

        # Define spans in raw text
        test_spans = [
            (0, 9),    # "Olenellus"
            (15, 32),  # "Wheeler Formation"
            (36, 47),  # "House Range"
        ]

        for raw_span in test_spans:
            # raw → norm → raw
            norm_span = project_span(raw_span, align, "raw_to_norm")
            recovered_span = project_span(norm_span, align, "norm_to_raw")

            # Round-trip should preserve span
            self.assertEqual(recovered_span, raw_span)

    def test_span_projection_error_handling(self):
        """Test error handling for invalid spans."""
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)

        # Invalid direction
        with self.assertRaises(ValueError):
            project_span((0, 5), align, "invalid_direction")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Test normalization of empty string."""
        raw = ""
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "")
        self.assertEqual(len(align), 0)

    def test_single_word(self):
        """Test normalization of single word (no multi-word terms)."""
        raw = "Trilobites"
        norm, align = normalize_text(raw)

        self.assertEqual(norm, "Trilobites")
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_punctuation(self):
        """Test handling of punctuation."""
        raw = "Olenellus wheeleri (Clark, 1924)"
        norm, align = normalize_text(raw)

        # Should normalize the binomial but keep punctuation
        self.assertIn("Olenellus_wheeleri", norm)
        self.assertIn("(Clark, 1924)", norm)

    def test_mixed_case(self):
        """Test handling of mixed case (patterns require proper case)."""
        raw = "wheeler formation"  # lowercase - should NOT match
        norm, align = normalize_text(raw)

        # Should NOT normalize (patterns require capital letters)
        self.assertEqual(norm, raw)

    def test_partial_match(self):
        """Test that partial matches don't get normalized."""
        raw = "Formation theory"  # "Formation" alone shouldn't match
        norm, align = normalize_text(raw)

        # Pattern requires "[Name] Formation", not standalone "Formation"
        # So this should not be normalized
        self.assertEqual(norm, raw)

    def test_long_text(self):
        """Test normalization of longer text."""
        raw = (
            "The Wheeler Formation in the House Range contains abundant "
            "Olenellus wheeleri specimens from Cambrian Stage 10. "
            "The Marjum Formation also yields Elrathia kingii."
        )
        norm, align = normalize_text(raw)

        # Check all entities are normalized
        self.assertIn("Wheeler_Formation", norm)
        self.assertIn("House_Range", norm)
        self.assertIn("Olenellus_wheeleri", norm)
        self.assertIn("Cambrian_Stage_10", norm)
        self.assertIn("Marjum_Formation", norm)
        self.assertIn("Elrathia_kingii", norm)

        # Alignment map should be complete
        self.assertTrue(validate_normalization(raw, norm, align))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_validate_normalization(self):
        """Test normalization validation."""
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)

        # Should be valid
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_get_normalization_stats(self):
        """Test normalization statistics."""
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        stats = get_normalization_stats(raw, norm)

        # Check stats structure
        self.assertIn('char_count_raw', stats)
        self.assertIn('char_count_norm', stats)
        self.assertIn('char_diff', stats)

        # Raw text has 2 spaces that become underscores
        # So normalized text should be same length
        self.assertEqual(stats['char_count_raw'], stats['char_count_norm'])


class TestRealWorldExamples(unittest.TestCase):
    """Test with realistic paleontology examples."""

    def test_example_1(self):
        """Example: typical figure caption."""
        raw = "Olenellus wheeleri from the Wheeler Formation, House Range, Utah."
        norm, align = normalize_text(raw)

        self.assertIn("Olenellus_wheeleri", norm)
        self.assertIn("Wheeler_Formation", norm)
        self.assertIn("House_Range", norm)
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_example_2(self):
        """Example: stratigraphic description."""
        raw = "The Upper Wheeler Formation contains Cambrian Stage 10 trilobites."
        norm, align = normalize_text(raw)

        self.assertIn("Upper_Wheeler_Formation", norm)
        self.assertIn("Cambrian_Stage_10", norm)
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_example_3(self):
        """Example: taxonomic description."""
        raw = "Asaphiscus wheeleri and Elrathia kingii occur together."
        norm, align = normalize_text(raw)

        self.assertIn("Asaphiscus_wheeleri", norm)
        self.assertIn("Elrathia_kingii", norm)
        self.assertTrue(validate_normalization(raw, norm, align))

    def test_example_4(self):
        """Example: geographic context."""
        raw = "Specimens from Yoho National Park, British Columbia."
        norm, align = normalize_text(raw)

        self.assertIn("Yoho_National_Park", norm)
        self.assertTrue(validate_normalization(raw, norm, align))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
