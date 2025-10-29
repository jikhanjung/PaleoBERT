"""
Text Normalization Module for PaleoBERT

This module provides dual text representation (raw ↔ normalized) with character-level
alignment maps for span projection. This is critical for maintaining provenance between
normalized text (used for NER/RE) and raw text (used for final output).

Core Functions:
    normalize_text(raw_text) → (norm_text, align_map)
    project_span(span, align_map, direction) → projected_span
    create_inverse_map(align_map) → inverse_map

Example:
    >>> raw = "Olenellus wheeleri occurs in Wheeler Formation"
    >>> norm, align = normalize_text(raw)
    >>> print(norm)
    "Olenellus_wheeleri occurs in Wheeler_Formation"

    >>> # NER extracts entity from normalized text
    >>> norm_span = (0, 19)  # "Olenellus_wheeleri"
    >>>
    >>> # Project back to raw text
    >>> raw_span = project_span(norm_span, align, "norm_to_raw")
    >>> print(raw[raw_span[0]:raw_span[1]])
    "Olenellus wheeleri"
"""

import re
from typing import Dict, Tuple, List, Optional


# ============================================================================
# NORMALIZATION PATTERNS (Cambrian-specific)
# ============================================================================

# Pattern priority order matters! More specific patterns should come first.
NORMALIZATION_PATTERNS = [
    # 1. Chronostratigraphic units (Cambrian-specific)
    # Match "Cambrian Stage 10", "Stage 10", "Series 2", etc.
    {
        "name": "cambrian_chrono_combined",
        "pattern": r"\b(Cambrian)\s+(Stage|Series)\s+(\d+)\b",
        "replacement": r"\1_\2_\3",
        "description": "Combined Cambrian chronostratigraphic units",
    },
    {
        "name": "chrono_stage_series",
        "pattern": r"\b(Stage|Series)\s+(\d+)\b",
        "replacement": r"\1_\2",
        "description": "Chronostratigraphic stages and series",
    },

    # 2. Stratigraphic units (formations, members, etc.)
    # Match "Upper Wheeler Formation", "Middle Member", etc.
    {
        "name": "strat_units_with_modifier",
        "pattern": r"\b([A-Z][a-z]+)\s+(Upper|Middle|Lower)\s+([A-Z][a-z]+)\s+(Formation|Member|Shale|Limestone|Sandstone|Quartzite|Dolomite)\b",
        "replacement": r"\1_\2_\3_\4",
        "description": "Stratigraphic units with position modifiers",
    },
    {
        "name": "strat_units_modifier_first",
        "pattern": r"\b(Upper|Middle|Lower)\s+([A-Z][a-z]+)\s+(Formation|Member|Shale|Limestone|Sandstone|Quartzite|Dolomite)\b",
        "replacement": r"\1_\2_\3",
        "description": "Stratigraphic units with leading modifiers",
    },
    {
        "name": "strat_units_basic",
        "pattern": r"\b([A-Z][a-z]+)\s+(Formation|Member|Shale|Limestone|Sandstone|Quartzite|Dolomite)\b",
        "replacement": r"\1_\2",
        "description": "Basic stratigraphic units",
    },
    {
        "name": "strat_units_multi_word",
        "pattern": r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(Formation|Member|Shale|Limestone|Sandstone)\b",
        "replacement": r"\1_\2_\3",
        "description": "Multi-word stratigraphic units (e.g., 'Burgess Shale')",
    },

    # 3. Geographic localities (multi-word)
    # These must come before binomials to take priority
    {
        "name": "geographic_multi_word",
        "pattern": r"\b([A-Z][a-z]+)\s+(Range|Mountains|Mountain|Canyon|Valley|Basin|Plateau|Park|Quarry)\b",
        "replacement": r"\1_\2",
        "description": "Multi-word geographic localities",
    },
    {
        "name": "geographic_three_word",
        "pattern": r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(Park|Canyon|Basin)\b",
        "replacement": r"\1_\2_\3",
        "description": "Three-word geographic localities (e.g., 'Yoho National Park')",
    },

    # 4. Taxonomic names (binomials) - LAST to avoid false positives
    # Match "Olenellus wheeleri", "Elrathia kingii", etc.
    # Conservative: require genus ≥4 chars and species ≥5 chars to avoid common words
    # Species ≥5 excludes: "is", "in", "from", "and", "the", etc.
    # Negative lookahead excludes geological terms as genus
    {
        "name": "binomial_nomenclature",
        "pattern": r"\b(?!(?:Formation|Member|Shale|Limestone|Sandstone|Group|Quartzite)\b)([A-Z][a-z]{3,})\s+([a-z]{5,})\b",
        "replacement": r"\1_\2",
        "description": "Binomial nomenclature (genus + species)",
    },
]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def normalize_text(raw_text: str) -> Tuple[str, Dict[int, int]]:
    """
    Normalize text and create character-level alignment map.

    Applies Cambrian-specific normalization rules to bind multi-word domain terms
    with underscores. Maintains character-level alignment map for round-trip
    span projection.

    Args:
        raw_text: Original input text

    Returns:
        norm_text: Normalized text with underscores
        align_map: {raw_idx: norm_idx} character-level mapping

    Example:
        >>> raw = "Olenellus wheeleri from Wheeler Formation"
        >>> norm, align = normalize_text(raw)
        >>> norm
        'Olenellus_wheeleri from Wheeler_Formation'
        >>> align[9]  # Space before 'wheeleri'
        9
        >>> align[10]  # 'w' in 'wheeleri'
        10

    Notes:
        - Patterns are applied in order (more specific first)
        - Each character in raw_text maps to exactly one position in norm_text
        - Space → underscore transformations are tracked in align_map
        - Alignment map enables bidirectional span projection
    """
    # Build alignment map as we transform
    align_map: Dict[int, int] = {}

    # Track replacements: List of (start, end, replacement) tuples
    replacements: List[Tuple[int, int, str]] = []

    # Find all matches for all patterns
    for pattern_spec in NORMALIZATION_PATTERNS:
        pattern = pattern_spec["pattern"]
        replacement = pattern_spec["replacement"]

        # Find all matches in the current raw_text
        for match in re.finditer(pattern, raw_text):
            # Get the matched text and compute replacement
            matched_text = match.group(0)
            replaced_text = re.sub(pattern, replacement, matched_text)

            # Store replacement (start, end, new_text)
            replacements.append((match.start(), match.end(), replaced_text))

    # Sort replacements by start position (reverse order for safe in-place edits)
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements and build alignment map
    norm_text = raw_text

    # Build alignment map for original text first (identity mapping)
    for i in range(len(raw_text)):
        align_map[i] = i

    # Now apply replacements from right to left (to preserve indices)
    for start, end, new_text in replacements:
        # Replace in norm_text
        norm_text = norm_text[:start] + new_text + norm_text[end:]

        # Update alignment map
        # Characters before the replacement stay the same
        # Characters in the replacement region need adjustment
        # Characters after the replacement shift by (len(new_text) - (end - start))

        length_diff = len(new_text) - (end - start)

        # Update mappings for characters in and after the replacement
        new_align_map: Dict[int, int] = {}
        for raw_idx, norm_idx in align_map.items():
            if raw_idx < start:
                # Before replacement: no change
                new_align_map[raw_idx] = norm_idx
            elif raw_idx < end:
                # Inside replacement region: map to corresponding position in new_text
                offset = raw_idx - start
                # Ensure offset doesn't exceed new_text length
                if offset < len(new_text):
                    new_align_map[raw_idx] = start + offset
                else:
                    # If raw text is longer than replacement (shouldn't happen with our patterns)
                    new_align_map[raw_idx] = start + len(new_text) - 1
            else:
                # After replacement: shift by length_diff
                new_align_map[raw_idx] = norm_idx + length_diff

        align_map = new_align_map

    return norm_text, align_map


def create_inverse_map(align_map: Dict[int, int]) -> Dict[int, int]:
    """
    Create inverse alignment map (norm_idx → raw_idx).

    Handles many-to-one mappings where multiple raw characters map to the same
    normalized character (e.g., when spaces are replaced with underscores).
    In such cases, the inverse map points to the first raw character.

    Args:
        align_map: Forward alignment map {raw_idx: norm_idx}

    Returns:
        inverse_map: Inverse alignment map {norm_idx: raw_idx}

    Example:
        >>> raw = "Wheeler Formation"
        >>> norm, align = normalize_text(raw)
        >>> inv = create_inverse_map(align)
        >>> # norm[7] is underscore, maps back to raw[7] (original space)
        >>> inv[7]
        7

    Notes:
        - For many-to-one mappings, returns the earliest raw_idx
        - Inverse map may be smaller than align_map if characters are deleted
    """
    inverse_map: Dict[int, int] = {}

    for raw_idx, norm_idx in sorted(align_map.items()):
        # Only store the first (earliest) raw_idx for each norm_idx
        if norm_idx not in inverse_map:
            inverse_map[norm_idx] = raw_idx

    return inverse_map


def project_span(
    span: Tuple[int, int],
    align_map: Dict[int, int],
    direction: str = "raw_to_norm"
) -> Tuple[int, int]:
    """
    Project span indices between raw and normalized text.

    Enables bidirectional span conversion:
    - raw_to_norm: Project raw text spans to normalized text (for NER input)
    - norm_to_raw: Project normalized spans to raw text (for final output)

    Args:
        span: (start, end) character offsets in source text
        align_map: Character-level alignment map
        direction: "raw_to_norm" or "norm_to_raw"

    Returns:
        Projected (start, end) in target text

    Example:
        >>> raw = "Olenellus wheeleri occurs in Wheeler Formation"
        >>> norm, align = normalize_text(raw)
        >>>
        >>> # NER extracts span in normalized text
        >>> norm_span = (0, 19)  # "Olenellus_wheeleri"
        >>>
        >>> # Project back to raw text
        >>> raw_span = project_span(norm_span, align, "norm_to_raw")
        >>> raw_span
        (0, 18)
        >>> raw[raw_span[0]:raw_span[1]]
        'Olenellus wheeleri'

    Raises:
        ValueError: If direction is not "raw_to_norm" or "norm_to_raw"
        KeyError: If span indices are not in alignment map

    Notes:
        - Span end is exclusive (Python convention)
        - Round-trip projection should preserve original span
        - Handles edge cases where multiple chars map to same position
    """
    start, end = span

    if direction == "raw_to_norm":
        # Forward projection: raw → norm
        # Use align_map directly
        if start not in align_map or (end - 1) not in align_map:
            raise KeyError(f"Span indices {span} not in alignment map")

        norm_start = align_map[start]
        # End is exclusive, so map (end - 1) and add 1
        norm_end = align_map[end - 1] + 1

        return (norm_start, norm_end)

    elif direction == "norm_to_raw":
        # Inverse projection: norm → raw
        # Create inverse map
        inverse_map = create_inverse_map(align_map)

        if start not in inverse_map or (end - 1) not in inverse_map:
            raise KeyError(f"Span indices {span} not in inverse alignment map")

        raw_start = inverse_map[start]
        # End is exclusive, so map (end - 1) and add 1
        raw_end = inverse_map[end - 1] + 1

        return (raw_start, raw_end)

    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'raw_to_norm' or 'norm_to_raw'")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_normalization(
    raw_text: str,
    norm_text: str,
    align_map: Dict[int, int]
) -> bool:
    """
    Validate normalization results.

    Checks:
    1. Alignment map covers all raw text characters
    2. Alignment map indices are valid for norm_text
    3. No missing or duplicate mappings

    Args:
        raw_text: Original text
        norm_text: Normalized text
        align_map: Alignment map

    Returns:
        True if valid, False otherwise
    """
    # Check 1: All raw characters are mapped
    if len(align_map) != len(raw_text):
        return False

    # Check 2: All indices are in range
    for raw_idx, norm_idx in align_map.items():
        if raw_idx < 0 or raw_idx >= len(raw_text):
            return False
        if norm_idx < 0 or norm_idx >= len(norm_text):
            return False

    # Check 3: Mapping is sequential (raw indices should be continuous)
    expected_indices = set(range(len(raw_text)))
    actual_indices = set(align_map.keys())
    if expected_indices != actual_indices:
        return False

    return True


def get_normalization_stats(raw_text: str, norm_text: str) -> Dict[str, int]:
    """
    Get statistics about normalization transformations.

    Args:
        raw_text: Original text
        norm_text: Normalized text

    Returns:
        Dictionary with stats: {
            'char_count_raw': int,
            'char_count_norm': int,
            'char_diff': int,
            'spaces_raw': int,
            'underscores_norm': int,
        }
    """
    return {
        'char_count_raw': len(raw_text),
        'char_count_norm': len(norm_text),
        'char_diff': len(norm_text) - len(raw_text),
        'spaces_raw': raw_text.count(' '),
        'underscores_norm': norm_text.count('_'),
    }
