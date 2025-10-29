# Text Normalization Module Implementation Plan

**Priority:** CRITICAL (blocks all downstream work)
**Timeline:** 2-3 days
**Status:** READY TO START

---

## Overview

Implement `src/normalization.py` to handle dual text representation (raw ↔ normalized) with character-level alignment maps for span projection.

**Core Functionality:**
1. `normalize_text(raw_text)` → `(norm_text, align_map)`
2. `project_span(span, align_map, direction)` → projected span
3. Reversible transformation with provenance tracking

---

## Normalization Rules (Cambrian-specific)

### 1. Chronostratigraphic Units

```python
# Cambrian stages
"Stage 10" → "Stage_10"
"Stage 9" → "Stage_9"
"Stage 2" → "Stage_2"
...

# Series
"Series 2" → "Series_2"
"Series 3" → "Series_3"
"Series 4" → "Series_4"

# Combined
"Cambrian Stage 10" → "Cambrian_Stage_10"
"Cambrian Series 2" → "Cambrian_Series_2"

# Regional stages
"Stage 5" → "Stage_5"
```

### 2. Stratigraphic Units

```python
# Formation names
"Wheeler Formation" → "Wheeler_Formation"
"Marjum Formation" → "Marjum_Formation"
"Burgess Shale" → "Burgess_Shale"
"Spence Shale" → "Spence_Shale"

# With modifiers
"Upper Wheeler Formation" → "Upper_Wheeler_Formation"
"Middle Member" → "Middle_Member"
```

### 3. Taxonomic Names (binomials)

```python
# Binomial nomenclature
"Olenellus wheeleri" → "Olenellus_wheeleri"
"Elrathia kingii" → "Elrathia_kingii"
"Asaphiscus wheeleri" → "Asaphiscus_wheeleri"

# With author/year (keep separate)
"Olenellus wheeleri Clark, 1924" → "Olenellus_wheeleri Clark, 1924"
```

### 4. Localities (multi-word)

```python
"House Range" → "House_Range"
"Drum Mountains" → "Drum_Mountains"
"Yoho National Park" → "Yoho_National_Park"
"Walcott Quarry" → "Walcott_Quarry"
```

---

## Implementation Specification

### Function 1: normalize_text()

```python
def normalize_text(raw_text: str) -> Tuple[str, Dict[int, int]]:
    """
    Normalize text and create character-level alignment map.

    Args:
        raw_text: Original input text

    Returns:
        norm_text: Normalized text with underscores
        align_map: {raw_idx: norm_idx} character mapping

    Example:
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)
        # norm = "Olenellus_wheeleri from Wheeler_Formation"
        # align = {0:0, 1:1, ..., 9:9, 10:10, 11:10, 12:11, ...}
    """
```

**Algorithm:**
1. Identify normalization patterns (regex or rule-based)
2. Replace spaces with underscores within matched patterns
3. Build character-level mapping (raw_idx → norm_idx)
4. Return both normalized text and alignment map

### Function 2: project_span()

```python
def project_span(
    span: Tuple[int, int],
    align_map: Dict[int, int],
    direction: str = "raw_to_norm"
) -> Tuple[int, int]:
    """
    Project span indices between raw and normalized text.

    Args:
        span: (start, end) character offsets
        align_map: Character-level alignment
        direction: "raw_to_norm" or "norm_to_raw"

    Returns:
        Projected (start, end) in target text

    Example:
        # NER extracts span in normalized text
        norm_span = (0, 19)  # "Olenellus_wheeleri"

        # Project back to raw text
        raw_span = project_span(norm_span, align_map, "norm_to_raw")
        # raw_span = (0, 18)  # "Olenellus wheeleri"
    """
```

### Function 3: create_inverse_map()

```python
def create_inverse_map(align_map: Dict[int, int]) -> Dict[int, int]:
    """
    Create inverse alignment map (norm_idx → raw_idx).

    Handles many-to-one mappings (multiple raw chars → one norm char).
    """
```

---

## Data Structure: Alignment Map

### Format

```python
align_map = {
    0: 0,    # raw[0] → norm[0]
    1: 1,    # raw[1] → norm[1]
    ...
    10: 10,  # raw[10] (space before 'w') → norm[10] (underscore)
    11: 11,  # raw[11] ('w') → norm[11] ('w')
    ...
}
```

### Edge Cases

**Case 1: Deletion (space → underscore)**
```
raw:  "Wheeler Formation"
       0123456 789...
norm: "Wheeler_Formation"
       0123456789...

align_map[7] = 7  (space → underscore)
```

**Case 2: Multi-word normalization**
```
raw:  "Cambrian Stage 10"
       01234567 89012 345
norm: "Cambrian_Stage_10"
       012345678901234

align_map[8] = 8   (space → underscore)
align_map[14] = 13 (space → underscore, shift)
```

---

## Testing Strategy

### Test Cases

```python
test_cases = [
    # Basic formation
    {
        "raw": "Wheeler Formation",
        "norm": "Wheeler_Formation",
        "spans": [(0, 17)],  # Full entity
    },

    # Cambrian stage
    {
        "raw": "Cambrian Stage 10",
        "norm": "Cambrian_Stage_10",
        "spans": [(0, 17)],
    },

    # Binomial
    {
        "raw": "Olenellus wheeleri",
        "norm": "Olenellus_wheeleri",
        "spans": [(0, 18)],
    },

    # Multiple entities
    {
        "raw": "Olenellus from Wheeler Formation in House Range",
        "norm": "Olenellus from Wheeler_Formation in House_Range",
        "spans": [
            (0, 9),   # Olenellus
            (15, 32), # Wheeler_Formation
            (36, 47), # House_Range
        ],
    },

    # No normalization needed
    {
        "raw": "This is normal text without special terms.",
        "norm": "This is normal text without special terms.",
        "spans": [],
    },
]
```

### Validation

```python
def test_normalization():
    for case in test_cases:
        norm, align = normalize_text(case["raw"])

        # Test 1: Normalized text matches expected
        assert norm == case["norm"]

        # Test 2: Alignment map is complete
        assert len(align) == len(case["raw"])

        # Test 3: Round-trip span projection
        for raw_span in case["spans"]:
            # raw → norm
            norm_span = project_span(raw_span, align, "raw_to_norm")

            # norm → raw (should recover original)
            recovered_span = project_span(norm_span, align, "norm_to_raw")

            assert recovered_span == raw_span, "Round-trip failed!"
```

---

## Implementation Phases

### Phase 1: Core Normalization (Day 1)

```
Task 1.1: Pattern definitions
├─ Define regex patterns for each category
├─ Chronostratigraphy: r"(Stage|Series)\s+\d+"
├─ Formations: r"\w+\s+(Formation|Shale|Limestone|Member)"
├─ Binomials: r"[A-Z][a-z]+\s+[a-z]+"
└─ Localities: List-based (from vocab files)

Task 1.2: Basic normalize_text()
├─ Apply patterns in order
├─ Replace spaces with underscores
├─ Return normalized text
└─ Test on simple cases

Task 1.3: Alignment map construction
├─ Track character position changes
├─ Build raw_idx → norm_idx mapping
└─ Test alignment correctness
```

### Phase 2: Span Projection (Day 2)

```
Task 2.1: project_span() implementation
├─ Forward projection (raw → norm)
├─ Inverse projection (norm → raw)
└─ Handle edge cases (boundary conditions)

Task 2.2: Inverse map creation
├─ create_inverse_map() function
├─ Handle many-to-one mappings
└─ Test bidirectional projection

Task 2.3: Comprehensive testing
├─ All test cases pass
├─ Edge case handling
└─ Performance profiling
```

### Phase 3: Integration & Documentation (Day 3)

```
Task 3.1: Integration with tokenizer
├─ Test with actual tokenizer
├─ Verify token alignment
└─ End-to-end example

Task 3.2: Documentation
├─ Docstrings for all functions
├─ Usage examples
├─ API reference
└─ README in src/

Task 3.3: Unit tests
├─ pytest test suite
├─ Coverage ≥90%
└─ CI/CD ready
```

---

## Usage Example

```python
from src.normalization import normalize_text, project_span

# Original text
raw_text = "Olenellus wheeleri occurs in Wheeler Formation, House Range."

# Normalize
norm_text, align_map = normalize_text(raw_text)
print(norm_text)
# "Olenellus_wheeleri occurs in Wheeler_Formation, House_Range."

# NER model extracts entities from norm_text
ner_results = [
    ("Olenellus_wheeleri", (0, 19), "TAXON"),
    ("Wheeler_Formation", (30, 47), "STRAT"),
    ("House_Range", (49, 60), "LOC"),
]

# Project spans back to raw text for final output
for entity, norm_span, type in ner_results:
    raw_span = project_span(norm_span, align_map, "norm_to_raw")
    print(f"{type}: {raw_text[raw_span[0]:raw_span[1]]}")
# TAXON: Olenellus wheeleri
# STRAT: Wheeler Formation
# LOC: House Range
```

---

## Next Steps

1. Create `src/` directory
2. Implement `src/normalization.py`
3. Create `tests/test_normalization.py`
4. Run tests and iterate
5. Proceed to data processing module

**Ready to start implementation?**
