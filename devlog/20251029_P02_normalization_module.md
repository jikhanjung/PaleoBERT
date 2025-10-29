# P02 Text Normalization Module Implementation

**Date:** 2025-10-29
**Milestone:** M1 Preparation (DAPT prerequisites)
**Priority:** CRITICAL
**Estimated Time:** 2-3 days
**Status:** PLANNED

---

## Executive Summary

Implement the text normalization module (`src/normalization.py`) to handle dual text representation (raw ↔ normalized) with character-level alignment maps. This is a critical prerequisite for all downstream tasks (DAPT, NER, RE) as it enables:

1. Training on normalized text (with underscored multi-word units)
2. Round-trip span projection (NER results → raw text offsets)
3. Provenance tracking (align character positions between representations)

**Decision Context:** Following Route B strategy (prototype with 120 tokens), normalization module is the first implementation task before DAPT.

---

## Background & Requirements

### Why Normalization is Critical

**Problem:** Multi-word domain terms fragment during tokenization
```
"Wheeler Formation" (base tokenizer)
→ ["Whe", "##eler", "Formation"] (3 tokens)

"Wheeler_Formation" (normalized, with added token)
→ ["Wheeler_Formation"] (1 token) ✅
```

**Solution:** Maintain dual representation
- **raw_text:** Original input ("Wheeler Formation")
- **norm_text:** Normalized view ("Wheeler_Formation")
- **align_map:** Character-level mapping for span conversion

### Architecture Context

```
┌─────────────────────────────────────────────────┐
│ Input Text (raw)                                │
│ "Olenellus wheeleri from Wheeler Formation"    │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   normalize_text()    │ ← P02 (this task)
        └───────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Normalized Text + Align Map                     │
│ norm: "Olenellus_wheeleri from Wheeler_Formation│
│ align: {0:0, 1:1, ..., 9:9, 10:10, 11:10, ...}  │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   Tokenizer (v1)      │ (uses normalized text)
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   NER Model           │ (operates on norm_text)
        └───────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ NER Results (normalized offsets)                │
│ ("Olenellus_wheeleri", 0, 19, "TAXON")          │
│ ("Wheeler_Formation", 25, 42, "STRAT")          │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   project_span()      │ ← P02 (this task)
        └───────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Final Output (raw offsets)                      │
│ ("Olenellus wheeleri", 0, 18, "TAXON")          │
│ ("Wheeler Formation", 24, 41, "STRAT")          │
└─────────────────────────────────────────────────┘
```

---

## Normalization Rules (Cambrian-Specific)

### Category 1: Chronostratigraphic Units

```python
# Cambrian stages (numbered)
"Stage 10" → "Stage_10"
"Stage 9" → "Stage_9"
"Stage 2" → "Stage_2"

# Cambrian series
"Series 2" → "Series_2"
"Series 3" → "Series_3"
"Series 4" → "Series_4"

# Combined forms
"Cambrian Stage 10" → "Cambrian_Stage_10"
"Cambrian Series 2" → "Cambrian_Series_2"

# Named stages (already single words)
"Paibian" → "Paibian" (no change)
"Guzhangian" → "Guzhangian" (no change)
```

### Category 2: Stratigraphic Units

```python
# Formation names
"Wheeler Formation" → "Wheeler_Formation"
"Marjum Formation" → "Marjum_Formation"
"Burgess Shale" → "Burgess_Shale"
"Spence Shale" → "Spence_Shale"
"Bright Angel Formation" → "Bright_Angel_Formation"

# With modifiers
"Upper Wheeler Formation" → "Upper_Wheeler_Formation"
"Middle Member" → "Middle_Member"
"Lower Member" → "Lower_Member"

# Generic terms (single words, no change)
"Formation" → "Formation"
"Shale" → "Shale"
"Limestone" → "Limestone"
```

### Category 3: Taxonomic Names (Binomials)

```python
# Trilobite binomials
"Olenellus wheeleri" → "Olenellus_wheeleri"
"Elrathia kingii" → "Elrathia_kingii"
"Asaphiscus wheeleri" → "Asaphiscus_wheeleri"
"Paradoxides davidis" → "Paradoxides_davidis"

# Genus only (single word, no change)
"Olenellus" → "Olenellus"
"Elrathia" → "Elrathia"

# With author/year (keep separate)
"Olenellus wheeleri Clark, 1924" → "Olenellus_wheeleri Clark, 1924"
```

### Category 4: Localities (Multi-word)

```python
"House Range" → "House_Range"
"Drum Mountains" → "Drum_Mountains"
"Yoho National Park" → "Yoho_National_Park"
"Walcott Quarry" → "Walcott_Quarry"
"British Columbia" → "British_Columbia"

# Single-word localities (no change)
"Utah" → "Utah"
"Nevada" → "Nevada"
```

---

## Implementation Design

### Module Structure

```
src/
├── __init__.py
└── normalization.py
    ├── normalize_text()        # Main function
    ├── project_span()          # Span conversion
    ├── create_inverse_map()    # Align map inversion
    ├── _apply_patterns()       # Pattern matching (internal)
    └── NORMALIZATION_PATTERNS  # Pattern definitions (constant)
```

### Core Functions

#### Function 1: normalize_text()

```python
def normalize_text(raw_text: str) -> Tuple[str, Dict[int, int]]:
    """
    Normalize text with Cambrian-specific rules and create alignment map.

    Args:
        raw_text: Original input text

    Returns:
        norm_text: Normalized text with underscores
        align_map: {raw_char_idx: norm_char_idx} mapping

    Example:
        >>> raw = "Olenellus wheeleri from Wheeler Formation"
        >>> norm, align = normalize_text(raw)
        >>> print(norm)
        "Olenellus_wheeleri from Wheeler_Formation"
        >>> align[10]  # Space between "Olenellus" and "wheeleri"
        10  # Maps to underscore position
    """
```

**Algorithm:**
1. Initialize: `norm_text = raw_text`, `align_map = {i: i for i in range(len(raw_text))}`
2. For each normalization pattern (in priority order):
   - Find all matches in current norm_text
   - Replace spaces with underscores within matches
   - Update align_map to reflect position shifts
3. Return final (norm_text, align_map)

**Implementation Notes:**
- Process patterns in order (chronostratigraphy → formations → binomials → localities)
- Track cumulative character position shifts
- Preserve character-level alignment throughout

#### Function 2: project_span()

```python
def project_span(
    span: Tuple[int, int],
    align_map: Dict[int, int],
    direction: str = "raw_to_norm"
) -> Tuple[int, int]:
    """
    Project span between raw and normalized text.

    Args:
        span: (start_idx, end_idx) in source text
        align_map: Character alignment from normalize_text()
        direction: "raw_to_norm" or "norm_to_raw"

    Returns:
        Projected (start_idx, end_idx) in target text

    Example:
        >>> # NER found entity in normalized text
        >>> norm_span = (0, 19)  # "Olenellus_wheeleri"
        >>> raw_span = project_span(norm_span, align, "norm_to_raw")
        >>> print(raw_span)
        (0, 18)  # "Olenellus wheeleri" in raw text
    """
```

**Algorithm:**
- **raw_to_norm:** Directly use align_map
  - `norm_start = align_map[raw_start]`
  - `norm_end = align_map[raw_end - 1] + 1`

- **norm_to_raw:** Use inverse align_map
  - Create inverse: `{norm_idx: raw_idx}`
  - Handle many-to-one mappings (multiple raw chars → one norm char)
  - `raw_start = inverse_map[norm_start]`
  - `raw_end = inverse_map[norm_end - 1] + 1`

#### Function 3: create_inverse_map()

```python
def create_inverse_map(align_map: Dict[int, int]) -> Dict[int, int]:
    """
    Create inverse alignment map for norm→raw projection.

    Args:
        align_map: {raw_idx: norm_idx}

    Returns:
        inverse_map: {norm_idx: raw_idx}

    Handles many-to-one mappings:
    - If raw[10] and raw[11] both map to norm[10]
    - inverse_map[10] = 10 (first occurrence)
    """
```

**Algorithm:**
1. Initialize: `inverse_map = {}`
2. For each `(raw_idx, norm_idx)` in align_map:
   - If `norm_idx` not in inverse_map: `inverse_map[norm_idx] = raw_idx`
   - Else: Keep first occurrence (maintains span boundaries)
3. Return inverse_map

---

## Pattern Definitions

### Pattern Priority Order

```python
NORMALIZATION_PATTERNS = [
    # Priority 1: Chronostratigraphic units (most specific)
    {
        "name": "cambrian_stage_full",
        "pattern": r"\bCambrian\s+Stage\s+(\d+)\b",
        "replacement": r"Cambrian_Stage_\1",
    },
    {
        "name": "cambrian_series_full",
        "pattern": r"\bCambrian\s+Series\s+(\d+)\b",
        "replacement": r"Cambrian_Series_\1",
    },
    {
        "name": "stage_number",
        "pattern": r"\bStage\s+(\d+)\b",
        "replacement": r"Stage_\1",
    },
    {
        "name": "series_number",
        "pattern": r"\bSeries\s+(\d+)\b",
        "replacement": r"Series_\1",
    },

    # Priority 2: Stratigraphic units
    {
        "name": "formation",
        "pattern": r"\b([A-Z][a-z]+)\s+(Formation|Shale|Limestone|Sandstone|Member|Group)\b",
        "replacement": r"\1_\2",
    },
    {
        "name": "formation_with_modifier",
        "pattern": r"\b(Upper|Middle|Lower|Basal)\s+([A-Z][a-z]+)\s+(Formation|Member)\b",
        "replacement": r"\1_\2_\3",
    },

    # Priority 3: Binomial nomenclature
    {
        "name": "binomial",
        "pattern": r"\b([A-Z][a-z]+)\s+([a-z]+)\b(?!\s+(Formation|Shale|Member))",
        "replacement": r"\1_\2",
    },

    # Priority 4: Localities (from vocab list)
    {
        "name": "locality_twoword",
        "pattern": r"\b(House|Drum|Wellsville|Yoho\s+National|Walcott|British)\s+(Range|Mountains|Park|Quarry|Columbia)\b",
        "replacement": r"\1_\2",
    },
]
```

**Pattern Design Notes:**
- Use word boundaries (`\b`) to avoid partial matches
- Process in order (specific → general)
- Use negative lookahead to avoid conflicts (e.g., binomial vs formation)
- Capture groups for flexible replacement

---

## Implementation Workflow

### Day 1: Core Normalization

**Morning (3-4 hours):**

```
Task 1.1: Project setup (30 min)
├─ Create src/ directory
├─ Create src/__init__.py
├─ Create src/normalization.py skeleton
└─ Set up imports and constants

Task 1.2: Pattern definitions (1 hour)
├─ Implement NORMALIZATION_PATTERNS list
├─ Define all regex patterns
├─ Test patterns individually (regex101.com)
└─ Validate pattern priority order

Task 1.3: Basic normalize_text() (2 hours)
├─ Implement pattern matching loop
├─ Space → underscore replacement
├─ Initial alignment map (no shift tracking yet)
└─ Test on simple cases
```

**Afternoon (3-4 hours):**

```
Task 1.4: Alignment map construction (2-3 hours)
├─ Track character position shifts during normalization
├─ Build {raw_idx: norm_idx} mapping
├─ Handle multiple replacements correctly
└─ Edge case testing (overlapping patterns, boundaries)

Task 1.5: Unit tests - Part 1 (1 hour)
├─ Create tests/test_normalization.py
├─ Test each pattern individually
├─ Test basic alignment map correctness
└─ Pytest setup and configuration
```

**End of Day 1 Deliverables:**
- ✅ Working normalize_text() function
- ✅ Correct alignment map generation
- ✅ Basic unit tests passing
- ✅ Can normalize all sample vocab terms

---

### Day 2: Span Projection & Testing

**Morning (3-4 hours):**

```
Task 2.1: create_inverse_map() (1 hour)
├─ Implement inverse map generation
├─ Handle many-to-one mappings
├─ Test bidirectional conversion
└─ Edge case testing

Task 2.2: project_span() - Forward (1-2 hours)
├─ Implement raw → norm projection
├─ Direct align_map lookup
├─ Boundary condition handling
└─ Test on diverse spans

Task 2.3: project_span() - Inverse (1 hour)
├─ Implement norm → raw projection
├─ Use inverse_map
├─ Test round-trip conversion
└─ Validate span recovery
```

**Afternoon (3-4 hours):**

```
Task 2.4: Comprehensive testing (2-3 hours)
├─ Complex test cases (multiple entities per text)
├─ Edge cases:
│   ├─ Empty text
│   ├─ No normalization needed
│   ├─ Consecutive normalized terms
│   ├─ Nested patterns
│   └─ Boundary overlaps
├─ Round-trip validation suite
└─ Performance testing (large texts)

Task 2.5: Integration testing (1 hour)
├─ Test with actual tokenizer
├─ End-to-end: raw text → normalize → tokenize
├─ Verify token boundaries align
└─ Test with 120-token vocabulary
```

**End of Day 2 Deliverables:**
- ✅ Complete normalization.py module
- ✅ All functions implemented
- ✅ Comprehensive test suite
- ✅ Integration verified with tokenizer

---

### Day 3: Documentation & Polish

**Morning (2-3 hours):**

```
Task 3.1: Documentation (1.5-2 hours)
├─ Function docstrings (numpy style)
├─ Module-level documentation
├─ Usage examples in docstrings
├─ API reference generation
└─ README.md in src/

Task 3.2: Code quality (1 hour)
├─ Type hints for all functions
├─ Code formatting (black)
├─ Linting (flake8)
├─ Docstring coverage check
└─ Remove debug code
```

**Afternoon (2-3 hours):**

```
Task 3.3: Advanced testing (1-2 hours)
├─ Property-based testing (hypothesis)
├─ Stress testing (100K character texts)
├─ Coverage analysis (aim for ≥90%)
├─ Edge case expansion
└─ Performance profiling

Task 3.4: Examples & demos (1 hour)
├─ Create examples/ directory
├─ end_to_end_example.py
├─ Interactive demo notebook (optional)
└─ Update NORMALIZATION_SPEC.md with results

Task 3.5: Final review (30 min)
├─ Code review checklist
├─ All tests passing
├─ Documentation complete
├─ Ready for integration with DAPT pipeline
└─ Git commit and tag
```

**End of Day 3 Deliverables:**
- ✅ Production-ready normalization module
- ✅ Complete documentation
- ✅ ≥90% test coverage
- ✅ Examples and demos
- ✅ Ready for M1 (DAPT)

---

## Test Strategy

### Test Categories

#### 1. Unit Tests (test_normalization.py)

```python
class TestNormalizationPatterns:
    """Test each normalization pattern individually"""

    def test_stage_normalization(self):
        raw = "Cambrian Stage 10"
        norm, align = normalize_text(raw)
        assert norm == "Cambrian_Stage_10"
        assert len(align) == len(raw)

    def test_formation_normalization(self):
        raw = "Wheeler Formation"
        norm, align = normalize_text(raw)
        assert norm == "Wheeler_Formation"

    def test_binomial_normalization(self):
        raw = "Olenellus wheeleri"
        norm, align = normalize_text(raw)
        assert norm == "Olenellus_wheeleri"

    def test_locality_normalization(self):
        raw = "House Range"
        norm, align = normalize_text(raw)
        assert norm == "House_Range"

    def test_no_normalization_needed(self):
        raw = "This is normal text without special terms."
        norm, align = normalize_text(raw)
        assert norm == raw
        assert align == {i: i for i in range(len(raw))}


class TestSpanProjection:
    """Test span conversion between raw and normalized text"""

    def test_forward_projection(self):
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # Test taxon span
        raw_span = (0, 18)  # "Olenellus wheeleri"
        norm_span = project_span(raw_span, align, "raw_to_norm")
        assert norm[norm_span[0]:norm_span[1]] == "Olenellus_wheeleri"

    def test_inverse_projection(self):
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # Simulate NER finding entity in normalized text
        norm_span = (24, 41)  # "Wheeler_Formation" in norm_text
        raw_span = project_span(norm_span, align, "norm_to_raw")

        assert raw[raw_span[0]:raw_span[1]] == "Wheeler Formation"

    def test_round_trip(self):
        """Verify raw → norm → raw preserves spans"""
        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        original_span = (0, 18)
        norm_span = project_span(original_span, align, "raw_to_norm")
        recovered_span = project_span(norm_span, align, "norm_to_raw")

        assert recovered_span == original_span


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_text(self):
        raw = ""
        norm, align = normalize_text(raw)
        assert norm == ""
        assert align == {}

    def test_consecutive_entities(self):
        raw = "Wheeler Formation Spence Shale"
        norm, align = normalize_text(raw)
        assert norm == "Wheeler_Formation Spence_Shale"

    def test_overlapping_patterns(self):
        # Binomial that could also match formation pattern
        raw = "Olenellus Formation"  # Not a real formation
        norm, align = normalize_text(raw)
        # Should normalize as formation (higher priority)
        assert norm == "Olenellus_Formation"

    def test_author_year(self):
        raw = "Olenellus wheeleri Clark, 1924"
        norm, align = normalize_text(raw)
        assert norm == "Olenellus_wheeleri Clark, 1924"
```

#### 2. Integration Tests

```python
class TestTokenizerIntegration:
    """Test normalization with actual tokenizer"""

    def test_with_tokenizer(self):
        from transformers import AutoTokenizer

        # Load PaleoBERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained("artifacts/tokenizer_v1")

        raw = "Olenellus wheeleri from Wheeler Formation"
        norm, align = normalize_text(raw)

        # Tokenize normalized text
        tokens = tokenizer.tokenize(norm)

        # Verify domain terms are single tokens
        assert "Olenellus_wheeleri" in tokens
        assert "Wheeler_Formation" in tokens

    def test_end_to_end_pipeline(self):
        """Simulate full NER pipeline"""

        raw = "Olenellus wheeleri occurs in Wheeler Formation, Utah."
        norm, align = normalize_text(raw)

        # Simulate NER results (normally from model)
        ner_results_norm = [
            ("Olenellus_wheeleri", (0, 19), "TAXON"),
            ("Wheeler_Formation", (30, 47), "STRAT"),
            ("Utah", (49, 53), "LOC"),
        ]

        # Project back to raw text
        ner_results_raw = []
        for text, norm_span, label in ner_results_norm:
            raw_span = project_span(norm_span, align, "norm_to_raw")
            raw_text = raw[raw_span[0]:raw_span[1]]
            ner_results_raw.append((raw_text, raw_span, label))

        # Verify
        assert ner_results_raw[0] == ("Olenellus wheeleri", (0, 18), "TAXON")
        assert ner_results_raw[1] == ("Wheeler Formation", (29, 46), "STRAT")
        assert ner_results_raw[2] == ("Utah", (48, 52), "LOC")
```

#### 3. Property-Based Tests (hypothesis)

```python
from hypothesis import given, strategies as st

class TestProperties:
    """Property-based tests using Hypothesis"""

    @given(st.text())
    def test_alignment_length(self, text):
        """Alignment map should have entry for every character"""
        norm, align = normalize_text(text)
        assert len(align) == len(text)

    @given(st.text(min_size=1))
    def test_monotonic_alignment(self, text):
        """Alignment should be monotonically non-decreasing"""
        norm, align = normalize_text(text)
        indices = list(align.values())
        assert all(indices[i] <= indices[i+1] for i in range(len(indices)-1))

    @given(st.text(min_size=10, max_size=1000))
    def test_round_trip_identity(self, text):
        """Any span should survive round-trip projection"""
        norm, align = normalize_text(text)

        if len(text) < 5:
            return

        # Random span
        start = len(text) // 4
        end = len(text) // 2

        # Round trip
        norm_span = project_span((start, end), align, "raw_to_norm")
        recovered = project_span(norm_span, align, "norm_to_raw")

        assert recovered == (start, end)
```

### Coverage Goals

```
Target Coverage: ≥90%

Line coverage:   ≥90%
Branch coverage: ≥85%
Function coverage: 100%

Key areas:
- normalize_text(): All patterns tested
- project_span(): Both directions + edge cases
- create_inverse_map(): All mapping scenarios
- Error handling: All exceptions tested
```

---

## Success Criteria

### Functional Requirements

```
✅ F1: normalize_text() correctly applies all Cambrian patterns
✅ F2: Alignment map has entry for every character in raw_text
✅ F3: project_span() correctly converts spans in both directions
✅ F4: Round-trip projection preserves original spans
✅ F5: No false positives (normal text unchanged)
✅ F6: Handles edge cases (empty, consecutive entities, overlaps)
```

### Performance Requirements

```
✅ P1: normalize_text() completes in <100ms for 10K character text
✅ P2: project_span() completes in <1ms per span
✅ P3: Memory usage scales linearly with text length
✅ P4: No memory leaks in batch processing
```

### Quality Requirements

```
✅ Q1: Test coverage ≥90%
✅ Q2: All functions have type hints
✅ Q3: All public functions have docstrings
✅ Q4: Passes flake8 and black formatting
✅ Q5: No pylint warnings (except approved exceptions)
```

---

## Integration Points

### Upstream Dependencies

```
- artifacts/vocab/*.txt (for pattern validation)
- artifacts/tokenizer_v1/ (for integration testing)
- Python 3.8+
- Standard library only (no external dependencies for core)
```

### Downstream Consumers

```
- scripts/train_dapt.py (M1)
  └─ Normalizes DAPT corpus before training

- scripts/train_ner.py (M2)
  └─ Normalizes NER training data
  └─ Stores align_map with training examples

- scripts/train_re.py (M3)
  └─ Normalizes RE training data

- scripts/infer_pipeline.py (M4)
  └─ Normalizes inference inputs
  └─ Projects NER/RE outputs back to raw text
```

---

## Risk Assessment

### Risk 1: Pattern Conflicts

**Risk:** Overlapping patterns cause incorrect normalization
**Example:** "Olenellus Formation" (not a real formation)

**Mitigation:**
- Order patterns by specificity (specific → general)
- Use negative lookahead in regex to avoid conflicts
- Comprehensive test coverage for edge cases
- Manual review of pattern priority

**Likelihood:** Medium
**Impact:** High
**Status:** Mitigated by careful pattern design

### Risk 2: Performance on Large Texts

**Risk:** Slow normalization on very large documents (>100K chars)

**Mitigation:**
- Profile performance on representative texts
- Optimize regex patterns (compile once, reuse)
- Consider chunking for extremely large texts
- Set performance benchmarks (100ms target for 10K chars)

**Likelihood:** Low
**Impact:** Medium
**Status:** Monitored, not critical for prototype

### Risk 3: Span Projection Errors

**Risk:** Off-by-one errors in span conversion

**Mitigation:**
- Extensive round-trip testing
- Property-based testing (Hypothesis)
- Manual validation on diverse examples
- Clear documentation of index conventions (inclusive/exclusive)

**Likelihood:** Medium (common bug type)
**Impact:** Critical (breaks NER output)
**Status:** **HIGH PRIORITY** - extensive testing required

### Risk 4: Incomplete Pattern Coverage

**Risk:** Missing important normalization patterns

**Mitigation:**
- Start with known vocabulary (120 tokens)
- Iteratively add patterns as needed
- Corpus analysis to find gaps
- Maintainable pattern list (easy to extend)

**Likelihood:** Medium
**Impact:** Medium
**Status:** Acceptable for v1.0, can extend later

---

## Dependencies & Prerequisites

### Required Before Starting

```
✅ P01 complete (tokenizer_v1 with 120 tokens)
✅ artifacts/vocab/*.txt files exist
✅ Python 3.8+ environment
✅ pytest installed (for testing)
```

### Optional (Nice to Have)

```
- hypothesis (property-based testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
```

---

## Deliverables Checklist

### Code

```
- [ ] src/__init__.py
- [ ] src/normalization.py
  - [ ] normalize_text()
  - [ ] project_span()
  - [ ] create_inverse_map()
  - [ ] NORMALIZATION_PATTERNS
  - [ ] Helper functions
```

### Tests

```
- [ ] tests/test_normalization.py
  - [ ] TestNormalizationPatterns
  - [ ] TestSpanProjection
  - [ ] TestEdgeCases
  - [ ] TestTokenizerIntegration
  - [ ] TestProperties (hypothesis)
- [ ] Coverage report ≥90%
```

### Documentation

```
- [ ] src/README.md (module overview)
- [ ] Function docstrings (numpy style)
- [ ] Type hints (all public functions)
- [ ] examples/normalization_demo.py
- [ ] Update docs/NORMALIZATION_SPEC.md with results
```

### Quality Assurance

```
- [ ] All tests passing
- [ ] Black formatted
- [ ] Flake8 clean
- [ ] Mypy type-checks pass
- [ ] Performance benchmarks met
- [ ] Integration with tokenizer verified
```

---

## Timeline & Milestones

```
Day 1 (6-8 hours):
├─ Morning:  Pattern setup + basic normalize_text()
├─ Afternoon: Alignment map + initial tests
└─ EOD: ✅ Working normalization, basic tests

Day 2 (6-8 hours):
├─ Morning:  Span projection (both directions)
├─ Afternoon: Comprehensive testing + integration
└─ EOD: ✅ Complete module, all functions working

Day 3 (4-6 hours):
├─ Morning:  Documentation + code quality
├─ Afternoon: Advanced testing + polish
└─ EOD: ✅ Production-ready, documented, tested

Total: 16-22 hours over 3 days
Parallel work possible: Testing can start Day 1 afternoon
```

---

## Next Steps After Completion

### Immediate (Day 4)

```
1. Integrate with prototype corpus processing
2. Test on 100 real Cambrian paper samples
3. Identify any missed patterns
4. Minor refinements if needed
```

### Short-term (Week 2)

```
5. Begin DAPT data preparation (uses normalization)
6. Create normalized JSONL corpus
7. Validate alignment maps on full corpus
8. Proceed to scripts/train_dapt.py implementation
```

### Medium-term (Week 3-4)

```
9. NER training data preparation (uses normalization)
10. RE training data preparation (uses normalization)
11. Inference pipeline integration
12. End-to-end validation
```

---

## References

- docs/NORMALIZATION_SPEC.md - Detailed specification
- OVERVIEW.md § 1.1 - DAPT corpus normalization requirements
- CLAUDE.md - Text normalization architecture notes
- devlog/20251028_001_architecture_review_and_phased_strategy.md - Phase 1 NER strategy

---

## Appendix: Example Session

### Input Text

```
"Olenellus wheeleri and Asaphiscus bonnensis occur in the Wheeler
Formation and Marjum Formation Middle Member, respectively. Both
are from House Range, Utah. These taxa characterize Cambrian Stage 10."
```

### Normalized Output

```
"Olenellus_wheeleri and Asaphiscus_bonnensis occur in the Wheeler_Formation
and Marjum_Formation Middle_Member, respectively. Both are from House_Range,
Utah. These taxa characterize Cambrian_Stage_10."
```

### Alignment Map (excerpt)

```python
{
    0: 0,    # 'O'
    ...
    9: 9,    # 's' in Olenellus
    10: 10,  # space → underscore
    11: 11,  # 'w' in wheeleri
    ...
    18: 18,  # 'i' (end of wheeleri)
    19: 19,  # space (unchanged)
    ...
    62: 62,  # space before Formation → underscore
    ...
}
```

### Span Projection Example

```python
# NER finds "Wheeler_Formation" in normalized text
norm_span = (67, 84)  # Indices in norm_text

# Project back to raw text
raw_span = project_span(norm_span, align_map, "norm_to_raw")
# raw_span = (66, 82)  # "Wheeler Formation" in raw_text

# Verify
assert raw_text[66:82] == "Wheeler Formation"
assert norm_text[67:84] == "Wheeler_Formation"
```

---

**Status:** READY TO START
**Owner:** To be assigned
**Blocker:** None
**Start Date:** TBD (after approval)
**Target Completion:** 3 working days from start
