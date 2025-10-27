# P01: Custom Tokenizer Setup with Domain-Specific Added Tokens

**Date:** 2025-10-27
**Status:** Planning
**Priority:** Critical (Foundation for all subsequent tasks)

## Overview

Build the custom tokenizer for PaleoBERT by extending DeBERTa-v3-base's tokenizer with domain-specific tokens from paleontology. This is the foundational task that must be completed before DAPT, NER, or RE training can begin.

## Rationale

Why this task is first:
1. **Dependency blocker**: All downstream tasks (DAPT, NER, RE) depend on a frozen tokenizer version
2. **Hard to change**: Tokenizer changes after DAPT require complete retraining (~20-30 hours)
3. **Core performance driver**: Added tokens reduce fragmentation of domain terminology, directly improving model understanding
4. **Version control**: Must establish `tokenizer_v1` baseline before any training begins

## Goals

### Primary Deliverables
1. Four added-token vocabulary files:
   - `artifacts/vocab/taxa.txt` - Taxonomic names (genera, species)
   - `artifacts/vocab/strat_units.txt` - Stratigraphic terms (Formation, Member, Group, etc.)
   - `artifacts/vocab/chrono_units.txt` - Chronostratigraphic units (Series_2, Stage_10, Paibian, etc.)
   - `artifacts/vocab/localities.txt` - Geographic localities

2. Custom tokenizer:
   - `artifacts/tokenizer_v1/` - Extended DeBERTa tokenizer with all added tokens
   - Configuration files (tokenizer_config.json, special_tokens_map.json)

3. Validation script:
   - `scripts/validate_tokenizer.py` - Test fragmentation rates and verify token additions

### Success Criteria
- [ ] All four vocabulary files created with at least 100 tokens each
- [ ] Tokenizer successfully loads and encodes sample paleontology text
- [ ] Fragmentation rate for domain terms < 20% (target: single-token encoding for most terms)
- [ ] Added tokens properly initialized in vocabulary (verify with tokenizer.get_vocab())
- [ ] Tokenizer serialized to `artifacts/tokenizer_v1/` and loadable via `AutoTokenizer.from_pretrained()`

## Task Breakdown

### Phase 1: Vocabulary Bootstrapping (4-6 hours)

**1.1 Research & Extract Base Vocabulary**
- [ ] Review paleontology literature sources for common terms
- [ ] Extract taxonomic names from:
  - Cambrian trilobite genera (Olenellus, Paradoxides, Asaphiscus, etc.)
  - Common fossil groups relevant to target literature
- [ ] Extract stratigraphic terms:
  - Formation/Member/Group suffixes
  - Common formation names (if specific corpus is known)
  - Lithology terms (Limestone, Shale, Sandstone)
- [ ] Extract chronostratigraphic units:
  - ICS stage names (Paibian, Jiangshanian, Drumian, Guzhangian)
  - Normalized forms: Series_2, Series_3, Stage_10, etc.
  - Cambrian-specific epochs/ages
- [ ] Extract geographic localities:
  - Well-known fossil localities
  - Geographic regions frequently mentioned

**1.2 Apply Normalization Rules**
- [ ] Implement underscore-binding for multi-token units:
  - `Stage 10` → `Stage_10`
  - `Series 2` → `Series_2`
  - `Cambrian Stage 10` → `Cambrian_Stage_10`
- [ ] Handle variants:
  - Roman numeral stages (I-X) → Arabic (1-10)
  - Hyphen/space variants (normalize to underscore)
- [ ] Enumerate realistic ranges:
  - Stage_1 through Stage_10
  - Series_2 through Series_4 (for Cambrian focus)
- [ ] Deduplicate and validate consistency

**1.3 Create Vocabulary Files**
- [ ] Write `artifacts/vocab/taxa.txt` (one term per line)
- [ ] Write `artifacts/vocab/strat_units.txt`
- [ ] Write `artifacts/vocab/chrono_units.txt`
- [ ] Write `artifacts/vocab/localities.txt`
- [ ] Document format and versioning in each file header

### Phase 2: Tokenizer Construction (2-3 hours)

**2.1 Load Base Tokenizer**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base",
    use_fast=True
)
print(f"Base vocab size: {len(tokenizer)}")
```

**2.2 Load and Merge Added Tokens**
```python
def load_vocab_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

taxa = load_vocab_file('artifacts/vocab/taxa.txt')
strat = load_vocab_file('artifacts/vocab/strat_units.txt')
chrono = load_vocab_file('artifacts/vocab/chrono_units.txt')
locs = load_vocab_file('artifacts/vocab/localities.txt')

custom_tokens = taxa + strat + chrono + locs
print(f"Adding {len(custom_tokens)} custom tokens")

num_added = tokenizer.add_tokens(custom_tokens)
print(f"Successfully added {num_added} new tokens")
print(f"New vocab size: {len(tokenizer)}")
```

**2.3 Save Custom Tokenizer**
```python
tokenizer.save_pretrained("artifacts/tokenizer_v1/")
```

**2.4 Verify Tokenizer Persistence**
- [ ] Reload tokenizer from saved path
- [ ] Verify vocab size matches
- [ ] Test encoding/decoding with domain terms

### Phase 3: Validation & Testing (2-3 hours)

**3.1 Create Validation Script**
`scripts/validate_tokenizer.py` should:
- [ ] Load tokenizer_v1
- [ ] Test encoding of sample paleontology sentences
- [ ] Calculate fragmentation rate for each vocabulary category
- [ ] Compare with base DeBERTa tokenizer (before added tokens)
- [ ] Generate report with metrics

**3.2 Fragmentation Analysis**
```python
def fragmentation_rate(tokenizer, terms):
    """Calculate % of terms that split into >1 token"""
    fragmented = 0
    for term in terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        if len(tokens) > 1:
            fragmented += 1
    return fragmented / len(terms) * 100

# Test on each category
print(f"Taxa fragmentation: {fragmentation_rate(tokenizer, taxa):.2f}%")
print(f"Strat fragmentation: {fragmentation_rate(tokenizer, strat):.2f}%")
# ... etc
```

**3.3 Sample Text Testing**
- [ ] Create `tests/data/sample_captions.txt` with real paleontology text
- [ ] Tokenize and inspect outputs
- [ ] Verify domain terms are single tokens
- [ ] Check special characters and punctuation handling

### Phase 4: Documentation (1 hour)

**4.1 Create Tokenizer README**
- [ ] `artifacts/tokenizer_v1/README.md` with:
  - Vocab size and breakdown by category
  - Added token counts (taxa: N, strat: M, chrono: K, loc: L)
  - Normalization rules applied
  - Example usage
  - Version and date

**4.2 Update Main Documentation**
- [ ] Add tokenizer setup section to README.md
- [ ] Document vocabulary sources and curation process
- [ ] Record any decisions or trade-offs made

## Technical Considerations

### Tokenizer Encoding Behavior
- DeBERTa uses SentencePiece-style tokenization
- Added tokens are treated as whole units (never split)
- Case sensitivity: preserve original case in vocabulary
- Whitespace handling: tokens should not include leading/trailing spaces

### Performance Impact
- Adding ~500-1000 tokens increases vocab by ~3%
- Embedding layer grows proportionally: `(vocab_size, hidden_dim)`
- For DeBERTa-v3-base (768 dim): ~0.5-1M additional parameters
- Memory impact negligible on 11GB GPU

### Versioning Strategy
- `tokenizer_v1`: Initial release, frozen post-DAPT
- If significant vocab changes needed later:
  - Create `tokenizer_v2`
  - Requires full DAPT retraining
  - Document migration path and compatibility

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Token explosion** (too many variants) | Diminishing returns, increased embedding size | Limit to realistic ranges; prefer patterns over exhaustive lists |
| **Inconsistent normalization** | Fragments still occur | Strict normalization rules; validation script catches issues |
| **Missing critical terms** | Poor performance on unseen entities | Iterative: can add tokens in v2, but requires retraining |
| **Case sensitivity errors** | Tokens don't match due to case | Decide on canonical case per category; document clearly |

## Dependencies

### Python Packages
```bash
pip install transformers==4.35.0  # or latest stable
pip install tokenizers>=0.14.0
```

### Data Sources (for vocabulary)
- ICS Chronostratigraphic Chart (for chrono units)
- Paleobiology Database (for taxa)
- USGS Geolex (for strat units, if US-focused)
- Manual curation from target corpus samples

## Timeline Estimate

- Phase 1 (Vocabulary): 4-6 hours
- Phase 2 (Construction): 2-3 hours
- Phase 3 (Validation): 2-3 hours
- Phase 4 (Documentation): 1 hour

**Total: 9-13 hours** (approximately 1.5-2 workdays)

## Next Steps After Completion

Once tokenizer_v1 is validated and frozen:
1. **P02: Corpus Collection & Preprocessing** - Gather ~100M tokens of domain text
2. **P03: DAPT Training Setup** - Configure training script for MLM pretraining
3. Model resize: `model.resize_token_embeddings(len(tokenizer))` before DAPT

## Open Questions

1. **Corpus preview**: Do we have sample paleontology text to extract real terms from?
2. **Geographic scope**: Global localities or region-specific (Antarctica, Laurentia, etc.)?
3. **Taxonomic scope**: Cambrian-focused or broader Paleozoic?
4. **Normalization strictness**: How aggressively to normalize (e.g., `Cambrian_Stage_10` vs just `Stage_10`)?

## References

- OVERVIEW.md § 1.2 (Added Tokens)
- OVERVIEW.md § 2.1 (Tokenizer setup code)
- OVERVIEW.md § 10 (Added-Token Tips)
- Hugging Face Tokenizers docs: https://huggingface.co/docs/transformers/main_classes/tokenizer
