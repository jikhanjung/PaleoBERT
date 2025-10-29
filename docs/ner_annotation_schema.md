# NER Annotation Schema for PaleoBERT-Cambrian

**Version:** 1.0
**Date:** 2025-10-30
**Purpose:** Define entity types and annotation guidelines for Named Entity Recognition in Cambrian trilobite paleontology text

---

## Entity Types

### 1. TAXON (Taxonomic Names)
**Description:** Scientific names of organisms at any taxonomic rank

**Examples:**
- Genus: `Olenellus`, `Paradoxides`, `Elrathia`
- Species: `Olenellus gilberti`, `Paradoxides davidis`
- Family: `Olenellidae`, `Paradoxididae`
- Order: `Redlichiida`, `Ptychopariida`
- Class: `Trilobita`

**Annotation Rules:**
- Include full binomial names as single entity
- Include author citations if present: `Olenellus gilberti Meek, 1874`
- Include informal taxonomic groups: `olenellids`, `paradoxidids`
- Exclude common names unless they map to specific taxa

**BIO Tags:**
```
Olenellus    B-TAXON
gilberti     I-TAXON
was          O
found        O
```

### 2. STRAT (Stratigraphic Units)
**Description:** Formal and informal stratigraphic unit names

**Examples:**
- Formation: `Wheeler Formation`, `Burgess Shale`
- Member: `Middle Member`, `Pioche Member`
- Group: `Carrara Group`
- Informal: `lower shale unit`, `middle limestone`

**Annotation Rules:**
- Include unit type: `Wheeler Formation` (full entity)
- Include geographic qualifiers: `Lower Cambrian Series`
- Bind with underscores in normalized text: `Wheeler_Formation`

**BIO Tags:**
```
Wheeler      B-STRAT
Formation    I-STRAT
contains     O
trilobites   O
```

### 3. CHRONO (Chronostratigraphic Units)
**Description:** Time-stratigraphic units and geochronologic terms

**Examples:**
- Stage: `Cambrian Stage 10`, `Jiangshanian`, `Stage 4`
- Series: `Cambrian Series 2`, `Furongian`
- System: `Cambrian`, `Ordovician`
- Epoch: `Early Cambrian`, `Middle Cambrian`
- Age (geochronologic): `Wuliuan Age`

**Annotation Rules:**
- Include numeric stages: `Stage 10`, `Series 2`
- Bind multi-word units: `Cambrian_Stage_10`
- Include both formal and informal time terms
- "Lower/Middle/Upper" are stratigraphic; "Early/Middle/Late" are geochronologic

**BIO Tags:**
```
Cambrian     B-CHRONO
Stage        I-CHRONO
10           I-CHRONO
is           O
the          O
latest       O
```

### 4. LOC (Geographic Localities)
**Description:** Geographic locations where fossils are found

**Examples:**
- Specific localities: `Wheeler Amphitheater`, `House Range`
- Regions: `Great Basin`, `western Utah`
- Countries: `United States`, `Morocco`
- Mountains: `Wasatch Mountains`

**Annotation Rules:**
- Include full locality names
- Geographic coordinates are NOT entities (metadata)
- Modern political boundaries: annotate as LOC
- Paleogeographic terms (e.g., "Laurentia"): annotate as LOC

**BIO Tags:**
```
Specimens    O
from         O
the          O
House        B-LOC
Range        I-LOC
in           O
Utah         B-LOC
```

---

## BIO Tagging Scheme

**Format:** `B-TYPE`, `I-TYPE`, `O`

- **B-TYPE**: Beginning of entity
- **I-TYPE**: Inside/continuation of entity
- **O**: Outside any entity (regular token)

**Multi-word Entities:**
```
Text:  "Olenellus gilberti from Wheeler Formation"
Tags:  B-TAXON I-TAXON O B-STRAT I-STRAT
```

**Adjacent Entities:**
```
Text:  "Cambrian Stage 10 trilobites"
Tags:  B-CHRONO I-CHRONO I-CHRONO B-TAXON
```

---

## Annotation Guidelines

### General Principles

1. **Maximize Coverage:** Annotate all entities, even informal mentions
2. **Context Matters:** Use sentence context to resolve ambiguity
3. **Consistency:** Same term → same label throughout dataset
4. **Nested Entities:** Not allowed; choose most specific type
5. **Abbreviations:** Annotate as full entity type

### Ambiguity Resolution

**Case 1: Formation names vs. Localities**
```
"Wheeler Formation" → STRAT (the rock unit)
"Wheeler Amphitheater" → LOC (the place)
"Wheeler Shale" → STRAT (lithologic unit)
```

**Case 2: Time vs. Strat**
```
"Cambrian Series 2" → CHRONO (time unit)
"Lower Cambrian strata" → O (descriptive, not formal unit)
```

**Case 3: Taxonomic vs. General**
```
"trilobites" → TAXON (refers to Class Trilobita)
"the trilobite" → O (generic reference)
"Olenellus" → TAXON (always)
```

### Edge Cases

**Adjectival Forms:**
```
"olenellid faunas" → "olenellid" is B-TAXON (informal taxon)
"trilobite diversity" → "trilobite" is O (descriptive)
```

**Quoted Names:**
```
"Olenellus" gilberti → Both are part of TAXON
cf. Olenellus → "cf." is O, "Olenellus" is B-TAXON
```

**Numbers in Names:**
```
Cambrian Stage 10 → All three tokens are CHRONO
Stage 4 → Both tokens are CHRONO
```

---

## Data Format

### JSONL Format (CoNLL-style)

Each line is a JSON object representing one sentence:

```json
{
  "text": "Olenellus gilberti occurs in the Wheeler Formation.",
  "tokens": ["Olenellus", "gilberti", "occurs", "in", "the", "Wheeler", "Formation", "."],
  "ner_tags": ["B-TAXON", "I-TAXON", "O", "O", "O", "B-STRAT", "I-STRAT", "O"],
  "metadata": {
    "doc_id": "Webster_2011",
    "sent_id": 42
  }
}
```

### Alternative: Token-per-line Format

```
Olenellus    B-TAXON
gilberti     I-TAXON
occurs       O
in           O
the          O
Wheeler      B-STRAT
Formation    I-STRAT
.            O

```

Blank line separates sentences.

---

## Label Statistics (Target Distribution)

Based on paleontology literature analysis:

| Entity Type | Approx. Frequency | Examples per 1000 tokens |
|-------------|-------------------|--------------------------|
| TAXON       | 40-50%            | 15-20                    |
| STRAT       | 25-30%            | 8-12                     |
| CHRONO      | 15-20%            | 5-8                      |
| LOC         | 10-15%            | 3-5                      |

**Target for training data:**
- Minimum: 5,000 annotated sentences
- Target: 10,000-20,000 sentences
- Expected entities: ~50,000-200,000 entity mentions

---

## Annotation Workflow

### Stage 1: Pre-annotation
1. Use existing vocabulary lists to auto-tag known entities
2. Apply regex patterns for common formats (e.g., "Formation", "Stage X")
3. Generate initial BIO tags with ~60-70% accuracy

### Stage 2: Manual Review
1. Review auto-annotated sentences
2. Correct errors and add missed entities
3. Handle edge cases and ambiguities
4. Mark difficult cases for discussion

### Stage 3: Validation
1. Inter-annotator agreement check (κ ≥ 0.85 target)
2. Consistency review across document
3. Edge case resolution
4. Final quality check

---

## Quality Metrics

**Inter-Annotator Agreement:**
- Cohen's κ ≥ 0.85 (substantial agreement)
- Per-entity F1 ≥ 0.90

**Coverage Metrics:**
- Sentence coverage: 100% of sentences reviewed
- Entity density: 3-6 entities per sentence (average)
- Label balance: No entity type < 5% of total

**Error Analysis:**
- Track common error types
- Document ambiguous cases
- Update guidelines as needed

---

## Example Annotated Sentences

```
1. Simple taxonomic mention:
   Text: "Olenellus is common in the Lower Cambrian."
   Tags: [B-TAXON] [O] [O] [O] [O] [B-CHRONO] [I-CHRONO] [O]

2. Multiple entity types:
   Text: "Paradoxides from the Wheeler Formation represent Stage 5."
   Tags: [B-TAXON] [O] [O] [B-STRAT] [I-STRAT] [O] [B-CHRONO] [I-CHRONO] [O]

3. Complex stratigraphic context:
   Text: "The Pioche Shale of the House Range contains diverse trilobites."
   Tags: [O] [B-STRAT] [I-STRAT] [O] [O] [B-LOC] [I-LOC] [O] [O] [B-TAXON] [O]

4. Chronostratigraphic detail:
   Text: "Cambrian Series 2 Stage 4 olenellids are widespread."
   Tags: [B-CHRONO] [I-CHRONO] [I-CHRONO] [I-CHRONO] [I-CHRONO] [B-TAXON] [O] [O] [O]
```

---

## Notes for Annotators

1. **When in doubt, annotate:** It's easier to remove entities than add them later
2. **Context is key:** Read full sentence, not just isolated tokens
3. **Be consistent:** Same entity, same label, every time
4. **Document questions:** Flag ambiguous cases for group discussion
5. **Use vocabulary lists:** Check against taxa.txt, strat_units.txt, etc.

---

## Version History

- **v1.0 (2025-10-30):** Initial schema for Cambrian trilobite NER
  - Four entity types: TAXON, STRAT, CHRONO, LOC
  - BIO tagging scheme
  - Guidelines and examples

---

**Contact:** PaleoBERT Development Team
**Project:** PaleoBERT-Cambrian v1.0
**Milestone:** M2 (Named Entity Recognition)
