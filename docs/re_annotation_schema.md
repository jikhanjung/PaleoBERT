# Relation Extraction (RE) Annotation Schema for PaleoBERT-Cambrian

**Version:** 1.0
**Date:** 2025-10-30
**Purpose:** Define relation types and annotation guidelines for extracting semantic relationships between paleontology entities
**Dependencies:** NER annotations (TAXON, STRAT, CHRONO, LOC entities)

---

## Overview

**Goal:** Extract structured semantic relationships from Cambrian paleontology text to build knowledge graphs of fossil occurrences, stratigraphic correlations, and geographic distributions.

**Approach:** Entity-pair classification with entity markers
- Input: Sentences with NER-annotated entities
- Output: Labeled relationships between entity pairs
- Model: DeBERTa with special tokens marking subject/object entities

---

## Relation Types

### 1. occurs_in

**Signature:** `TAXON → STRAT`

**Description:** A taxonomic entity (fossil organism) is found in or occurs within a stratigraphic unit.

**Examples:**
```
Text: "Olenellus gilberti occurs in the Wheeler Formation"
Triple: (Olenellus_gilberti, occurs_in, Wheeler_Formation)

Text: "The Pioche Shale yields diverse paradoxidids"
Triple: (paradoxidids, occurs_in, Pioche_Shale)

Text: "Elrathia kingii from the Wheeler Formation"
Triple: (Elrathia_kingii, occurs_in, Wheeler_Formation)
```

**Pattern Indicators:**
- "X occurs in Y"
- "X found in Y"
- "X collected from Y"
- "Y yields X"
- "Y contains X"
- "X from Y" (in fossil context)
- "X in Y" (when Y is stratigraphic unit)

**Negative Examples (NOT occurs_in):**
```
❌ "Olenellus in the House Range" → found_at (LOC, not STRAT)
❌ "Wheeler Formation in Stage 5" → assigned_to (STRAT→CHRONO, not TAXON→STRAT)
```

**Directionality:** TAXON (subject) → STRAT (object)

---

### 2. found_at

**Signature:** `TAXON → LOC`

**Description:** A taxonomic entity is discovered or found at a geographic locality.

**Examples:**
```
Text: "Olenellus wheeleri was found in the House Range, Utah"
Triple: (Olenellus_wheeleri, found_at, House_Range)

Text: "Paradoxides from Morocco"
Triple: (Paradoxides, found_at, Morocco)

Text: "The Burgess Shale fauna at Yoho National Park"
Triple: (Burgess_Shale_fauna, found_at, Yoho_National_Park)
```

**Pattern Indicators:**
- "X found at Y"
- "X from Y" (when Y is locality)
- "X in Y" (when Y is geographic location)
- "X at Y"
- "Y yields X" (when Y is locality)
- "specimens from Y"

**Negative Examples:**
```
❌ "Olenellus in the Wheeler Formation" → occurs_in (STRAT, not LOC)
❌ "Wheeler Formation at House Range" → This is STRAT→LOC, currently not modeled
```

**Directionality:** TAXON (subject) → LOC (object)

---

### 3. assigned_to

**Signature:** `STRAT → CHRONO`

**Description:** A stratigraphic unit is correlated, dated, or assigned to a chronostratigraphic/geochronologic time interval.

**Examples:**
```
Text: "The Wheeler Formation is assigned to Cambrian Stage 5"
Triple: (Wheeler_Formation, assigned_to, Cambrian_Stage_5)

Text: "Pioche Shale, Dyeran (Stage 4)"
Triple: (Pioche_Shale, assigned_to, Stage_4)

Text: "Middle Cambrian Burgess Shale"
Triple: (Burgess_Shale, assigned_to, Middle_Cambrian)
```

**Pattern Indicators:**
- "X assigned to Y"
- "X correlated to Y"
- "X dated to Y"
- "Y X" (chronology before formation: "Cambrian Wheeler Formation")
- "X of Y age" (e.g., "Pioche Shale of Stage 4 age")

**Negative Examples:**
```
❌ "Olenellus in Stage 5" → occurs_in (if implicit STRAT context)
❌ "Stage 5 fauna" → NO_RELATION (descriptive, not assignment)
```

**Directionality:** STRAT (subject) → CHRONO (object)

---

### 4. part_of

**Signature:** `STRAT → STRAT`

**Description:** A stratigraphic unit is a subdivision, member, or component of another stratigraphic unit.

**Examples:**
```
Text: "Wheeler Member of the Marjum Formation"
Triple: (Wheeler_Member, part_of, Marjum_Formation)

Text: "Middle Member within the Carrara Group"
Triple: (Middle_Member, part_of, Carrara_Group)

Text: "The Pioche Shale is part of the Highland Peak Formation"
Triple: (Pioche_Shale, part_of, Highland_Peak_Formation)
```

**Pattern Indicators:**
- "X Member of Y"
- "X Bed of Y"
- "X within Y"
- "X part of Y"
- "X in Y" (when hierarchical relationship)

**Negative Examples:**
```
❌ "Wheeler Formation and Marjum Formation" → NO_RELATION (co-occurrence, not hierarchy)
❌ "Wheeler Formation overlies Marjum Formation" → NO_RELATION (stratigraphic position, not containment)
```

**Directionality:** STRAT (smaller unit, subject) → STRAT (larger unit, object)

---

### 5. NO_RELATION

**Signature:** `ANY → ANY`

**Description:** No semantic relationship exists between the entity pair, or the relationship is not one of the four defined types.

**Examples:**
```
Text: "Olenellus and Paradoxides represent different evolutionary lineages"
Pair: (Olenellus, Paradoxides)
Label: NO_RELATION

Text: "The Wheeler Formation is exposed in the House Range"
Pair: (Wheeler_Formation, House_Range)
Label: NO_RELATION (STRAT→LOC not modeled)

Text: "Trilobites include Olenellus and paradoxidids"
Pair: (trilobites, Olenellus)
Label: NO_RELATION (taxonomic hierarchy not modeled)
```

**When to use:**
- Valid entity type pairs but no textual evidence of relationship
- Relationships not in our 4-class schema
- Entities appear in same sentence but are unrelated
- Distant entities (>15 tokens apart, typically)

**Importance:** Negative examples are crucial for training. They prevent the model from over-predicting relations.

---

## Valid Entity Type Pairs

### Allowed Combinations

| Subject Type | Object Type | Valid Relations |
|--------------|-------------|-----------------|
| TAXON | STRAT | occurs_in |
| TAXON | LOC | found_at |
| STRAT | CHRONO | assigned_to |
| STRAT | STRAT | part_of |
| TAXON | TAXON | NO_RELATION only |
| TAXON | CHRONO | NO_RELATION only* |
| CHRONO | CHRONO | NO_RELATION only |
| LOC | LOC | NO_RELATION only |
| Other combinations | NO_RELATION only |

\* *Note: TAXON→CHRONO (e.g., "Olenellus in Stage 5") is typically implicit via STRAT. If explicit, consider as occurs_in with implicit stratigraphic context.*

### Invalid Pairs (Do not annotate)

- Same entity paired with itself
- Entities from different sentences
- Overlapping entities

---

## Annotation Guidelines

### General Principles

1. **Textual Evidence Required:** Only annotate relations explicitly stated or strongly implied in the text
2. **Directionality Matters:** (A, rel, B) ≠ (B, rel, A)
3. **One Relation per Pair:** Each entity pair gets exactly one label (including NO_RELATION)
4. **Context Window:** Consider entities within same sentence (or adjacent sentences if clear reference)
5. **Multiple Relations:** If one subject relates to multiple objects, create separate pairs

### Annotation Process

**Step 1: Identify entity pairs**
- Extract all entities from NER annotations
- Generate candidate pairs based on valid type combinations
- Filter: entities must be within 20 tokens of each other

**Step 2: Classify relationship**
- Read sentence carefully
- Check for pattern indicators
- Determine most specific relation
- Default to NO_RELATION if uncertain

**Step 3: Mark entities**
- Subject entity: `[SUBJ] ... [/SUBJ]`
- Object entity: `[OBJ] ... [/OBJ]`

**Step 4: Validate**
- Correct entity types for relation?
- Textual evidence present?
- Directionality correct?

---

## Ambiguity Resolution

### Case 1: Implicit Stratigraphic Context

```
Text: "Olenellus occurs in Stage 5"
```

**Analysis:** Stage 5 is CHRONO, not STRAT. But geologically, this implies occurrence in strata of Stage 5 age.

**Resolution:**
- If STRAT entity also present: Use occurs_in with STRAT
- If only CHRONO: Mark as NO_RELATION (or create STRAT→CHRONO if appropriate)

### Case 2: Compound Relationships

```
Text: "Olenellus from the Wheeler Formation at House Range"
```

**Analysis:** Multiple relationships present.

**Resolution:** Create two separate pairs:
- (Olenellus, occurs_in, Wheeler_Formation)
- (Olenellus, found_at, House_Range)

### Case 3: Uncertain Directionality

```
Text: "The Pioche Shale and Cambrian Stage 4"
```

**Analysis:** Conjunction, unclear if assignment.

**Resolution:**
- If context suggests correlation: (Pioche_Shale, assigned_to, Stage_4)
- If just co-mention: NO_RELATION

### Case 4: Negative vs. Positive Sampling

```
Text: "Olenellus and Paradoxides are both trilobites"
```

**Analysis:** Both TAXON, no occurs_in/found_at relation.

**Resolution:** (Olenellus, NO_RELATION, Paradoxides)

**Sampling Strategy:** Include ~20-30% such pairs to prevent model from over-predicting positive relations.

---

## Edge Cases

### Geographic Ambiguity

```
Text: "Wheeler Formation in Utah"
```

**Issue:** Is this STRAT→LOC (formation location)?

**Resolution:** NO_RELATION (we don't model STRAT→LOC currently)

### Temporal Modifiers

```
Text: "Early Cambrian Olenellus"
```

**Issue:** Is this TAXON→CHRONO?

**Resolution:** NO_RELATION (descriptive modifier, not explicit correlation)

### List Constructions

```
Text: "Olenellus, Paradoxides, and Elrathia from the Wheeler Formation"
```

**Resolution:** Create three separate occurs_in relations:
- (Olenellus, occurs_in, Wheeler_Formation)
- (Paradoxides, occurs_in, Wheeler_Formation)
- (Elrathia, occurs_in, Wheeler_Formation)

### Nested Stratigraphic Units

```
Text: "Wheeler Member of the Marjum Formation from Stage 5"
```

**Resolution:** Create two relations:
- (Wheeler_Member, part_of, Marjum_Formation)
- (Marjum_Formation, assigned_to, Stage_5)

---

## Data Format

### Input Format (Entity Markers)

```json
{
  "text": "Olenellus gilberti occurs in the Wheeler Formation",
  "marked_text": "[SUBJ] Olenellus gilberti [/SUBJ] occurs in the [OBJ] Wheeler Formation [/OBJ]",
  "subject": {
    "id": "e1",
    "type": "TAXON",
    "text": "Olenellus gilberti",
    "start": 0,
    "end": 18
  },
  "object": {
    "id": "e2",
    "type": "STRAT",
    "text": "Wheeler Formation",
    "start": 37,
    "end": 54
  },
  "relation": "occurs_in",
  "label_id": 1,
  "metadata": {
    "doc_id": "Webster_2011",
    "sent_id": 42,
    "confidence": 1.0
  }
}
```

### Label Encoding

```python
LABEL_MAP = {
    "NO_RELATION": 0,
    "occurs_in": 1,
    "found_at": 2,
    "assigned_to": 3,
    "part_of": 4
}
```

### JSONL Format

Each line is one entity pair:

```jsonl
{"text": "Olenellus occurs in Wheeler Formation", "marked_text": "[SUBJ]Olenellus[/SUBJ] occurs in [OBJ]Wheeler Formation[/OBJ]", "subject": {"type": "TAXON", "text": "Olenellus"}, "object": {"type": "STRAT", "text": "Wheeler Formation"}, "relation": "occurs_in", "label_id": 1}
{"text": "Paradoxides from Morocco", "marked_text": "[SUBJ]Paradoxides[/SUBJ] from [OBJ]Morocco[/OBJ]", "subject": {"type": "TAXON", "text": "Paradoxides"}, "object": {"type": "LOC", "text": "Morocco"}, "relation": "found_at", "label_id": 2}
```

---

## Negative Sampling Strategy

### Purpose

- Prevent model from predicting relations where none exist
- Balance dataset (avoid overwhelming positive bias)
- Teach model to distinguish meaningful from accidental co-occurrences

### Sampling Methods

**1. Random Pairs (Type-Valid)**
- Generate pairs of valid type combinations
- Sample from sentences with multiple entities
- Exclude pairs matching positive patterns

**2. Distance-Based**
- Entities >15 tokens apart unlikely to relate
- Use as NO_RELATION examples

**3. Type-Invalid Pairs**
- TAXON→TAXON, LOC→LOC, etc.
- Always NO_RELATION

**4. Cross-Sentence**
- Entities from adjacent sentences
- Controlled negative examples

### Target Ratio

**Recommended:** 1:2 to 1:4 (positive:negative)
- Positive relations: 20-40K
- NO_RELATION: 40-120K
- Total: 60-160K pairs

**Rationale:** Real text has more non-relations than relations. Over-sampling positives leads to false positives.

---

## Evaluation Metrics

### Per-Relation Metrics

For each relation type (occurs_in, found_at, assigned_to, part_of, NO_RELATION):

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Overall Metrics

**Micro-averaged F1:**
```
Sum TP, FP, FN across all classes
F1_micro = 2 × TP / (2×TP + FP + FN)
```

**Macro-averaged F1:**
```
F1_macro = mean(F1_occurs_in, F1_found_at, F1_assigned_to, F1_part_of, F1_NO_RELATION)
```

**Accuracy:**
```
Accuracy = (Correct predictions) / (Total predictions)
```

### Success Criteria (from CLAUDE.md)

**Overall:**
- Micro-F1 ≥ 0.75
- Macro-F1 ≥ 0.70
- Accuracy ≥ 0.85

**Per-Relation F1 Targets:**
- occurs_in ≥ 0.80 (most common, must be reliable)
- found_at ≥ 0.70
- assigned_to ≥ 0.70
- part_of ≥ 0.70
- NO_RELATION precision ≥ 0.80 (avoid false positives)

### Confusion Matrix

Track common errors:
```
                Predicted
              occ  fnd  asg  prt  NO
      occ  [  X    ?    ?    ?    ? ]
Actual fnd  [  ?    X    ?    ?    ? ]
      asg  [  ?    ?    X    ?    ? ]
      prt  [  ?    ?    ?    X    ? ]
      NO   [  ?    ?    ?    ?    X ]
```

---

## Example Annotated Pairs

### Example 1: occurs_in (TAXON → STRAT)

```
Text: "Olenellus gilberti occurs in the Wheeler Formation"

Marked: "[SUBJ] Olenellus gilberti [/SUBJ] occurs in the [OBJ] Wheeler Formation [/OBJ]"

Annotation:
  Subject: Olenellus gilberti (TAXON)
  Object: Wheeler Formation (STRAT)
  Relation: occurs_in
  Evidence: "occurs in" (explicit pattern)
  Label: 1
```

### Example 2: found_at (TAXON → LOC)

```
Text: "Paradoxides davidis from Morocco represents Middle Cambrian fauna"

Marked: "[SUBJ] Paradoxides davidis [/SUBJ] from [OBJ] Morocco [/OBJ] represents Middle Cambrian fauna"

Annotation:
  Subject: Paradoxides davidis (TAXON)
  Object: Morocco (LOC)
  Relation: found_at
  Evidence: "from" (locality context)
  Label: 2
```

### Example 3: assigned_to (STRAT → CHRONO)

```
Text: "The Pioche Shale is dated to Cambrian Stage 4"

Marked: "The [SUBJ] Pioche Shale [/SUBJ] is dated to [OBJ] Cambrian Stage 4 [/OBJ]"

Annotation:
  Subject: Pioche Shale (STRAT)
  Object: Cambrian Stage 4 (CHRONO)
  Relation: assigned_to
  Evidence: "is dated to" (explicit correlation)
  Label: 3
```

### Example 4: part_of (STRAT → STRAT)

```
Text: "Wheeler Member of the Marjum Formation"

Marked: "[SUBJ] Wheeler Member [/SUBJ] of the [OBJ] Marjum Formation [/OBJ]"

Annotation:
  Subject: Wheeler Member (STRAT)
  Object: Marjum Formation (STRAT)
  Relation: part_of
  Evidence: "Member of" (hierarchical pattern)
  Label: 4
```

### Example 5: NO_RELATION

```
Text: "Olenellus and Paradoxides are both trilobites from the Cambrian"

Marked: "[SUBJ] Olenellus [/SUBJ] and [OBJ] Paradoxides [/OBJ] are both trilobites from the Cambrian"

Annotation:
  Subject: Olenellus (TAXON)
  Object: Paradoxides (TAXON)
  Relation: NO_RELATION
  Evidence: TAXON→TAXON pair, co-mention but no occurs_in/found_at
  Label: 0
```

---

## Annotation Workflow

### Stage 1: Automatic Pair Generation

```python
# From NER-annotated sentences
for sentence in ner_corpus:
    entities = sentence.entities

    for subj in entities:
        for obj in entities:
            if subj != obj:
                # Check type compatibility
                if is_valid_pair(subj.type, obj.type):
                    # Check distance
                    if token_distance(subj, obj) <= 20:
                        pairs.append((subj, obj, sentence))
```

### Stage 2: Pattern-Based Labeling

```python
# Apply regex patterns to assign initial labels
patterns = {
    "occurs_in": [r"(\[SUBJ\].*?\[/SUBJ\]).*?(occurs?|found|collected).*?in.*?(\[OBJ\].*?\[/OBJ\])"],
    "found_at": [r"(\[SUBJ\].*?\[/SUBJ\]).*?(from|at).*?(\[OBJ\].*?\[/OBJ\])"],
    # ... more patterns
}

for pair in pairs:
    marked_text = insert_markers(pair.sentence, pair.subject, pair.object)

    for relation, pattern_list in patterns.items():
        if any(re.search(p, marked_text) for p in pattern_list):
            pair.relation = relation
            break
    else:
        # No pattern matched
        pair.relation = "NO_RELATION"
```

**Expected Accuracy:** 50-70% (pattern-based)

### Stage 3: Manual Review (Optional)

1. Sample 1000-2000 pairs for manual annotation
2. Correct auto-labeled relations
3. Calculate accuracy of auto-labeling
4. Refine patterns if needed
5. Use as gold-standard test set

### Stage 4: Data Split

```python
# Stratified split to maintain class balance
train, dev, test = stratified_split(
    pairs,
    ratios=[0.8, 0.1, 0.1],
    stratify_by="relation"
)
```

---

## Quality Control

### Inter-Annotator Agreement

**Target:** Cohen's κ ≥ 0.80 (substantial agreement)

**Procedure:**
1. Two annotators label same 500 pairs independently
2. Calculate κ statistic
3. Resolve disagreements through discussion
4. Update guidelines based on disagreements

### Label Distribution

**Monitor class balance:**
```
NO_RELATION: 60-70%
occurs_in:   15-20%
found_at:    5-10%
assigned_to: 5-10%
part_of:     2-5%
```

**Warning:** If any positive relation < 2%, increase pattern coverage or manual annotation.

### Error Analysis

Track common errors during validation:
- Pattern false positives
- Type mismatches (NER errors propagated)
- Ambiguous cases (annotator disagreement)
- Edge cases requiring guideline updates

---

## Notes for Annotators

1. **Read full sentence context:** Don't judge relations from marked text alone
2. **When uncertain, choose NO_RELATION:** Conservative labeling prevents noise
3. **Check entity types first:** Invalid type pairs should always be NO_RELATION
4. **Consider geological semantics:** "from" can indicate occurrence or locality depending on object type
5. **Document ambiguous cases:** Flag for team discussion and guideline refinement
6. **Use vocabulary lists:** Verify entity types against taxa.txt, strat_units.txt, etc.

---

## Troubleshooting

### Issue: Low occurs_in recall

**Symptom:** Many TAXON→STRAT pairs labeled NO_RELATION incorrectly

**Solutions:**
- Expand occurs_in patterns (add more verbs, prepositions)
- Check for implicit relations ("X, Y" can mean "X from Y")
- Review NER annotations (missed STRAT entities?)

### Issue: High NO_RELATION false positives

**Symptom:** Model predicts positive relations where none exist

**Solutions:**
- Increase negative sampling ratio
- Add more distance-based negatives
- Include type-invalid pairs as negatives

### Issue: Confusion between occurs_in and found_at

**Symptom:** TAXON→STRAT vs. TAXON→LOC confusion

**Solutions:**
- Verify NER entity types (is "House Range" STRAT or LOC?)
- Check object type explicitly before labeling
- Add clarifying examples to guidelines

---

## Version History

- **v1.0 (2025-10-30):** Initial schema for Cambrian trilobite RE
  - Four positive relation types: occurs_in, found_at, assigned_to, part_of
  - NO_RELATION for negative examples
  - Entity marker approach
  - Pattern-based annotation with manual review option

---

## References

**Related Documentation:**
- NER Annotation Schema: `docs/ner_annotation_schema.md`
- P07 RE Preparation Plan: `devlog/20251030_P07_re_preparation.md`
- CLAUDE.md: Project guidelines and success criteria

**Methodology:**
- Entity markers for relation extraction (proven effective in literature)
- Pattern-based bootstrapping followed by neural learning
- Class-balanced training with weighted loss

---

**Contact:** PaleoBERT Development Team
**Project:** PaleoBERT-Cambrian v1.0
**Milestone:** M3 (Relation Extraction)
**Status:** Schema Complete, Implementation Pending
