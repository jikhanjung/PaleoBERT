# PaleoBERT Family Architecture

**Concept:** Model collection approach instead of single monolithic model
**Date:** 2025-10-29
**Status:** DRAFT proposal

---

## Executive Summary

Instead of building a single "PaleoBERT" covering all paleontology, develop a **family of specialized models** targeting specific geological periods and taxonomic groups. Each model shares the same architecture but has period/group-specific vocabulary and DAPT training.

**Advantages:**
- Smaller, more efficient models
- Independent development and versioning
- Better specialization and accuracy
- Easier maintenance and updates
- Multiple publication opportunities

---

## Proposed Model Family

### Core Architecture (Shared)

All models share:
- Base: DeBERTa-v3-base
- Pipeline: DAPT → NER → RE
- Entity types: TAXON, STRAT, CHRONO, LOC
- Relation types: occurs_in, found_at, part_of, assigned_to
- Text normalization approach
- Evaluation metrics

### Model Variants (Specialized)

```
PaleoBERT Family (v1.x series)
│
├─ PaleoBERT-Cambrian (v1.0)
│   ├─ Time: Cambrian Period (541-485 Ma)
│   ├─ Taxa: Trilobites, Brachiopods, Archaeocyaths, Burgess fauna
│   ├─ Vocabulary: ~400 tokens
│   ├─ DAPT corpus: Cambrian literature (~50M tokens)
│   └─ Status: PRIORITY (v1.0 target)
│
├─ PaleoBERT-Ordovician (v1.1)
│   ├─ Time: Ordovician Period (485-443 Ma)
│   ├─ Taxa: Graptolites, Crinoids, Bryozoans, Brachiopods, Early fish
│   ├─ Vocabulary: ~450 tokens
│   ├─ DAPT corpus: Ordovician literature (~40M tokens)
│   └─ Status: Future (after v1.0 success)
│
├─ PaleoBERT-Silurian-Devonian (v1.2)
│   ├─ Time: Silurian-Devonian (443-359 Ma)
│   ├─ Taxa: Fish radiation, Placoderms, Early tetrapods, Corals
│   ├─ Vocabulary: ~500 tokens
│   ├─ DAPT corpus: Silurian-Devonian literature (~50M tokens)
│   └─ Status: Future
│
├─ PaleoBERT-Carboniferous (v1.3)
│   ├─ Time: Carboniferous Period (359-299 Ma)
│   ├─ Taxa: Coal forest flora, Amphibians, Early amniotes
│   ├─ Vocabulary: ~450 tokens
│   ├─ DAPT corpus: Carboniferous literature (~30M tokens)
│   └─ Status: Future
│
├─ PaleoBERT-Mesozoic-Marine (v2.0)
│   ├─ Time: Mesozoic Era (252-66 Ma)
│   ├─ Taxa: Ichthyosaurs, Plesiosaurs, Mosasaurs, Ammonites, Belemnites
│   ├─ Vocabulary: ~500 tokens
│   ├─ DAPT corpus: Mesozoic marine literature (~80M tokens)
│   └─ Status: Future (different research community)
│
├─ PaleoBERT-Mesozoic-Terrestrial (v2.1)
│   ├─ Time: Mesozoic Era (252-66 Ma)
│   ├─ Taxa: Dinosaurs, Pterosaurs, Early mammals, Crocodilians
│   ├─ Vocabulary: ~600 tokens
│   ├─ DAPT corpus: Dinosaur literature (~100M tokens)
│   └─ Status: Future (HIGH IMPACT potential)
│
├─ PaleoBERT-Cenozoic-Mammals (v3.0)
│   ├─ Time: Cenozoic Era (66 Ma-present)
│   ├─ Taxa: Mammals (all orders), Birds, Flowering plants
│   ├─ Vocabulary: ~700 tokens
│   ├─ DAPT corpus: Cenozoic mammal literature (~80M tokens)
│   └─ Status: Future
│
└─ PaleoBERT-Plants (v4.0)
    ├─ Time: All periods (focus Devonian onwards)
    ├─ Taxa: Vascular plants, Bryophytes, Palynomorphs
    ├─ Vocabulary: ~500 tokens
    ├─ DAPT corpus: Paleobotany literature (~60M tokens)
    └─ Status: Future (specialized community)
```

---

## Comparison: Single vs Family

### Single Model Approach

```
PaleoBERT-Universal
├─ Coverage: All periods, all groups
├─ Vocabulary: 5,000-10,000 tokens
├─ DAPT corpus: 500M tokens (entire paleontology)
├─ Training time: 6-12 months
├─ Maintenance: Complex (any change affects everything)
├─ Publications: 1-2 papers
└─ User adoption: Slow (too general?)

Challenges:
❌ Vocabulary explosion (most tokens rarely used)
❌ Long development cycle
❌ Difficult to specialize
❌ Hard to maintain and update
❌ One-size-fits-all may fit none well
```

### Family Approach (Proposed)

```
PaleoBERT Family (8 models)
├─ Coverage: Same total coverage
├─ Vocabulary: 400-700 tokens per model (total 4,000)
├─ DAPT corpus: 40-100M tokens per model
├─ Training time: 2-4 months per model
├─ Maintenance: Independent versioning
├─ Publications: 8+ papers (one per model)
└─ User adoption: Fast (specialists choose their model)

Advantages:
✅ Efficient vocabulary usage
✅ Faster iteration per model
✅ Better specialization
✅ Easy to update independently
✅ Multiple publication opportunities
✅ Users download only what they need
```

---

## Technical Implementation

### Shared Codebase

```
PaleoBERT/
├─ core/
│   ├─ normalization.py       # Shared text normalization
│   ├─ models.py              # Shared NER/RE architectures
│   ├─ data_processing.py     # Shared data loaders
│   └─ metrics.py             # Shared evaluation
│
├─ configs/
│   ├─ cambrian.yaml          # Cambrian-specific config
│   ├─ ordovician.yaml        # Ordovician-specific config
│   ├─ mesozoic_marine.yaml   # Mesozoic Marine config
│   └─ ...
│
├─ vocabularies/
│   ├─ cambrian/
│   │   ├─ taxa.txt
│   │   ├─ strat_units.txt
│   │   ├─ chrono_units.txt
│   │   └─ localities.txt
│   ├─ ordovician/
│   │   └─ ... (separate vocabulary)
│   └─ ...
│
├─ scripts/
│   ├─ train_dapt.py          # Shared training script
│   ├─ train_ner.py           # Shared NER training
│   ├─ train_re.py            # Shared RE training
│   └─ infer_pipeline.py      # Shared inference
│
└─ models/
    ├─ cambrian/
    │   ├─ tokenizer_v1/
    │   ├─ dapt_v1/
    │   ├─ ner_v1/
    │   └─ re_v1/
    ├─ ordovician/
    │   └─ ... (separate model checkpoints)
    └─ ...
```

### Model Selection API

```python
from paleobert import load_model

# User selects appropriate model for their research
model = load_model("cambrian")  # For Cambrian papers
# or
model = load_model("mesozoic-terrestrial")  # For dinosaur papers

# Inference works the same for all models
results = model.extract(paper_text)
```

---

## Development Strategy

### Phase 1: Proof of Concept (6 months)

**Target:** PaleoBERT-Cambrian v1.0

- Scope: Cambrian trilobites + major groups
- Vocabulary: 400 tokens
- Corpus: 50M tokens Cambrian literature
- Milestones: M1-M5 (DAPT → NER → RE → Pipeline → Release)
- Deliverable: Working model + paper

**Success Criteria:**
- NER F1 ≥ 0.85 on Cambrian taxa
- RE F1 ≥ 0.60 on Cambrian relations
- User feedback positive

### Phase 2: Expansion (4 months per model)

If Phase 1 successful, develop additional models:

**Priority Order:**
1. **PaleoBERT-Ordovician** (natural continuation)
   - Reuse pipeline infrastructure
   - New vocabulary + DAPT
   - Paper 2

2. **PaleoBERT-Mesozoic-Terrestrial** (high impact)
   - Large user base (dinosaur research)
   - Well-funded area
   - Paper 3

3. **PaleoBERT-Mesozoic-Marine**
   - Complement terrestrial model
   - Paper 4

4. Others based on demand/funding

### Phase 3: Ecosystem (ongoing)

- Model zoo / Hugging Face hub
- User documentation per model
- Community contributions
- Regular updates

---

## Advantages Detailed

### 1. Scientific Advantages

**Specialization:**
- Each model learns period-specific language
- Better understanding of temporal context
- Formation names naturally cluster by period

**Example:**
```python
# Cambrian model knows:
"Wheeler Formation" → always Cambrian
"Burgess Shale" → Middle Cambrian, British Columbia

# Mesozoic model knows:
"Hell Creek Formation" → Late Cretaceous, Montana
"Morrison Formation" → Late Jurassic, Western US

# No confusion between periods!
```

**Accuracy:**
- Focused vocabulary → less ambiguity
- Period-specific DAPT → better context
- Specialized NER/RE → higher F1 scores

### 2. Practical Advantages

**Development Speed:**
- Each model: 2-4 months (not 12+ months)
- Parallel development possible
- Faster iteration based on feedback

**Resource Efficiency:**
- Smaller models → less VRAM
- Faster training → less GPU time
- Focused corpus → easier to collect

**User Experience:**
- Download only needed model (smaller)
- Faster inference (smaller vocabulary)
- Clear model selection (based on research period)

### 3. Publication Strategy

**Multiple Papers:**
- Paper 1: "PaleoBERT-Cambrian: NER for Cambrian Explosion"
- Paper 2: "PaleoBERT-Ordovician: Graptolite and Crinoid Extraction"
- Paper 3: "PaleoBERT-Mesozoic-Terrestrial: Dinosaur Literature Mining"
- Paper 4: "PaleoBERT Family: A Suite of Period-Specific Models"
- ...

**Each paper:**
- Novel contribution
- Different venues possible
- Different author teams possible
- Citations build over time

vs Single model:
- 1 big paper
- All-or-nothing publication
- Harder to get accepted (too broad?)

### 4. Community Building

**Specialists Contribute:**
- Cambrian experts → improve Cambrian model
- Dinosaur experts → improve Mesozoic-Terrestrial model
- Independent teams, independent timelines

**Modular Ecosystem:**
- New models can be added
- Community-driven expansion
- Sustainable long-term development

---

## Risks and Mitigations

### Risk 1: Code Duplication

**Risk:** Each model duplicates code
**Mitigation:** Shared core library + config files (already planned)

### Risk 2: Inconsistent Quality

**Risk:** Different models have different quality
**Mitigation:**
- Shared evaluation framework
- Consistent minimum criteria
- Regular benchmarking

### Risk 3: User Confusion

**Risk:** Users unsure which model to use
**Mitigation:**
- Clear documentation
- Model selection guide
- Auto-detection based on text (future)

### Risk 4: Maintenance Burden

**Risk:** Too many models to maintain
**Mitigation:**
- Prioritize high-impact models
- Community maintenance (open source)
- Deprecate low-use models if needed

---

## Recommended Next Steps

### Immediate (This Project)

1. **Confirm approach:** PaleoBERT-Cambrian as v1.0
2. **Document scope:** Update OVERVIEW.md with Cambrian focus
3. **Expand vocabulary:** 120 → 400 tokens (Cambrian-specific)
4. **Proceed with M1-M5:** DAPT through release

### Short-term (Next 6 months)

5. **Complete PaleoBERT-Cambrian v1.0**
6. **Publish results**
7. **Gather user feedback**
8. **Assess viability of family approach**

### Long-term (If successful)

9. **Launch PaleoBERT-Ordovician** (reuse pipeline)
10. **Build model zoo infrastructure**
11. **Engage paleontology community**
12. **Expand family based on demand**

---

## Comparison to Existing Approaches

### BioBERT Family Analogy

BioBERT spawned multiple domain models:
- BioBERT (general biomedical)
- BioBERT-PubMed (PubMed only)
- BioBERT-PMC (PMC only)
- ClinicalBERT (clinical notes)
- BioMedBERT (multiple variants)

**Success factors:**
- Clear domain boundaries
- Shared infrastructure
- Independent development
- Multiple research groups

**PaleoBERT can follow same pattern!**

### SciBERT as General Model

SciBERT tried to cover "all science":
- Physics, Chemistry, Biology, etc.
- Result: Good at general scientific text
- But: Specialized models (ChemBERT, MatBERT) outperform on specific domains

**Lesson:** Specialization wins for domain-specific tasks

---

## Conclusion

**Recommendation:** Adopt **PaleoBERT Family** approach

**v1.0 Focus:** PaleoBERT-Cambrian
- Prove the concept
- Establish pipeline
- Publish first model
- Assess expansion viability

**Long-term Vision:** Suite of period/group-specific models
- Better accuracy through specialization
- Sustainable development model
- Multiple publication opportunities
- Community-driven ecosystem

**Decision Point:** Confirm approach before proceeding with vocabulary expansion

---

## References

- BioBERT: Lee et al. (2020) "BioBERT: a pre-trained biomedical language representation model"
- SciBERT: Beltagy et al. (2019) "SciBERT: A Pretrained Language Model for Scientific Text"
- Domain-Adaptive Pretraining: Gururangan et al. (2020) "Don't Stop Pretraining"

---

**Status:** Awaiting stakeholder decision on family vs single model approach
**Next:** Update OVERVIEW.md based on decision
