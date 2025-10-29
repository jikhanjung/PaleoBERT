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

## Router Architecture (Phase 3+)

### Motivation

Once multiple models exist (Cambrian, Ordovician, Mesozoic, etc.), users need a way to automatically select the appropriate model for their text. A **Router** system provides intelligent model selection based on input text analysis.

### Router Concept

```
                    ┌─────────────┐
User Input Text →   │   Router    │  (Automatic classifier)
"Olenellus from     │  Classifier │
 Wheeler Fm..."     └─────────────┘
                           ↓
                    Analyzes text:
                    - Detects period keywords
                    - Identifies taxonomic groups
                    - Recognizes formations
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │PaleoBERT-   │ │PaleoBERT-   │ │PaleoBERT-   │
      │Cambrian     │ │Ordovician   │ │Mesozoic     │
      └─────────────┘ └─────────────┘ └─────────────┘
              ↓
        NER + RE Results
```

### Implementation Options

#### Option 1: Keyword-Based Router (Simple) ✅

**Best for:** Initial implementation (2-3 models)

```python
class KeywordRouter:
    """Simple keyword matching for model selection"""

    def __init__(self):
        self.period_keywords = {
            "cambrian": [
                "Cambrian", "Olenellus", "Burgess", "Wheeler",
                "Stage_10", "Terreneuvian", "Asaphiscus", "Elrathia"
            ],
            "ordovician": [
                "Ordovician", "Graptolites", "Crinoid", "Cincinnatian",
                "Tremadocian", "Caradoc"
            ],
            "mesozoic-terrestrial": [
                "Tyrannosaurus", "Cretaceous", "Jurassic", "Hell_Creek",
                "Morrison", "Dinosaur", "Pterosaur"
            ],
        }

    def route(self, text):
        """Score each model by keyword matches"""
        scores = {}
        for period, keywords in self.period_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[period] = score

        # Select model with highest score
        if max(scores.values()) == 0:
            # No keywords found - use default
            return "cambrian"  # or prompt user

        best_period = max(scores, key=scores.get)
        return self.load_model(best_period)

    def extract(self, text):
        """Route to appropriate model and extract"""
        model = self.route(text)
        return model.extract(text)
```

**Pros:**
- Fast (~1ms routing overhead)
- Simple to implement
- Explainable (can show which keywords matched)
- Easy to maintain

**Cons:**
- Brittle on edge cases
- Requires manual keyword curation
- May fail on ambiguous text

**Accuracy:** ~85-90% for clear cases

---

#### Option 2: ML-Based Router (Advanced) 🎯

**Best for:** 4+ models, production deployment

```python
class MLRouter:
    """Learned classifier for model selection"""

    def __init__(self):
        # Small BERT classifier (multi-class)
        # Input: text (first 200-300 tokens)
        # Output: model ID (0-7 for 8 models)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            "paleobert-router-v1",
            num_labels=8  # Number of models in family
        )
        self.tokenizer = AutoTokenizer.from_pretrained("paleobert-router-v1")

    def route(self, text):
        """Classify text to select appropriate model"""
        # Take first 512 tokens for classification
        inputs = self.tokenizer(text[:2000], return_tensors="pt", truncation=True)

        # Classify
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits

        # Get model with highest confidence
        model_id = logits.argmax().item()
        confidence = torch.softmax(logits, dim=1).max().item()

        models = ["cambrian", "ordovician", "silurian-devonian",
                  "carboniferous", "mesozoic-marine", "mesozoic-terrestrial",
                  "cenozoic-mammals", "plants"]

        selected_model = models[model_id]

        # Log routing decision
        print(f"Router selected: {selected_model} (confidence: {confidence:.2f})")

        return self.load_model(selected_model)
```

**Training Data:**

```python
# Collect training samples from each model's DAPT corpus
training_data = []

# Cambrian samples
for doc in cambrian_corpus.sample(1000):
    training_data.append((doc.text[:500], "cambrian"))

# Ordovician samples
for doc in ordovician_corpus.sample(1000):
    training_data.append((doc.text[:500], "ordovician"))

# ... for each model

# Train small classifier
trainer.train(training_data)
```

**Pros:**
- High accuracy (~95%+)
- Learns complex patterns
- Handles ambiguous cases
- Improves with more data

**Cons:**
- Requires training data
- Additional model to maintain
- Slightly slower (~10-50ms routing)

**Accuracy:** ~95-98%

---

#### Option 3: Ensemble Router (Maximum Accuracy) 🚀

**Best for:** Critical applications, research benchmarking

```python
class EnsembleRouter:
    """Run all models, select best by confidence"""

    def __init__(self):
        self.models = {
            "cambrian": load_model("paleobert-cambrian-v1"),
            "ordovician": load_model("paleobert-ordovician-v1"),
            "mesozoic-terrestrial": load_model("paleobert-mesozoic-terrestrial-v1"),
            # ... load all available models
        }

    def route_and_extract(self, text):
        """Run all models in parallel, return best result"""
        results = {}

        # Run all models (can be parallelized)
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                executor.submit(model.extract, text): name
                for name, model in self.models.items()
            }

            for future in as_completed(futures):
                model_name = futures[future]
                result = future.result()
                results[model_name] = result

        # Select result with highest average confidence
        best_model = max(
            results.keys(),
            key=lambda m: results[m].get('avg_confidence', 0)
        )

        print(f"Ensemble selected: {best_model}")
        return results[best_model]
```

**Pros:**
- Maximum accuracy (~99%+)
- No routing errors (all models tried)
- Confidence-based selection
- Handles all edge cases

**Cons:**
- Slow (N× inference time)
- High resource usage (all models in memory)
- Expensive for batch processing

**Accuracy:** ~99%+ (best result from N models)

---

### Comparison Matrix

| Feature | Keyword | ML-Based | Ensemble |
|---------|---------|----------|----------|
| **Accuracy** | ~85-90% | ~95-98% | ~99% |
| **Speed** | ~1ms | ~10-50ms | ~N×inference |
| **Resource** | Minimal | Low | High |
| **Maintenance** | Manual | Automatic | Minimal |
| **Complexity** | Simple | Medium | Complex |
| **Best for** | 2-3 models | 4+ models | Benchmarks |

---

### Development Roadmap

#### Phase 1 (v1.0): No Router

```
PaleoBERT-Cambrian only
└─ Users explicitly load: load_model("cambrian")
```

#### Phase 2 (v2.0): Simple Keyword Router

```
2-3 models available (Cambrian, Ordovician)
└─ Keyword-based router
└─ Users can use: PaleoBERT().extract(text)  # auto-routes
```

#### Phase 3 (v3.0): ML-Based Router

```
4+ models available
└─ Train ML classifier on corpus samples
└─ High accuracy automatic routing
```

#### Phase 4 (Research): Ensemble

```
For critical applications / benchmarking
└─ Optional ensemble mode
└─ Maximum accuracy
```

---

### User API (Future)

```python
from paleobert import PaleoBERT

# Simple API - router handles everything
extractor = PaleoBERT()  # Loads router + all models

# Example 1: Cambrian paper
text1 = "Olenellus wheeleri from the Wheeler Formation..."
result1 = extractor.extract(text1)
# Router detects: Cambrian → uses PaleoBERT-Cambrian
# Result: {entities: [...], relations: [...], model_used: "cambrian"}

# Example 2: Mesozoic paper
text2 = "Tyrannosaurus rex from Hell Creek Formation..."
result2 = extractor.extract(text2)
# Router detects: Mesozoic → uses PaleoBERT-Mesozoic-Terrestrial
# Result: {entities: [...], relations: [...], model_used: "mesozoic-terrestrial"}

# Users don't need to know which model to use!
```

**Advanced options:**

```python
# Override router decision
extractor = PaleoBERT()
result = extractor.extract(text, force_model="cambrian")

# See routing decision
result = extractor.extract(text, explain_routing=True)
# Returns: {
#   "entities": [...],
#   "routing": {
#     "selected_model": "cambrian",
#     "confidence": 0.95,
#     "alternatives": {"ordovician": 0.03, "mesozoic": 0.02}
#   }
# }

# Use ensemble mode (slow but accurate)
result = extractor.extract(text, mode="ensemble")
```

---

### Implementation Priority

**NOT IMMEDIATE** - Focus on PaleoBERT-Cambrian v1.0 first.

**Router needed when:**
- ✅ 2+ models exist
- ✅ Users request automatic model selection
- ✅ Batch processing of mixed-period papers

**Estimated timeline:**
- After v2.0 (Cambrian + Ordovician both exist)
- ~1-2 weeks to implement keyword router
- ~4-6 weeks to implement ML-based router (including training)

---

## References

- BioBERT: Lee et al. (2020) "BioBERT: a pre-trained biomedical language representation model"
- SciBERT: Beltagy et al. (2019) "SciBERT: A Pretrained Language Model for Scientific Text"
- Domain-Adaptive Pretraining: Gururangan et al. (2020) "Don't Stop Pretraining"

---

**Status:** ✅ Family approach confirmed, PaleoBERT-Cambrian v1.0 in development
**Next:** Complete vocabulary expansion (120 → 400 tokens)
