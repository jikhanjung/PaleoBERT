# PaleoBERT Project Scope Definition

**Status:** DRAFT - Requires decision
**Date:** 2025-10-29

---

## Current Implicit Scope (v1.0)

Based on sample vocabulary in `artifacts/vocab/`:

**Geological Time:**
- Primary: Cambrian Period (541-485 Ma)
- Secondary: Ordovician-Devonian (limited coverage)

**Taxa Coverage:**
- Primary: Trilobites (Arthropoda)
  - Orders: Agnostida, Redlichiida, Ptychopariida, Corynexochida
  - ~30 genera/species in sample
- Secondary: None explicitly included

**Geographic Coverage:**
- Primary: North America (Utah, Nevada, British Columbia)
- Secondary: Europe (England), Asia (China) - limited

**Stratigraphic Units:**
- ~12 specific formations (Wheeler, Marjum, Burgess, Spence, etc.)
- Generic terms (Formation, Member, Shale, Limestone, etc.)

**Limitations:**
- Trilobite-centric (no other major groups)
- Cambrian-heavy (limited Ordovician-Devonian)
- North America-focused
- Small sample (120 terms total)

---

## Proposed Scope Options

### Option A: Cambrian Trilobites (Narrow Focus)

**Advantages:**
- Manageable vocabulary size (200-400 terms)
- Coherent geological/taxonomic scope
- Rich literature base
- Well-defined stratigraphic framework

**Disadvantages:**
- Limited applicability to other taxa
- Single period focus
- May not generalize well

**Target Users:**
- Cambrian specialists
- Trilobite researchers
- Burgess Shale / Cambrian Explosion studies

**Vocabulary Estimate:**
- Taxa: 150-200 (trilobite genera/species + orders)
- Formations: 80-100 (Cambrian formations globally)
- Chronology: 30-40 (Cambrian stages/series)
- Localities: 100-150 (major Cambrian fossil sites)
- **Total: 360-490 tokens**

---

### Option B: Cambrian Explosion Fauna (Moderate Focus)

**Advantages:**
- Broader taxonomic coverage
- Still coherent time period
- High scientific interest (Cambrian Explosion)
- Diverse morphology (important NER challenge)

**Disadvantages:**
- More complex taxonomy
- Some groups poorly known
- Larger vocabulary

**Target Users:**
- Cambrian paleontologists (all groups)
- Evolutionary biologists
- Burgess/Chengjiang/Sirius Passet researchers

**Taxa Groups:**
1. Trilobites (arthropods) - 150 terms
2. Brachiopods - 50 terms
3. Archaeocyaths (reef builders) - 30 terms
4. Hyoliths - 20 terms
5. Early echinoderms - 30 terms
6. Anomalocaridids (predators) - 20 terms
7. Burgess Shale fauna (soft-bodied) - 50 terms
8. Early sponges - 20 terms

**Vocabulary Estimate:**
- Taxa: 350-400
- Formations: 100-120
- Chronology: 30-40
- Localities: 120-150
- **Total: 600-710 tokens**

---

### Option C: Paleozoic Invertebrates (Broad Focus)

**Advantages:**
- Wide applicability
- Covers major fossil groups
- Entire Paleozoic Era
- Maximum user base

**Disadvantages:**
- Large vocabulary (potential inefficiency)
- Less coherent scope
- Complex chronostratigraphy
- Risk of dilution

**Target Users:**
- General paleontologists
- Stratigraphers
- Museum curators
- Paleobiology researchers

**Taxa Groups:**
1. Trilobites (Cambrian-Devonian) - 200 terms
2. Brachiopods (all Paleozoic) - 150 terms
3. Graptolites (Ordovician-Silurian) - 80 terms
4. Crinoids (Ordovician-Permian) - 80 terms
5. Bryozoans - 60 terms
6. Corals (rugose, tabulate) - 80 terms
7. Ammonoids (early) - 50 terms
8. Conodonts - 40 terms
9. Ostracods - 40 terms
10. Early fish - 40 terms

**Vocabulary Estimate:**
- Taxa: 800-1000
- Formations: 300-400 (Paleozoic globally)
- Chronology: 80-100 (Cambrian-Permian stages)
- Localities: 200-300
- **Total: 1380-1800 tokens**

---

### Option D: Staged Expansion (Recommended)

**Phase 1 (v1.0): Cambrian Trilobites**
- Quick win, proof of concept
- Vocabulary: 360-490 tokens
- Timeline: Current (M1-M2)

**Phase 2 (v2.0): Cambrian Explosion**
- Add major Cambrian groups
- Vocabulary: +250 tokens → 600-710 total
- Timeline: After M5 (v1.0 release)
- Requires: New DAPT (tokenizer v2)

**Phase 3 (v3.0): Early Paleozoic**
- Add Ordovician-Silurian
- Add graptolites, crinoids, early fish
- Vocabulary: +400 tokens → 1000-1100 total
- Timeline: Separate project/publication

**Phase 4 (v4.0): Full Paleozoic**
- Complete Paleozoic coverage
- Vocabulary: +500 tokens → 1500-1600 total
- Timeline: Long-term (community-driven?)

---

## Decision Criteria

### Technical Factors

| Factor | Narrow | Moderate | Broad | Staged |
|--------|--------|----------|-------|--------|
| Vocabulary size | ✅ Small | ⚠️ Medium | ❌ Large | ✅ Grows |
| VRAM requirements | ✅ Low | ✅ Low | ⚠️ Medium | ✅ Low |
| Training time | ✅ Short | ✅ Short | ⚠️ Longer | ✅ Short |
| Maintenance | ✅ Easy | ✅ Easy | ⚠️ Complex | ✅ Versioned |

### Scientific Factors

| Factor | Narrow | Moderate | Broad | Staged |
|--------|--------|----------|-------|--------|
| Taxonomic coherence | ✅ High | ✅ High | ⚠️ Medium | ✅ High |
| User base | ⚠️ Small | ✅ Medium | ✅ Large | ✅ Grows |
| Literature base | ✅ Rich | ✅ Rich | ✅ Rich | ✅ Rich |
| Generalizability | ❌ Low | ⚠️ Medium | ✅ High | ✅ Improves |

### Practical Factors

| Factor | Narrow | Moderate | Broad | Staged |
|--------|--------|----------|-------|--------|
| Time to v1.0 | ✅ Fast | ✅ Fast | ❌ Slow | ✅ Fast |
| Risk | ✅ Low | ✅ Low | ⚠️ Medium | ✅ Low |
| Publications | ⚠️ Niche | ✅ Good | ✅ High impact | ✅ Multiple |
| Community adoption | ⚠️ Limited | ✅ Good | ✅ Wide | ✅ Grows |

---

## Recommendation

**Option D: Staged Expansion** ✅

**Rationale:**

1. **Fast iteration:** v1.0 delivers value quickly (Cambrian trilobites)
2. **Risk mitigation:** Prove architecture before expansion
3. **Publication strategy:** Multiple papers as scope grows
4. **Resource efficiency:** Each version justified by results
5. **Community feedback:** Learn from v1.0 users before v2.0

**v1.0 Scope (Recommended):**

```yaml
name: PaleoBERT v1.0 - Cambrian Trilobites
geological_time:
  primary: Cambrian Period (541-485 Ma)
  stages: Terreneuvian, Series 2, Miaolingian, Furongian

taxa_coverage:
  primary_group: Trilobita (Arthropoda)
  orders:
    - Agnostida
    - Redlichiida
    - Ptychopariida
    - Corynexochida
  genera: 100-150 most frequent
  species: 30-50 key index fossils

geographic_regions:
  primary: North America (Laurentia)
  secondary: China (South China block)
  tertiary: Europe (Baltica, Avalonia)

formations:
  count: 80-100
  examples:
    - Wheeler Formation (Utah)
    - Marjum Formation (Utah)
    - Burgess Shale (British Columbia)
    - Spence Shale (Utah/Idaho)
    - Chengjiang Formation (China)
    - Kaili Formation (China)

vocabulary_target:
  taxa: 150-200
  strat_units: 80-100
  chrono_units: 30-40
  localities: 100-150
  total: 360-490 tokens
```

---

## Action Items

- [ ] **Decision required:** Confirm scope (Cambrian Trilobites for v1.0)
- [ ] Update OVERVIEW.md with explicit scope statement
- [ ] Expand vocabulary files to 360-490 tokens (from current 120)
- [ ] Document expansion criteria (frequency-based)
- [ ] Plan v2.0 scope (Cambrian Explosion) for future

---

## References

- Current sample vocabulary: `artifacts/vocab/*.txt` (120 terms)
- Architecture docs: `CLAUDE.md`, `OVERVIEW.md`
- Phase 1 strategy: `devlog/20251028_001_architecture_review_and_phased_strategy.md`

---

**Status:** Awaiting scope decision
**Next step:** Confirm v1.0 focus area with project stakeholder
