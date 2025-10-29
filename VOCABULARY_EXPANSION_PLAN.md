# Vocabulary Expansion Plan: 120 → 400 Tokens

**Target:** PaleoBERT-Cambrian v1.0
**Current:** 120 sample tokens (30 per category)
**Goal:** 400 Cambrian-specific tokens
**Timeline:** 2-4 weeks (data collection + curation)
**Status:** PLAN

---

## Executive Summary

Expand sample vocabulary (120 tokens) to production vocabulary (~400 tokens) for PaleoBERT-Cambrian v1.0, using frequency-based selection from Cambrian paleontology literature.

**Target Distribution:**
- Taxa: 150-200 tokens (trilobites, brachiopods, archaeocyaths, etc.)
- Stratigraphic Units: 80-100 tokens (Cambrian formations globally)
- Chronostratigraphic Units: 30-40 tokens (Cambrian stages/series)
- Localities: 100-150 tokens (major Cambrian fossil sites)

**Total:** 360-490 tokens (target ~400)

---

## Current Status (v0.1 Sample)

### Existing Vocabulary (120 tokens)

| Category | Count | Coverage | Status |
|----------|-------|----------|--------|
| Taxa | 30 | Trilobites only, North America-focused | Prototype |
| Strat Units | 30 | Major formations + generic terms | Partial |
| Chrono Units | 30 | Cambrian stages/series complete | Good |
| Localities | 30 | North America + China/Europe samples | Partial |
| **Total** | **120** | Proof-of-concept | → Expand to 400 |

**Limitations:**
- Trilobite-heavy (no brachiopods, archaeocyaths)
- North America bias (limited Asia/Europe coverage)
- Sample only (not frequency-based)

---

## Expansion Strategy

### Principle: Frequency-Based Selection

**NOT:** Exhaustive enumeration of all Cambrian taxa
**YES:** Top N most frequent terms from Cambrian literature

**Rationale:**
- Top 20% of taxa account for 80% of occurrences (Zipf's Law)
- Rare taxa handled well by subword tokenization
- Efficient vocabulary usage

### Data Sources

#### Primary: Cambrian Literature Corpus

```
Phase 1: Corpus Collection (1 week)
├─ Gather Cambrian paleontology papers (open-access)
│   ├─ PaleoBios, Palaeontology, Journal of Paleontology
│   ├─ Geological Society bulletins (Cambrian sections)
│   └─ Museum publications (Burgess Shale, Chengjiang)
│
├─ Target: 5-10M tokens (subset for frequency analysis)
│   └─ Full corpus for DAPT: 40-50M tokens
│
└─ Output: Term frequency lists per category
```

#### Secondary: Reference Sources

```
Taxa:
- Treatise on Invertebrate Paleontology (Trilobita volumes)
- Paleobiology Database (pbdb.org) - Cambrian entries
- Sepkoski's Compendium (Cambrian genera)

Formations:
- Macrostrat database (Cambrian formations)
- Regional geological surveys
- GeoDeepDive Cambrian formation mentions

Localities:
- GBIF (Global Biodiversity Information Facility)
- Paleobiology Database localities
- Major monograph locality lists
```

---

## Category-Specific Plans

### 1. Taxa (150-200 tokens)

**Current:** 30 trilobite genera/species
**Target:** 150-200 Cambrian invertebrates

#### 1.1 Trilobites (100-120 tokens)

**Orders (10-12):** All major Cambrian trilobite orders
```
Agnostida
Redlichiida
Ptychopariida
Corynexochida
Olenellida
Asaphida (early Cambrian)
Harpetida
Phacopida (early representatives)
Lichida
Proetida (if early Cambrian)
...
```

**Genera (60-80):** Frequency-based from literature
```
Selection criteria:
- Appears in ≥5 papers OR
- >20 occurrences in corpus OR
- Index fossil / type genus

Examples:
Olenellus (very frequent)
Paradoxides (very frequent)
Elrathia (very frequent)
Asaphiscus (frequent)
Peronopsis (frequent)
Olenus (frequent)
Bolaspidella (moderate)
Bathynotus (moderate)
...
```

**Species (30-40):** Highly cited index fossils
```
Olenellus_wheeleri (Marjum/Wheeler Fm index)
Elrathia_kingii (Wheeler Shale, very common)
Paradoxides_davidis (Cambrian Series 3 index)
Asaphiscus_wheeleri (Wheeler Fm)
Olenus_gibbosus (Upper Cambrian index)
...
```

#### 1.2 Brachiopods (20-30 tokens)

**Major Cambrian groups:**
```
Lingulata
Obolellida
Acrotretida
Orthida (early)
Kutorginata
Paterinida

Genera:
Lingulella
Obolella
Acrothele
Nisusia
Micromitra
...
```

#### 1.3 Archaeocyaths (10-15 tokens)

**Reef-building sponges (Early-Middle Cambrian):**
```
Archaeocyatha (general)
Ajacicyathida
Monocyathida
Capsulocyathida

Genera (if frequent):
Archaeocyathus
Dictyocyathus
...
```

#### 1.4 Other Key Groups (20-25 tokens)

```
Hyolitha / Hyolithids
Anomalocaridida / Anomalocaris
Echinoderms:
  - Eocrinoidea
  - Helicoplacoidea
Halkieriida
Chancelloriida
Sponges (Porifera)
Coeloscleritophora
```

#### 1.5 Soft-bodied Fauna (10-15 tokens)

**Burgess Shale / Chengjiang fauna:**
```
Marrella
Opabinia
Wiwaxia
Hallucigenia
Pikaia
Canadaspis
Aysheaia
Sanctacaris
Yunnanozoon
Haikouella
...
```

---

### 2. Stratigraphic Units (80-100 tokens)

**Current:** 30 (major formations + generic terms)
**Target:** 80-100 (global Cambrian formations)

#### 2.1 North America (30-40)

**Laurentia - USA:**
```
Wheeler_Formation (Utah)
Marjum_Formation (Utah)
Weeks_Formation (Utah)
Spence_Shale (Utah/Idaho)
Bright_Angel_Formation (Arizona)
Tapeats_Sandstone (Arizona)
Pioche_Shale (Nevada)
Combined_Metals_Member (Nevada)
Langston_Formation (Utah)
Orr_Formation (Utah)
Notch_Peak_Formation (Utah)
Tatow_Formation (Nevada)
Poleta_Formation (California)
Sekwi_Formation (Northwest Territories)
...
```

**Laurentia - Canada:**
```
Burgess_Shale (British Columbia) [already in]
Stephen_Formation (British Columbia)
Cathedral_Formation (British Columbia)
Eldon_Formation (British Columbia)
Pika_Formation (British Columbia)
Chancellor_Group (British Columbia)
Backbone_Ranges_Formation (British Columbia)
...
```

#### 2.2 China (20-25)

**South China Block:**
```
Chengjiang_Formation (Yunnan) [already in]
Kaili_Formation (Guizhou) [already in]
Balang_Formation (Guizhou)
Qiongzhusi_Formation (Sichuan)
Niutitang_Formation (widespread)
Guanshan_Biota (Yunnan)
Shipai_Formation (Hubei)
...
```

**North China Block:**
```
Mantou_Formation
Zhangxia_Formation
Gushan_Formation
Chaomidian_Formation
...
```

#### 2.3 Europe (15-20)

**Baltica:**
```
Alum_Shale_Formation (Scandinavia)
File_Haidar_Formation (Sweden)
...
```

**Avalonia (UK, Wales):**
```
Comley_Sandstone
Wrekin_Quartzite
Harlech_Grits
...
```

**Iberia:**
```
Vegadeo_Formation (Spain)
...
```

#### 2.4 Other Regions (10-15)

**Australia:**
```
Emu_Bay_Shale (South Australia)
Henson_Glauconite (Queensland)
...
```

**Antarctica:**
```
Shackleton_Limestone
...
```

**Morocco / Gondwana:**
```
Tatelt_Formation
Issafeniense_Formation
...
```

#### 2.5 Generic Terms (keep all from v0.1)

```
Formation
Member
Group
Shale
Limestone
Sandstone
Mudstone
Dolomite
Conglomerate
Siltstone
Upper_Member
Middle_Member
Lower_Member
Basal_Member
...
```

---

### 3. Chronostratigraphic Units (30-40 tokens)

**Current:** 30 (good coverage)
**Target:** 30-40 (minor additions)

**Status:** Current v0.1 vocabulary is already comprehensive for Cambrian.

**Additions (if needed):**
```
Additional regional stages:
- Atdabanian (Siberian regional)
- Botomian (Siberian regional)
- Toyonian (Siberian regional)
- Sunwaptan (North American regional)
- Marjuman (North American regional)

Trilobite zones (if very frequent):
- Olenellus_Zone
- Paradoxides_Zone
- Olenus_Zone
```

**Priority:** LOW (current coverage good)

---

### 4. Localities (100-150 tokens)

**Current:** 30 (sample)
**Target:** 100-150 (global coverage)

#### 4.1 North America (40-50)

**Utah:**
```
House_Range [already in]
Wellsville_Mountains [already in]
Drum_Mountains [already in]
Notch_Peak [already in]
Swasey_Peak [already in]
Wheeler_Amphitheater
Antelope_Spring
Spence_Gulch
Millard_County
Box_Elder_County
...
```

**British Columbia:**
```
Yoho_National_Park [already in]
Mount_Stephen [already in]
Burgess_Pass [already in]
Walcott_Quarry [already in]
Raymond_Quarry [already in]
Field [already in]
Kicking_Horse_Pass
Stanley_Glacier
Mount_Field
...
```

**Other States:**
```
Nevada [already in]
Idaho [already in]
Montana [already in]
Wyoming [already in]
California
Arizona
Vermont
New_York
Pennsylvania
...
```

#### 4.2 China (20-30)

```
Yunnan [partial]
Chengjiang [add]
Kunming
Maotianshan
Guizhou
Kaili
Balang
Jianhe
Sichuan
Hubei
Shaanxi
...
```

#### 4.3 Europe (15-20)

**Scandinavia:**
```
Sweden [partial]
Västergötland
Kinnekulle
Närke
Norway
Mjøsa
...
```

**UK:**
```
England [partial]
Wales
Shropshire
Nuneaton
St_Davids
Harlech
...
```

**Czechia (Bohemia):**
```
Jince
Skryje
Prague_Basin
Príbram
...
```

**Spain:**
```
Cantabrian_Mountains
Asturias
...
```

#### 4.4 Other Regions (20-30)

**Australia:**
```
South_Australia
Kangaroo_Island
Emu_Bay
Flinders_Ranges
Queensland
...
```

**Antarctica:**
```
Transantarctic_Mountains [already in]
Ellsworth_Mountains
Shackleton_Range
Pensacola_Mountains
...
```

**Morocco:**
```
Anti_Atlas
Jebel_Wawrmast
Zagora
...
```

**Siberia:**
```
Siberian_Platform
Lena_River
Aldan_River
Olenek_River
...
```

#### 4.5 Paleogeography (10)

```
Laurentia [already in]
Gondwana [already in]
Baltica [already in]
South_China [already in]
North_China [add]
Siberia [add]
Avalonia [add]
Iapetus_Ocean [add]
Panthalassa [add]
...
```

---

## Implementation Workflow

### Phase 1: Corpus Analysis (Week 1)

```
Task 1.1: Collect Cambrian Literature
├─ Download open-access papers (500-1000 papers)
├─ OCR / extract text from PDFs
├─ Clean and normalize text
└─ Total: ~10M tokens for analysis

Task 1.2: Frequency Analysis
├─ Extract candidate taxa (capitalized 1-3 word sequences)
├─ Extract formations (Pattern: X Formation, X Shale, etc.)
├─ Extract localities (geo-referenced entities)
├─ Count frequencies per category
└─ Output: Ranked frequency lists

Task 1.3: Manual Curation
├─ Review top 500 candidates per category
├─ Remove false positives (person names, modern places, etc.)
├─ Add known important terms (even if low frequency)
└─ Output: Curated candidate lists
```

### Phase 2: Selection & Validation (Week 2)

```
Task 2.1: Apply Selection Criteria
Taxa:
├─ Frequency ≥ 20 mentions: AUTO-INCLUDE
├─ Frequency 5-19: REVIEW (include if important)
├─ Frequency <5: EXCLUDE (unless index fossil)
└─ Target: 150-200 tokens

Formations:
├─ Frequency ≥ 10 mentions: AUTO-INCLUDE
├─ Frequency 3-9: REVIEW
├─ Add major formations even if low in corpus
└─ Target: 80-100 tokens

Localities:
├─ Major fossil sites: INCLUDE
├─ Paleogeographic units: INCLUDE
├─ Modern administrative regions if frequently used
└─ Target: 100-150 tokens

Chronology:
├─ Keep all v0.1 terms (30)
├─ Add 5-10 regional stages if needed
└─ Target: 30-40 tokens

Task 2.2: Validate Additions
├─ Check spelling consistency
├─ Resolve duplicates (e.g., Wheeler Fm vs Wheeler Formation)
├─ Normalize to underscore format (Wheeler_Formation)
├─ Cross-reference with authority sources
└─ Output: Final validated lists

Task 2.3: Fragmentation Testing
├─ Load base DeBERTa tokenizer
├─ Test fragmentation rate on new terms
├─ If term fragments <3 tokens, may skip adding
└─ Confirm benefit of adding each term
```

### Phase 3: File Generation (Week 3)

```
Task 3.1: Update Vocabulary Files
├─ artifacts/vocab/taxa.txt (150-200 lines)
├─ artifacts/vocab/strat_units.txt (80-100 lines)
├─ artifacts/vocab/chrono_units.txt (30-40 lines)
├─ artifacts/vocab/localities.txt (100-150 lines)
└─ Total: 360-490 lines

Task 3.2: Documentation
├─ Update artifacts/vocab/README.md
│   ├─ Selection methodology
│   ├─ Frequency thresholds
│   ├─ Sources cited
│   └─ Version history (v0.1 → v1.0)
└─ Create VOCABULARY_SOURCES.md (data provenance)

Task 3.3: Rebuild Tokenizer
├─ Run scripts/build_tokenizer.py
├─ Verify vocabulary size (~128,400 tokens)
├─ Run scripts/validate_tokenizer.py
├─ Confirm 0% fragmentation rate
└─ Output: artifacts/tokenizer_v1/ (updated)
```

### Phase 4: Quality Assurance (Week 4)

```
Task 4.1: Coverage Testing
├─ Test on diverse Cambrian papers
├─ Measure coverage (% of domain terms as single tokens)
├─ Identify gaps (missing important terms)
└─ Iterate if needed

Task 4.2: Documentation Review
├─ Ensure all sources documented
├─ Verify licensing compliance
├─ Complete MODEL_CARD.md vocabulary section
└─ Final review

Task 4.3: Version Tagging
├─ Tag vocabulary as v1.0
├─ Freeze vocabulary for DAPT
├─ Document in git (commit + tag)
└─ Proceed to DAPT (M1)
```

---

## Selection Criteria Summary

### Inclusion Criteria

**Taxa:**
1. Frequency ≥ 20 in Cambrian corpus → AUTO-INCLUDE
2. Index fossil or type species → INCLUDE
3. Genus appears in major monographs → INCLUDE
4. Order/Family level → INCLUDE ALL

**Formations:**
1. Frequency ≥ 10 in corpus → AUTO-INCLUDE
2. Major fossil locality (Burgess, Chengjiang) → INCLUDE
3. Mentioned in standard references → INCLUDE

**Localities:**
1. Type locality for important taxa → INCLUDE
2. Major fossil Lagerstätten → INCLUDE
3. Paleogeographic units → INCLUDE

**Chronology:**
1. International stages → INCLUDE ALL
2. Regional stages if frequent (≥5 mentions) → INCLUDE

### Exclusion Criteria

1. **Frequency too low:** <3 mentions (except index fossils)
2. **Modern contamination:** Modern species, person names
3. **Overly generic:** "trilobite", "fossil", "specimen" (handled by base vocabulary)
4. **Redundancy:** Already covered by base tokenizer
5. **Non-Cambrian:** Ordovician/Silurian taxa appearing in comparative sections

---

## Expected Outcomes

### Quantitative Targets

```
Category Breakdown (Final v1.0):

Taxa:                150-200 tokens
├─ Trilobites:       100-120
├─ Brachiopods:       20-30
├─ Archaeocyaths:     10-15
├─ Other inverts:     20-25
└─ Soft-bodied:       10-15

Strat Units:          80-100 tokens
├─ North America:     30-40
├─ China:             20-25
├─ Europe:            15-20
├─ Other:             10-15
└─ Generic terms:     ~15

Chrono Units:         30-40 tokens
├─ International:     ~25
├─ Regional:          5-10
└─ Zones:             0-5

Localities:          100-150 tokens
├─ North America:     40-50
├─ China:             20-30
├─ Europe:            15-20
├─ Other:             20-30
└─ Paleogeography:    ~10

TOTAL:               360-490 tokens
Target:              ~400 tokens
```

### Qualitative Goals

1. **Coverage:** ≥80% of Cambrian domain terms as single tokens
2. **Balance:** Good geographic representation (not just North America)
3. **Utility:** Focus on high-impact terms (frequent + important)
4. **Efficiency:** No wasted tokens on rare terms
5. **Maintainability:** Well-documented sources and rationale

---

## Success Metrics

### After Expansion

```
Test on 100 diverse Cambrian papers:

Metric 1: Domain Term Coverage
├─ Target: ≥80% of domain-specific terms are single tokens
├─ Current (v0.1): ~40-50% (120 tokens)
└─ Expected (v1.0): 80-85% (400 tokens)

Metric 2: Fragmentation Rate
├─ v1.0 vocabulary: 0% (all 400 terms single tokens)
└─ Novel terms: <30% (handled by subword)

Metric 3: Token Count Reduction
├─ vs Base DeBERTa: 40-60% fewer tokens for Cambrian text
└─ Example: "Olenellus wheeleri from Wheeler Formation"
    Base: 7-8 tokens
    PaleoBERT-Cambrian: 4 tokens (57% reduction)

Metric 4: Vocabulary Efficiency
├─ Utilization rate: ≥60% of added tokens used in corpus
└─ No "dead" tokens (unused in any document)
```

---

## Risks & Mitigations

### Risk 1: Incomplete Coverage

**Risk:** Missing important taxa/formations
**Mitigation:**
- Cross-reference with standard treatises
- Consult domain experts (if available)
- Iterative refinement based on DAPT corpus

### Risk 2: Geographic Bias

**Risk:** Over-representation of North American terms
**Mitigation:**
- Explicit quotas per region
- Include Chinese/European/Australian localities
- Review for balance before finalizing

### Risk 3: Temporal Contamination

**Risk:** Including Ordovician/Silurian terms
**Mitigation:**
- Strict temporal filtering (Cambrian only)
- Check each term against geological time scale
- Remove cross-period taxa from comparative studies

### Risk 4: OCR Errors

**Risk:** Misspelled terms entering vocabulary
**Mitigation:**
- Manual review of all additions
- Cross-check with authority databases
- Spell-check against scientific databases

---

## Timeline & Dependencies

```
Week 1: Corpus Collection & Frequency Analysis
├─ Dependencies: Access to Cambrian literature
├─ Output: Frequency-ranked lists
└─ Risk: Literature access restrictions

Week 2: Selection & Validation
├─ Dependencies: Week 1 outputs
├─ Output: Curated vocabulary lists (400 tokens)
└─ Risk: Expert consultation if needed

Week 3: File Generation & Rebuild
├─ Dependencies: Week 2 outputs
├─ Output: Updated tokenizer_v1
└─ Risk: Build script issues (should be resolved)

Week 4: QA & Documentation
├─ Dependencies: Week 3 outputs
├─ Output: Frozen v1.0 vocabulary, ready for DAPT
└─ Risk: Coverage gaps requiring iteration
```

**Critical Path:** Corpus collection → Frequency analysis → Selection
**Parallel Work:** Documentation can be drafted during selection phase

---

## Next Steps

### Immediate (This Session)

1. ✅ Document expansion plan (this file)
2. Create corpus collection task list
3. Identify open-access Cambrian paper sources
4. Set up frequency analysis scripts

### Short-term (Next 2 Weeks)

5. Collect 500-1000 Cambrian papers
6. Run frequency analysis
7. Generate candidate lists
8. Manual curation and validation

### Medium-term (Weeks 3-4)

9. Generate final vocabulary files (400 tokens)
10. Rebuild and validate tokenizer
11. Update documentation
12. Freeze v1.0 and proceed to DAPT (M1)

---

## References

- PROJECT_SCOPE.md - Rationale for Cambrian focus
- FAMILY_ARCHITECTURE.md - Long-term family strategy
- OVERVIEW.md - Updated with Cambrian scope
- devlog/20251029_002_P01_tokenizer_completion.md - P01 status

---

**Status:** PLAN - Ready for execution
**Owner:** To be assigned
**Timeline:** 2-4 weeks
**Blocker:** None (can start immediately)
**Next:** Begin corpus collection
