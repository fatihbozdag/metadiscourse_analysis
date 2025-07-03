# Things To Do - Metadiscourse Marker Extraction Improvements

## 🎯 Current Status
- **Overall Accuracy**: 41% (16/39 validated samples)
- **Major Issue**: High false positive rates in BOOSTERS (8.3%) and HEDGES (25%)
- **Strong Categories**: FRAME_MARKERS (100%), EVIDENTIALS (100%), CODE_GLOSSES (85.7%)

## 🚨 Priority 1: Critical Pattern Fixes

### 1.1 Fix Booster Over-Detection (Current: 8.3% accuracy)
- **Problem**: Detecting quantifiers (`many`, `most`, `all`, `much`) as emphatic boosters
- **Action**: 
  - Create strict booster patterns requiring emphatic context
  - Add anti-patterns for descriptive/quantitative usage
  - Focus on certainty expressions: `clearly`, `obviously`, `definitely`, `certainly`
  - Remove general quantifiers from booster patterns

### 1.2 Refine Hedge Detection (Current: 25% accuracy)  
- **Problem**: Detecting prepositional phrases and obligation modals
- **Action**:
  - Remove `about X` patterns completely 
  - Distinguish epistemic vs deontic modals (`may` vs `should`)
  - Focus on uncertainty markers: `seem`, `appear`, `might`, `perhaps`, `probably`
  - Add context checks for epistemic usage

### 1.3 Fix Engagement Marker Patterns (Current: 0% accuracy)
- **Problem**: All `consider that` instances were false positives
- **Action**:
  - Rebuild engagement patterns from scratch
  - Focus on direct reader address: `you`, `note that`, `consider this`
  - Add imperative context detection
  - Remove indirect reference patterns

## 🔧 Priority 2: Pattern Refinement

### 2.1 Category Confusion Resolution
- **Transition/Code Gloss Mix-up**: `for instance` incorrectly classified as transition
- **Action**: Move exemplification markers (`for instance`, `for example`) exclusively to CODE_GLOSSES

### 2.2 Context Sensitivity Improvements
- **Problem**: Detecting markers in personal narratives vs discourse function
- **Action**:
  - Add narrative context detection
  - Implement discourse vs personal context scoring
  - Penalize markers in personal story contexts

### 2.3 Pattern Weight Rebalancing
- **Current weights causing over-detection**
- **Action**:
  - Reduce booster pattern weights to 0.3-0.4
  - Increase hedge specificity requirements
  - Boost confidence for validated high-quality patterns

## 🎯 Priority 3: Quality Assurance

### 3.1 Implement Pattern Testing Framework
- Create test cases for each category with known true/false positives
- Automated validation against manually verified samples
- Regression testing for pattern changes

### 3.2 Enhanced Anti-Pattern System
- **Narrative anti-patterns**: `I went`, `we were`, `my family`
- **Conversational anti-patterns**: `very good`, `so much`, `many people`
- **Quantitative anti-patterns**: Numbers + quantifiers context

### 3.3 Confidence Calibration
- Align confidence scores with actual accuracy
- Lower confidence for problematic categories
- Higher confidence for validated patterns

## 📊 Priority 4: Category-Specific Improvements

### 4.1 Maintain High-Performing Categories
- **FRAME_MARKERS**: Keep current patterns (100% accuracy)
- **EVIDENTIALS**: Preserve `according to` and citation patterns  
- **CODE_GLOSSES**: Minor refinements for 85.7% → 90%+ accuracy

### 4.2 Rebuild Problematic Categories
- **BOOSTERS**: Complete pattern overhaul focusing on emphasis
- **HEDGES**: Epistemic-only detection with strict context
- **ENGAGEMENT_MARKERS**: Reader-directed patterns only

### 4.3 Improve Medium-Performing Categories  
- **TRANSITIONS**: Better discourse connection detection
- **SELF_MENTIONS**: Balance academic vs personal references

## 🔄 Priority 5: Iterative Testing

### 5.1 Validation Pipeline
1. Pattern modification
2. Test on validation samples
3. Manual verification of changes
4. Performance measurement
5. Iterative refinement

### 5.2 Target Metrics
- **Overall Accuracy**: 41% → 90%+
- **BOOSTERS**: 8.3% → 85%+
- **HEDGES**: 25% → 90%+
- **ENGAGEMENT_MARKERS**: 0% → 80%+

### 5.3 Continuous Monitoring
- Random sampling validation after each iteration
- Category-specific accuracy tracking
- False positive rate monitoring

## 🛠️ Implementation Order

1. **Week 1**: Fix critical booster over-detection
2. **Week 2**: Refine hedge epistemic detection  
3. **Week 3**: Rebuild engagement marker patterns
4. **Week 4**: Resolve category confusion issues
5. **Week 5**: Implement enhanced anti-patterns
6. **Week 6**: Quality assurance and testing framework

## 📋 Success Criteria

- [ ] Overall accuracy > 90%
- [ ] No category below 80% accuracy
- [ ] False positive rate < 10%
- [ ] Stable pattern matching (no regex errors)
- [ ] Balanced category distribution
- [ ] Validated on 100+ random samples

## 🎯 Focus Areas

**PRIMARY**: Precision over recall - better to miss markers than extract false positives
**SECONDARY**: Category balance and appropriate confidence scoring
**TERTIARY**: Processing speed and system stability

---

*Note: This plan focuses solely on marker identification and extraction quality, removing academic context requirements that may not apply to learner corpus analysis.* 