# Appendix A. Pattern inventory, disambiguation, and implementation details

## A1. Pattern distribution by category

**Table A1**
*Pattern distribution by metadiscourse category with illustrative examples*

| Category | Pattern count | Illustrative examples |
|----------|---------------|----------------------|
| Transitions | 16 | however; therefore; moreover; in contrast; on the other hand |
| Evidentials | 15 | according to; demonstrate; show; indicate; research finds |
| Frame markers | 14 | first; second; in conclusion; to summarize; finally |
| Code glosses | 11 | namely; such as; for example; in other words; i.e. |
| Engagement markers | 12 | note that; consider; you should; let us examine |
| Self-mentions | 12 | I argue; we propose; our research; this study |
| Boosters | 12 | clearly; obviously; certainly; definitely; strongly |
| Hedges | 12 | might; could; perhaps; seem; appear; possibly |
| **Total** | **104** | |

*Note.* Pattern counts include both single-word markers (e.g., *however*) and multi-word academic phrases (e.g., *on the other hand*, *according to*).

---

## A2. Ambiguity resolution: three worked examples

### Example A1. Temporal *however* vs. contrastive *however*

**Rule.** Transitions must signal logical relationships between propositions, not degree/quantity or concessive constructions.

**Accepted (academic metadiscourse).**
"The initial results supported the hypothesis. However, subsequent analysis revealed important limitations."

**Rejected (non-metadiscourse).**
"However much I try, I can't understand this problem."

**Disambiguation logic.** Accept when *however* precedes a complete clause introducing contrast and co-occurs with academic lexis (e.g., *results*, *analysis*, *hypothesis*). Reject when *however* is part of a quantity/degree construction (*however much/many*) or appears in personal-narrative contexts.

---

### Example A2. Academic "I argue" vs. narrative "I went"

**Rule.** Self-mentions count as metadiscourse only when they perform academic speech acts (arguing, proposing, analyzing), not when narrating personal actions.

**Accepted (academic self-mention).**
"I argue that this methodology offers significant advantages over previous approaches."

**Rejected (personal narrative).**
"I went to school yesterday to meet my advisor."

**Disambiguation logic.** Accept when first-person pronouns co-occur with academic verbs (*argue*, *propose*, *suggest*, *conclude*, *demonstrate*) and nearby academic nouns (*methodology*, *evidence*, *results*). Reject with action/motion verbs (*went*, *came*, *saw*) or with narrative indicators (*yesterday*, *home*, *family*).

---

### Example A3. Epistemic *perhaps* vs. conversational *maybe*

**Rule.** Hedges must express epistemic modality about research claims/findings, not casual suggestions about personal plans.

**Accepted (academic hedge).**
"Perhaps this finding warrants further investigation in different contexts."

**Rejected (conversational suggestion).**
"Maybe tomorrow we can go shopping together."

**Disambiguation logic.** Accept when hedging modifies propositions containing academic vocabulary (*finding*, *investigation*, *data*, *results*). Reject when paired with temporal adverbs for plans (*tomorrow*, *later*) or everyday activities (*shopping*, *movie*, *dinner*).

---

## A3. Implementation details

### Contextual scoring and classification

Each detected pattern is evaluated within a symmetric context window (~50 characters on either side). The system computes two scores: an *academic indicator score* (e.g., presence of academic verbs such as *argue*, *demonstrate*, *analyze* and nouns such as *study*, *research*, *evidence*, *data*) and a *non-academic indicator score* (e.g., narrative verbs *went*, *said*, *told*; personal-life contexts *family*, *home*, *store*). Classification confidence increases with the dominance of academic over non-academic indicators; all candidates are then evaluated by the machine-learning (ML) validator (when available), which refines classification through confidence-weighted combination with rule-based scores.

### Pseudocode (contextual scoring)

```python
# Step 1: Extract context window around marker
context = window(text, marker_position, width=50)

# Step 2: Count academic and non-academic indicators
ACADEMIC_VERBS = {'argue', 'demonstrate', 'show', 'indicate', 'suggest',
                  'propose', 'conclude', 'find', 'observe', 'analyze', ...}
ACADEMIC_NOUNS = {'study', 'research', 'analysis', 'investigation',
                  'findings', 'results', 'data', 'evidence', 'method', ...}
NARRATIVE_VERBS = {'went', 'came', 'said', 'told', ...}
PERSONAL_CONTEXT = {'family', 'friend', 'home', 'store', 'movie',
                    'game', 'food', 'dinner', ...}

acad_score = count_in(context, ACADEMIC_VERBS ∪ ACADEMIC_NOUNS)
nonacad_score = count_in(context, NARRATIVE_VERBS ∪ PERSONAL_CONTEXT)

# Step 3: Rule-based classification with confidence estimation
if acad_score > nonacad_score:
    confidence = min(0.80, 0.40 + 0.10 * acad_score)
    label = METADISCOURSE
else:
    confidence = min(0.80, 0.30 + 0.10 * nonacad_score)
    label = NON_METADISCOURSE

# Step 4: ML validator refines classification (applied to all candidates)
if ml_model_available:
    ml_label, ml_conf = ml_validator(context, marker)

    # High ML confidence: Trust ML prediction
    if ml_conf >= THRESHOLD:  # default THRESHOLD = 0.60
        label, confidence = ml_label, ml_conf

    # Low ML confidence: Combine ML with rule-based results
    else:
        # Case A: ML and rules agree
        if ml_label == label:
            confidence = (confidence + ml_conf) / 2

        # Case B: ML and rules disagree
        else:
            if ml_conf < 0.30:
                # Very low ML confidence → Trust rules
                pass  # Keep rule-based label and confidence
            else:
                # Moderate ML confidence → Use ML with penalty
                label, confidence = ml_label, ml_conf * 0.80
```

### Performance validation

On an independently annotated test set spanning all eight categories (86 test cases), the hybrid system achieved:

- **89.6%** true-positive detection (recall)
- **92.1%** specificity (true-negative rate)
- **90.8%** overall accuracy

Using a confidence threshold of **0.60** (balanced precision–recall setting).

---

## A4. System architecture summary

### Detection pipeline

1. **Pattern matching** → Identify candidate markers via keyword/phrase patterns
2. **Context extraction** → Extract ±50-character window around each candidate
3. **Rule-based scoring** → Compute academic vs. non-academic indicator scores
4. **ML validation** (when available) → Refine classification using a Random Forest classifier with linguistically informed features
5. **Confidence calibration** → Apply threshold-based filtering (default 0.60)
6. **Deduplication** → Resolve overlapping markers using category-specificity weights

### Feature engineering (ML classifier)

- **Lexical**: marker length, word count, capitalization, punctuation
- **Syntactic**: POS tags, dependency relations, head POS
- **Contextual**: left/right context POS, sentence position, distance to boundaries
- **Semantic**: sentence start/end, punctuation adjacency
- **Academic**: academic verb-phrase membership, academic context score

### NLP foundation

- **Transformer pipeline**: `en_core_web_trf` (RoBERTa-base)
- **spaCy version**: 3.7+
- **ML classifier**: Random Forest (100 estimators, max_depth = 15)

---

*End of Appendix A*
