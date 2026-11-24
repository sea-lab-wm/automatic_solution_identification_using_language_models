# Explaining the Main Findings (R2.C5 / ED.C2)

Script: `results_analysis/explain_findings.py` (deterministic: no training, no sampling).

**Data.** Per-comment predictions from the paper's 10-fold CV (`ensembled_models.xlsx`): 4,867 test comments (788 solution comments; the 50 prompt-development comments are excluded). Saved ML tuning results (`results/ml/comment_data_results copy.csv`), restricted to the four representations in the paper. Full dataset (4,917 comments) for analysis 1a.

Models: SVM, RoBERTa, and Llama-3_ft (the best three), plus prompted Llama-3 (best prompt: CoT, seven examples; metrics averaged over its three executions, as in Table 5).

---

## 1. What do the models learn? Distinctive words and recall per solution code

**Method.**
- (a) Chi-square test of word presence (stop words removed; the top words are the same for any minimum frequency from 1 to 20 comments) against the solution label. We list the top words that are more frequent in solution comments.
- (b) Recall of each model per solution code.

**Findings.**

(a) Top words:

| Word | χ² | % solution | % non-solution |
|---|---|---|---|
| created | 528.1 | 49.5 | 11.9 |
| attachment | 143.6 | 49.8 | 24.9 |
| patch | 111.3 | 27.0 | 11.7 |
| way | 109.8 | 9.5 | 2.1 |
| need / make / think / use / instead | 60–80 | 7–15 | 2–6 |
| property / event / function | 52–82 | 2–6 | 0.2–1.7 |
| solution / feedback | 53–55 | 2–3 | 0.1–0.3 |

The strongest signal is Bugzilla's "Created attachment" text, which is added when a patch is uploaded (the phrase appears in 48.4% of solution vs. 11.3% of non-solution comments; `1a_created_attachment_phrase.csv`). The other signals are design language ("way", "instead", "think", "need") and code terms ("function", "property", "event").

(b) Recall per solution code:

| Code | n | SVM | RoBERTa | Llama-3_ft | Llama-3 (prompted) |
|---|---|---|---|---|---|
| CODE_DESCRIPTION | 442 | 0.837 | 0.871 | **0.912** | 0.593 |
| PROPOSED_SOLUTION | 220 | 0.505 | 0.550 | 0.755 | **0.845** |
| SOLUTION_REVIEW | 126 | 0.563 | 0.437 | 0.722 | **0.735** |

- Code descriptions are easy for all fine-tuned models because they carry surface cues (attachment/patch text).
- Proposed solutions and reviews have no such cues. Here, SVM and RoBERTa recall drops to about 0.44–0.56, while Llama-3_ft keeps 0.72–0.76. **This explains most of Llama-3_ft's recall advantage.**
- Prompted Llama-3 shows the opposite pattern: it is best on proposals and reviews but misses 41% of code descriptions. It relies on the semantics of the definition, not on Bugzilla-specific cues.

## 2. Why data balancing (SMOTE) does not consistently help

**Method.**
- Imbalance ratio in the dataset.
- Class weights chosen by hyperparameter tuning in the runs without SMOTE (LR, DT, RF, SVM).
- Pooled precision/recall/F1 with vs. without SMOTE per ML model, over all folds and representations.

**Findings.**
- Imbalance is moderate: 808 vs. 4,109 comments (1:5.1).
- Without SMOTE, tuning selected `class_weight='balanced'` in **156 of 160** runs. The imbalance is therefore already handled for the models that support class weights.

| Model | Class weights? | P (no SMOTE → SMOTE) | R (no SMOTE → SMOTE) | F1 (no SMOTE → SMOTE) |
|---|---|---|---|---|
| LR | yes | 0.471 → 0.504 | 0.708 → 0.543 | 0.566 → 0.523 |
| SVM | yes | 0.460 → 0.555 | 0.708 → 0.545 | 0.558 → 0.550 |
| DT | yes | 0.322 → 0.371 | 0.636 → 0.487 | 0.427 → 0.421 |
| RF | yes | 0.620 → 0.623 | 0.324 → 0.432 | 0.425 → 0.511 |
| KNN | no | 0.515 → 0.296 | 0.286 → 0.650 | 0.368 → 0.407 |
| NB | no | 0.291 → 0.253 | 0.720 → 0.656 | 0.414 → 0.366 |

- For models already weighted (LR, SVM, DT), SMOTE mainly trades recall for precision, and F1 does not improve.
- SMOTE helps where the model does not compensate for the imbalance: KNN (no class weights) and RF (low recall despite class weights).

**Guidance.** With a moderate imbalance, use class weights when the model supports them. Try SMOTE only for models that cannot be weighted, or that still show low recall.

## 3. Why prompting has low precision

**Method.**
- Precision/recall of each model.
- False-positive rate on non-solution comments with vs. without technical content (regex for code identifiers, function calls, source-file names, `::`). 31.7% of the non-solution comments contain technical content.

**Findings.**

| Model | P | R | F1 | FP rate (technical) | FP rate (non-technical) |
|---|---|---|---|---|---|
| Llama-3 (prompted) | 0.414 | 0.686 | 0.516 | **0.255** | 0.156 |
| SVM | 0.629 | 0.701 | 0.663 | 0.118 | 0.062 |
| RoBERTa | 0.673 | 0.712 | 0.692 | 0.095 | 0.053 |
| Llama-3_ft | 0.626 | 0.838 | 0.716 | 0.146 | 0.074 |

- Prompted Llama-3 has a recall similar to SVM and RoBERTa, so its gap is precision. It flags **25% of technical non-solution comments** (e.g., problem analyses that mention code) and 16% of the others, roughly twice the fine-tuned models' rates in both groups.
- Technical content raises false positives for every model (about 1.6–2× higher), so the effect is not unique to prompting. Without labeled examples, however, the prompted model cannot learn where Firefox annotators draw the line between discussing code and proposing a solution.

## 4. Why ensembles give limited gains: error overlap

**Method.** Overlap of the misses and false positives of Llama-3_ft with those of SVM and RoBERTa.

**Findings.**
- Of 128 solution comments missed by Llama-3_ft, **75 (59%) are also missed by both** SVM and RoBERTa. Only 22 are detected by both, which is the only case where majority voting fixes a Llama-3_ft miss.
- Of 395 Llama-3_ft false positives, 224 (57%) are shared with at least one other model, and 108 with both.
- Llama-3_ft alone detects 71 solution comments that both others miss. Majority voting overrules these.
- Even an ideal "any model detects" rule reaches only 0.905 recall.

The models fail on largely the same comments, so combining them adds little, and voting discards Llama-3_ft's unique detections.

## 5. Why performance differs across problem categories

**Method.** For categories with ≥20 solution comments (10 categories): the share of CODE_DESCRIPTION among solution comments vs. the mean F1 of the three best models (Spearman).

**Findings.**
- Mean F1 ranges from 0.639 (unnecessary code removal) to 0.747 (performance optimization).
- The category with the highest share of code descriptions (code improvement, 76%) is among the best. Feature development (44%) and UI issues (50%) are at or below the median.
- The correlation is positive but **not significant** (ρ = 0.49, p = 0.15, n = 10).

The trend is consistent with analysis 1b (categories with more code descriptions are easier), but with 10 categories it is only indicative and should be reported as such.

---

**Output files** (same folder): `1a_distinctive_words.csv`, `1b_recall_per_solution_code.csv`, `2_imbalance_summary.csv`, `2_smote_effect_per_model.csv`, `3_prompting_and_technical_fp.csv`, `4_error_overlap.csv`, `5_categories.csv`.
