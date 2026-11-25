# Secondary RQ1 and RQ2 analysis

Exploratory comparisons selected after inspecting primary results. These supplement, and do not replace, the nine primary comparisons.

945 shared included questions; equal-weight average of seed-specific AUROCs for seeds 123, 456 and 789. 20,000 paired question-bootstrap draws resample the same questions for all models and scores. Seeds are not resampled. All original primary estimates and intervals were reproduced exactly.

One family of 19 secondary contrasts uses Bonferroni-adjusted percentile intervals: nominal individual coverage 1 − 0.05/19 = 99.7368%. This gives approximate family-wise 95% coverage under bootstrap assumptions for this specified family, not for every analysis considered during the project. Extreme-tail endpoints have Monte Carlo uncertainty (about 26 draws in each tail).

## Definitions

- Direct comparison: semantic AUROC minus token-entropy or answer-NLL AUROC within a condition. Positive values favor semantic entropy.
- Change in advantage: (semantic minus token-based AUROC under noise) minus (semantic minus token-based AUROC under clipping). Positive values mean semantic entropy changes more favorably when adding noise.
- Clipping effect: clipping-only AUROC minus standard-training AUROC for the same score.

## Direct estimator comparison

| Comparison | AUROC difference | Adjusted interval |
|---|---:|---|
| standard_poisson: semantic minus mean_token_entropy | +0.0079 | [-0.0404, +0.0536] |
| standard_poisson: semantic minus mean_answer_nll | +0.0275 | [-0.0160, +0.0695] |
| clipping_poisson: semantic minus mean_token_entropy | -0.0311 | [-0.0692, +0.0062] |
| clipping_poisson: semantic minus mean_answer_nll | -0.0167 | [-0.0520, +0.0183] |
| dp_sigma_0p1: semantic minus mean_token_entropy | +0.0530 | [+0.0076, +0.0985] |
| dp_sigma_0p1: semantic minus mean_answer_nll | +0.0406 | [+0.0007, +0.0793] |
| dp_sigma_0p25: semantic minus mean_token_entropy | +0.0980 | [+0.0469, +0.1494] |
| dp_sigma_0p25: semantic minus mean_answer_nll | +0.1012 | [+0.0542, +0.1497] |
| dp_sigma_0p5: semantic minus mean_token_entropy | +0.0987 | [+0.0521, +0.1481] |
| dp_sigma_0p5: semantic minus mean_answer_nll | +0.1017 | [+0.0565, +0.1487] |

## Change in semantic advantage versus clipping

| Comparison | AUROC difference | Adjusted interval |
|---|---:|---|
| dp_sigma_0p1: semantic versus mean_token_entropy | +0.0841 | [+0.0384, +0.1297] |
| dp_sigma_0p1: semantic versus mean_answer_nll | +0.0573 | [+0.0200, +0.0951] |
| dp_sigma_0p25: semantic versus mean_token_entropy | +0.1291 | [+0.0731, +0.1840] |
| dp_sigma_0p25: semantic versus mean_answer_nll | +0.1180 | [+0.0689, +0.1694] |
| dp_sigma_0p5: semantic versus mean_token_entropy | +0.1298 | [+0.0762, +0.1815] |
| dp_sigma_0p5: semantic versus mean_answer_nll | +0.1184 | [+0.0691, +0.1687] |

## Clipping minus standard

| Comparison | AUROC difference | Adjusted interval |
|---|---:|---|
| mean_token_entropy | +0.0299 | [+0.0082, +0.0520] |
| mean_answer_nll | +0.0351 | [+0.0117, +0.0587] |
| discrete_semantic_entropy | -0.0092 | [-0.0381, +0.0199] |

## Findings for the research questions

RQ1: All six noisy-condition direct comparisons favor semantic entropy, with adjusted intervals above zero. The sigma 0.1 comparison against answer NLL is borderline (lower endpoint approximately 0.00066); its classification is sensitive to sampling and extreme-tail Monte Carlo uncertainty. All six changes in semantic advantage versus clipping are positive with adjusted intervals above zero. This supports a more favorable response of semantic entropy to added noise in these fitted models. The original hypothesis of a smaller semantic decline is only partly aligned with these findings: semantic AUROC increased versus clipping, rather than declining less. This does not demonstrate that stronger privacy universally improves semantic uncertainty.

RQ2: Clipping minus standard is positive for token entropy (+0.0299) and answer NLL (+0.0351), with adjusted intervals excluding zero. Semantic entropy changes by −0.0092, with an interval spanning zero. Together with the primary noise-versus-clipping results, the evidence distinguishes a favorable clipping association for token-based scores from their subsequent decline with added noise, while semantic entropy improves relative to clipping.

RQ3: Risk and coverage uncertainty is not computed by this secondary AUROC script; frozen-threshold point estimates remain in the primary report.

## Interpretation limits

Intervals describe question-sampling uncertainty conditional on the 15 fitted adapters and fixed correctness labels. They do not account for training-run population variability, label errors or semantic-clustering uncertainty. Labels are AI-assisted with partial human review. The test split had prior Qwen exposure, and the noise range was selected after development exploration. No direct noise-level contrasts are tested here, so this analysis does not establish a monotonic dose response. AUROC does not measure answer accuracy.

The lower-noise conditions can have weak privacy: reported per-run epsilon is approximately 916.27, 53.72 and 9.17 for sigma 0.1, 0.25 and 0.5 respectively at delta 1e-5. These are not joint-release privacy guarantees.

## Reproduction

Extract the complete ZIP, install numpy, pandas and scikit-learn, then run `python secondary_test_analysis/analyze.py`. Included source report ZIPs and reviewed CSVs are hash-verified. The original primary results JSON is included as a regression check. No model inference or training is required. See analysis_environment.json for the analysis package versions.
