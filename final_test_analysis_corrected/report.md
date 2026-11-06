# Three-seed Llama low-noise test analysis

945 shared included questions; 55 held. Seeds 123, 456, 789. All 45 reported metric rows reproduced from per-question data. Review hashes match the merged human-review packages. Frozen thresholds and their acceptance decisions verified. Shared frozen protocols and question-validity tables match.

## Mean results across seeds

| Condition | Accuracy | Token AUROC | NLL AUROC | Semantic AUROC |
|---|---:|---:|---:|---:|
| standard_poisson | 81.80% | 0.8388 | 0.8192 | 0.8467 |
| clipping_poisson | 81.55% | 0.8686 | 0.8542 | 0.8375 |
| dp_sigma_0p1 | 81.23% | 0.8293 | 0.8418 | 0.8823 |
| dp_sigma_0p25 | 79.19% | 0.7835 | 0.7803 | 0.8815 |
| dp_sigma_0p5 | 77.71% | 0.7939 | 0.7909 | 0.8926 |

## Frozen-threshold selective answering

Coverage and risk are separately averaged across seeds. Target was at most 10% observed DEVELOPMENT error; this is not a guarantee of TEST error.

| Condition | Token coverage / risk | NLL coverage / risk | Semantic coverage / risk |
|---|---:|---:|---:|
| standard_poisson | 78.34% / 8.83% | 76.33% / 8.87% | 83.99% / 8.99% |
| clipping_poisson | 82.29% / 9.16% | 77.92% / 7.82% | 81.69% / 8.90% |
| dp_sigma_0p1 | 69.70% / 8.34% | 74.04% / 8.38% | 78.77% / 7.70% |
| dp_sigma_0p25 | 55.94% / 8.32% | 61.69% / 9.71% | 69.84% / 6.29% |
| dp_sigma_0p5 | 56.12% / 8.42% | 56.30% / 8.95% | 67.23% / 5.66% |

## Primary comparisons: noise minus clipping

20,000 paired question-bootstrap draws, equal-weight mean of three seed-specific AUROCs. Each draw uses the same resampled questions for every seed, condition and estimator. Percentile intervals have nominal 99.4444% individual coverage, using Bonferroni alpha=0.05/9. No seeds are resampled.

| Condition | Score | AUROC change | Adjusted interval |
|---|---|---:|---|
| dp_sigma_0p1 | mean_token_entropy | -0.0393 | [-0.0717, -0.0068] |
| dp_sigma_0p1 | mean_answer_nll | -0.0125 | [-0.0463, +0.0214] |
| dp_sigma_0p1 | discrete_semantic_entropy | +0.0448 | [+0.0040, +0.0845] |
| dp_sigma_0p25 | mean_token_entropy | -0.0851 | [-0.1326, -0.0370] |
| dp_sigma_0p25 | mean_answer_nll | -0.0740 | [-0.1235, -0.0250] |
| dp_sigma_0p25 | discrete_semantic_entropy | +0.0440 | [+0.0043, +0.0840] |
| dp_sigma_0p5 | mean_token_entropy | -0.0747 | [-0.1225, -0.0250] |
| dp_sigma_0p5 | mean_answer_nll | -0.0633 | [-0.1126, -0.0121] |
| dp_sigma_0p5 | discrete_semantic_entropy | +0.0551 | [+0.0168, +0.0953] |

## Limits and remaining work

- Labels remain AI-assisted with partial human review. Audit agreement after corrections is adjudicated agreement, not independent first-pass agreement. Question-wording concerns and incomplete extra reviews remain documented in review packages.
- Human decisions were merged per seed; this analysis verifies exact supplied review hashes, but does not establish semantic consistency across all differently spelled answers between seeds. A cross-seed label consistency audit remains appropriate before publication.
- These bootstrap intervals quantify question-sampling uncertainty conditional on the 15 fitted adapters and fixed labels. They do not establish variation across the full population of training runs or account for label/cluster uncertainty.
- This test split was previously used for Qwen, and the noise range was selected after development exploration. Describe it as the Llama test evaluation with prior exposure disclosed.
- These nine comparisons test noise versus clipping only, not direct differences between noise levels or superiority of one estimator over another.
- Original generation environments cannot be revalidated from these report bundles alone: analysis_inputs contains hashes, not the full generation identity files. Keep the original evaluation_identity.json and analysis_environment files in the reproducibility record.
- Epsilon values are per-run accounting, not privacy accounting for jointly releasing multiple adapters. Low noise can imply very weak privacy.

## Reproduction

Place the three source ZIPs in ../upload relative to this folder, then run python analyze.py with numpy, pandas and scikit-learn. results.json contains per-seed metrics, aggregate values, hashes and all primary contrasts.
