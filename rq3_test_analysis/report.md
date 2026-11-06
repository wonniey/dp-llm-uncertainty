# RQ3: selective answering at frozen development thresholds

945 shared included test questions; seeds 123, 456 and 789; five conditions and three estimators. The acceptance decisions and original point estimates were verified against saved reports and the frozen development protocol. No thresholds were retuned.

## Method

20,000 paired question-bootstrap draws (seed 20261024), using the same question resampling weights across every seed, condition and estimator. In each draw, compute coverage and accepted-answer error separately for each seed, then average equally over the three seeds. Risk is not pooled across seeds. No draw had zero accepted answers.

Both ordinary 95% percentile intervals and multiplicity-adjusted intervals are reported. The adjusted family contains all 30 quantities (15 coverage and 15 risk), using alpha=0.05/30, or nominal 99.8333% individual intervals. Approximate family-wise coverage is 95% under bootstrap assumptions. Extreme endpoints use about 17 draws per tail and have Monte Carlo uncertainty. This is a secondary analysis after inspecting results.

## Results: ordinary 95% intervals

| Condition | Estimator | Coverage [95% interval] | Error risk [95% interval] |
|---|---|---|---|
| standard_poisson | mean_token_entropy | 78.34% [76.01%, 80.67%] | 8.83% [7.07%, 10.69%] |
| standard_poisson | mean_answer_nll | 76.33% [73.97%, 78.62%] | 8.87% [7.08%, 10.77%] |
| standard_poisson | discrete_semantic_entropy | 83.99% [81.83%, 86.10%] | 8.99% [7.21%, 10.88%] |
| clipping_poisson | mean_token_entropy | 82.29% [80.00%, 84.51%] | 9.16% [7.37%, 11.03%] |
| clipping_poisson | mean_answer_nll | 77.92% [75.56%, 80.25%] | 7.82% [6.13%, 9.61%] |
| clipping_poisson | discrete_semantic_entropy | 81.69% [79.54%, 83.84%] | 8.90% [7.14%, 10.77%] |
| dp_sigma_0p1 | mean_token_entropy | 69.70% [66.95%, 72.52%] | 8.34% [6.36%, 10.42%] |
| dp_sigma_0p1 | mean_answer_nll | 74.04% [71.36%, 76.75%] | 8.38% [6.43%, 10.39%] |
| dp_sigma_0p1 | discrete_semantic_entropy | 78.77% [76.23%, 81.23%] | 7.70% [5.95%, 9.54%] |
| dp_sigma_0p25 | mean_token_entropy | 55.94% [52.84%, 59.01%] | 8.32% [6.09%, 10.63%] |
| dp_sigma_0p25 | mean_answer_nll | 61.69% [58.69%, 64.69%] | 9.71% [7.40%, 12.11%] |
| dp_sigma_0p25 | discrete_semantic_entropy | 69.84% [67.05%, 72.66%] | 6.29% [4.56%, 8.09%] |
| dp_sigma_0p5 | mean_token_entropy | 56.12% [53.02%, 59.26%] | 8.42% [6.20%, 10.75%] |
| dp_sigma_0p5 | mean_answer_nll | 56.30% [53.26%, 59.40%] | 8.95% [6.66%, 11.39%] |
| dp_sigma_0p5 | discrete_semantic_entropy | 67.23% [64.34%, 70.12%] | 5.66% [4.02%, 7.44%] |

## Multiplicity-adjusted intervals

| Condition | Estimator | Coverage interval | Risk interval |
|---|---|---|---|
| standard_poisson | mean_token_entropy | [74.53%, 81.98%] | [6.03%, 11.74%] |
| standard_poisson | mean_answer_nll | [72.62%, 79.96%] | [6.00%, 11.83%] |
| standard_poisson | discrete_semantic_entropy | [80.49%, 87.34%] | [6.08%, 12.03%] |
| clipping_poisson | mean_token_entropy | [78.66%, 85.75%] | [6.38%, 12.25%] |
| clipping_poisson | mean_answer_nll | [74.13%, 81.60%] | [5.15%, 10.69%] |
| clipping_poisson | discrete_semantic_entropy | [78.20%, 84.97%] | [6.13%, 12.00%] |
| dp_sigma_0p1 | mean_token_entropy | [65.10%, 74.07%] | [5.23%, 11.77%] |
| dp_sigma_0p1 | mean_answer_nll | [69.81%, 78.39%] | [5.33%, 11.59%] |
| dp_sigma_0p1 | discrete_semantic_entropy | [74.83%, 82.68%] | [4.88%, 10.66%] |
| dp_sigma_0p25 | mean_token_entropy | [50.93%, 60.72%] | [4.88%, 12.18%] |
| dp_sigma_0p25 | mean_answer_nll | [56.95%, 66.49%] | [6.17%, 13.72%] |
| dp_sigma_0p25 | discrete_semantic_entropy | [65.26%, 74.32%] | [3.70%, 9.30%] |
| dp_sigma_0p5 | mean_token_entropy | [51.06%, 61.06%] | [4.83%, 12.22%] |
| dp_sigma_0p5 | mean_answer_nll | [51.62%, 61.28%] | [5.29%, 12.82%] |
| dp_sigma_0p5 | discrete_semantic_entropy | [62.41%, 71.84%] | [3.19%, 8.40%] |

## Interpretation

All point-estimate mean test risks are below the 10% target. This alone does not establish that population risk is at most 10%. The target was an observed development-error criterion, not a formal risk-control guarantee. An upper interval below 10% provides conditional statistical evidence for the mean risk of these three fixed policies; it does not certify every seed or future run.

Semantic entropy retains numerically greater coverage in every noisy condition. No pairwise coverage superiority tests are performed here. More coverage at different achieved risks is not an equal-risk comparison.

These intervals condition on fitted adapters, fixed development thresholds, fixed correctness labels and clustering. They do not account for training variability, development-threshold selection uncertainty, label or clustering errors. Human review is partial and AI-assisted. The test split had prior Qwen exposure. Three seeds do not establish population-of-training-runs behavior.

## Reproduce

Extract the full archive and run `python rq3_test_analysis/analyze.py` with numpy, pandas and scikit-learn. No model reruns are needed. Source report ZIPs and reviewed label files are included for hash validation. Package versions are in analysis_environment.json.
