# Notebook usage

These notebooks contain the final source snapshots supplied on
25 October 2026. Outputs and execution counts have been cleared.
All three are configured for seed 789; select the intended seed
before running other experiments.

## 01_train.ipynb
Runs the low-noise LoRA training suite.

## 02_evaluate_dev.ipynb
Evaluates uncertainty on the development set. Observed empty
samples are retained rather than silently resampled.
The source snapshot alone does not establish execution history.

## 03_evaluate_test.ipynb
Contains the held-out test workflow and reporting.

The ANALYSIS ONLY setup requires completed caches and checks
their identity and analysis dependencies. For existing caches,
run the setup and reporting Section 6; do not use “Run all”
without reviewing the execution flow.

The installation cell pins sentencepiece==0.2.0, which may fail
under Python 3.13. Use the original generation environment
for inference. Do not bypass identity checks or regenerate
samples to force compatibility with an existing cache.

## Reproducing saved results
For statistical analysis without a GPU or Colab, use the
analysis scripts and supplied report ZIPs, following the
repository README.

The notebooks are source snapshots, not a guarantee of
identical end-to-end retraining.