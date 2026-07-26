# Cross-survey CPU experiments

This directory implements the post-freeze programme specified in
`PROJECT_MEMORY/01_narrative/CROSS_SURVEY_BLUEPRINT.md`.

The first stages are deliberately LLM-free:

1. acquire and fingerprint public CES/ANES releases;
2. audit respondent linkage, pre/post coverage, missingness, and item types;
3. build a locked source/target catalogue;
4. estimate K-by-coverage identifiability with population, demographic, scalar,
   supervised, and latent-factor baselines;
5. run matched-person shuffles and warm-item ceilings before any GPU use.

Raw survey files are not committed. Every result records source URLs and
SHA-256 fingerprints. Restricted datasets are not copied here.

## Layout

```text
configs/       dataset-specific declarations
data/raw/      downloaded archives (ignored)
data/processed/respondent-item long tables (ignored until disclosure review)
metadata/      inventories, item catalogues, split fingerprints
scripts/       acquisition, audit, preprocessing, and CPU experiments
results/       phase outputs
tests/         deterministic unit tests
```

## Reproducibility

All scripts use seed 1701 unless overridden. The primary dataset is CES 2020;
ANES 2020 is the locked replication dataset and must not be used to redesign a
method after CES target results are inspected.
