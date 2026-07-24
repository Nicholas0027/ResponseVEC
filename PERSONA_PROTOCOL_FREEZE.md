# Persona Silicon-Sampling Protocol Freeze

Frozen on 2026-07-24 after development fold 0 and before inspecting outer folds 1-5.

## Research Question

Under cold-respondent and cold-item evaluation, does an LLM persona choice head add predictive information beyond a supervised statistical choice head when both use behaviorally grounded respondent history?

## Fixed Data Protocol

- Use the semantic-cluster-disjoint item split saved by `build_item_split` with seed 1701.
- Select whole semantic clusters for the per-domain calibration pool; no semantic cluster may cross calibration and target roles.
- For outer fold `f`, test uses target fold `f`, validation uses `(f + 1) mod 6`, and training uses the other four target folds.
- Train, validation, and test respondents remain disjoint.
- Historical answers come only from the respondent's calibration items in the same respondent split.
- The primary history budget is K=5. K=0,1,3,8 are secondary router analyses only.
- The question-aligned shuffle preserves the calibration question and demographic stratum while breaking respondent-answer correspondence.

## Fixed Persona Model

- Use M=8 domain-specific behavior personas.
- Fit KMeans profiles and persona response distributions using train respondents and calibration questions only.
- Use Dirichlet smoothing alpha=0.5.
- Fit the demographic router from country, sex, age bin, education, and income quintile using train respondents only.
- Update the persona posterior with calibration history by Bayes' rule.

## Fixed Statistical Components

- Fit PCA rank 32 only on calibration and target-train option vectors; transform validation and test option vectors without refitting.
- The primary statistical component is `HistGradientBoostingClassifier` with 300 iterations, learning rate 0.05, 31 leaves, L2 regularization 1e-3, and seed 1701.
- HistGBDT uses demographic features, calibration-history mean, option vectors, uniform log prior, normalized option position/count, history-option dot product, and history-option distance.
- A calibration item used as a training target is excluded from its own history.
- The statistical persona head is a secondary decomposition baseline, not the primary hybrid component.

## Fixed LLM Component

- Use Qwen3-8B in bfloat16 with thinking disabled.
- Use the canonical survey prediction instruction and require exactly one option letter.
- Use the same M=8 behavior personas and history-updated persona posterior.
- Compute persona-conditional option-token probabilities once per prompt and mix them with the router posterior; do not inject history into the conditional persona prompt a second time.
- Preserve raw uncalibrated probabilities in the output.

## Fixed Calibration and Hybrid

- Split validation respondents deterministically into A/B with SHA-256 and seed 1701.
- Fit one global LLM temperature on validation-A only.
- Fit one global log-opinion-pool weight on validation-B only over the grid 0.00, 0.01, ..., 1.00.
- The primary method is `hist_gbdt_llm_hybrid`.
- The primary hybrid combines HistGBDT and temperature-scaled LLM persona-history probabilities geometrically.
- Apply the validation-selected temperature and weight to test predictions without reading test labels.
- Repeat this fixed A/B calibration procedure independently inside each confirmatory outer fold; do not alter its grid, components, or objective.

## Primary Metrics and Gates

- Primary metric: individual-level micro NLL.
- Secondary individual metrics: Brier score, accuracy, and ordinal-only RPS.
- Exploratory aggregate metric: survey-weighted normalized population Wasserstein distance.
- Use respondent-clustered paired bootstrap confidence intervals with seed 1701.
- Primary success gate: pooled folds 1-5 `hist_gbdt_llm_hybrid` NLL is lower than HistGBDT and the 95% paired-bootstrap interval for NLL difference excludes zero.
- Router validity gate: real history beats question-aligned shuffled history in pooled folds 1-5 with a 95% interval excluding zero.
- Report all folds and pooled results even if either gate fails.

## Locked Development Evidence

- Fold-0 HistGBDT NLL: 1.4723.
- Fold-0 LLM persona-history calibrated NLL: 1.4747.
- Fold-0 HistGBDT-LLM hybrid NLL: 1.4482.
- Hybrid versus HistGBDT NLL difference: -0.0241, 95% CI [-0.0371, -0.0119].
- Hybrid versus LLM NLL difference: -0.0265, 95% CI [-0.0499, -0.0025].
- Fold-0 hybrid weight selected on validation-B: 0.33.

No architecture, split, prompt, feature, temperature procedure, hybrid grid, metric, or gate may be changed after inspecting folds 1-5. Any post-freeze analysis must be labeled exploratory.
