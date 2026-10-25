# Gaussian Fragility Audit Protocol
Version: gaussian_v1
Status: Reference protocol

## 1. Objective

Evaluate the stability of published regression estimates and
substantive claims under standardized Gaussian perturbations
of observed explanatory variables.

Gaussian perturbations are numerical sensitivity tests, not
estimates of actual historical measurement error.

## 2. Baseline replication

Before perturbation:
- Reproduce the original estimation sample.
- Reproduce point estimates and standard errors.
- Preserve the original fixed effects and clustering procedure.
- Document any discrepancies from the published results.
- Assign persistent identifiers to models and coefficients.

## 3. Perturbation designs

### A. Structure-preserving Gaussian perturbations

Perturb original measured explanatory variables:

    X_ij(r) = X_ij(0) + epsilon_ij(r)

    epsilon_ij(r) ~ N(0, delta^2 * s_j^2)

Here, s_j is the standard deviation across independent
measurement units.

Apply one perturbation per measurement unit and propagate
that value to all observations sharing the unit.

Reconstruct interactions and other deterministic transformations
from perturbed underlying variables.

Preserve outcomes, fixed effects, treatment-period indicators,
and original regression samples unless explicitly specified.

### B. Unrestricted design-matrix perturbations

Add Gaussian noise directly to selected design-matrix columns,
using their original standard deviations.

This design does not necessarily preserve algebraic
relationships among regressors.

Report results separately from structured perturbations.

## 4. Standard noise levels

    delta = 0.001, 0.01, 0.025, 0.05, 0.10

Use 1,000 simulations per level by default.

Record random seeds and synchronize draws across models
whenever joint claim-level inference requires them.

## 5. Categorical transformations

If a categorical variable is reconstructed from an underlying
continuous measurement, document whether its categories
are held fixed or recomputed after perturbation.

If recomputed, reproduce the original classification exactly
at zero perturbation before proceeding.

Different choices constitute separate perturbation designs.

## 6. Coefficient-level diagnostics

For each coefficient and perturbation level report:

- Baseline coefficient, SE, t-statistic, and p-value.
- Baseline inferential margin.
- Sign stability.
- Inferential stability at a frozen significance threshold.
- Mean and SD of perturbed coefficient estimates.
- Perturbation quantiles.
- Majority-failure indicator.

For originally significant coefficients, inferential failure
means losing significance.

For originally insignificant coefficients, inferential failure
means gaining significance.

Keep those two cases distinguishable.

## 7. Claim-level diagnostics

Define and freeze substantive claims before calculating
claim-level stability.

Record supporting model/coefficient IDs, predicted signs,
significance thresholds, and whether all supporting conditions
must hold jointly.

Claim stability is the fraction of simulation draws in which
the complete frozen claim remains supported.

Do not combine independently generated draws as though
they represent synchronized perturbations.

## 8. Breakdown points

A majority-failure breakdown occurs when claim or
coefficient stability falls below 0.5.

Report:
- First tested majority-failure level.
- Last tested nonfailure level.
- Refined/interpolated threshold, where available.
- Number of simulations.
- Monte Carlo uncertainty.

Do not treat an interpolated crossing as an exact threshold.

## 9. Archival requirements

Preserve:
- Original article and replication materials.
- Executable audit code.
- Model, coefficient, claim, and perturbation registries.
- Individual simulation draws.
- Coefficient- and claim-level summaries.
- Breakdown results.
- Audit manifest and file hashes.

## 10. Cross-paper aggregation

Aggregate separately by perturbation design and noise level.

Report denominators explicitly.

Distinguish:
- All estimated coefficients.
- Audited target coefficients.
- Originally significant audited coefficients.
- Substantive claims.

Do not treat coefficient occurrences or multiple claims
within a paper as statistically independent observations.

## 11. Limitations

This protocol assesses sensitivity under stipulated
perturbations. It does not by itself establish:
- Actual historical measurement error.
- Causal validity of the original research design.
- Numerical ill-conditioning as the cause of fragility.

Conditioning and singular-value diagnostics should be
reported separately using documented scaling conventions.
