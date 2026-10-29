# Bulk-anchored recalibration (BAR) — method note (R5)

## Problem
A drift alarm says the benign score distribution moved. Re-calibrating on the raw window is unsafe in a different way: the window contains attacks, they occupy the upper tail, and the (1−α) quantile is exactly where the threshold is read — so the threshold rises and recall collapses (observed: recall 0.15 vs 0.46 for an oracle at ε = 0.20 in the controlled study).

## Method
Use the part of the unlabelled window that attacks barely touch — its **lower `bulk` fraction (default 70%)** — to learn how benign scores moved, then carry that map into the tail.
1. Quantile pairs: x_u = Q_cal(u), y_u = Q_window(u(1−ε_hint)) for u ∈ [0.05, 0.7].
2. Fit a monotone map g (modes: `location`, `affine`, `loglinear`, `local`).
   `auto` picks the mode by **label-free extrapolation validation**: fit on u ≤ 0.55, test the error at 0.55–0.70, refit the winner on the full bulk.
3. Transport the reference sample: C′ = max(g(C), window bulk edge). Its conformal threshold equals g(old threshold). C′ also serves as the new reference for the monitor.
4. Optional **guard** (`guard_eps_max`): reject the transport if the window alert rate exceeds α + ε_max + slack and fall back to the raw window.

## What is and is not guaranteed
- **P1 (exactness).** If benign deployment scores are g(benign calibration scores) for a monotone g in the model class, and n → ∞, the transported threshold has FPR → α. (tests: `test_exact_recovery_*`, `test_transported_threshold_is_the_mapped_threshold`)
- **P2 (bulk quantiles are conservative under contamination).** If F_attack ≤ F_benign on the bulk range (attacks dominate), the window's bulk quantiles are biased *upward*: Q_W(u) ≥ Q_dep(u) (proved, Appendix). **Whether the extrapolated tail threshold is also conservative is NOT proved**: it holds in the tested affine-family simulations (FPR/α < 1) but fails for `overlap` attacks and non-affine drift. The price of the bias is recall; `eps_hint` (operator input) removes part of it. (test: `test_conservative_under_dominating_contamination` documents the empirical behaviour)
- **P3 (limit).** Two deployment laws that agree on the bulk but differ in the tail give *identical* bulk quantile pairs, so **no bulk-based method can see tail-only drift**. The guard detects it only when the excess alert rate is larger than the operator's ε_max. (test: `test_tail_only_drift_is_invisible_*`)
- Not guaranteed: finite-sample coverage (only asymptotic exactness plus empirical evidence), non-monotone detector behaviour, saturated scores.

## Negative results to report
- Estimating ε jointly with the shift from the bulk alone failed: ε is not identifiable (it is absorbed by the slope/intercept). Dropped; `eps_hint` is an operator input.
- On the prototype's Sentinel simulator BAR did **not** improve recall over naive recalibration: IsolationForest scores saturate (benign and attack scores both reach ≈0.233 after drift), so the score ordering itself is lost. This is a detector limitation, not a recalibration one.

## Controlled study (score-level, ground truth known) — `recal-study`
Run `python -m cgresearch.cli recal-study -c configs/v2_recal_study.yaml`. Factors: 3 base laws × 6 drifts (none, location, scale, affine, power, tail-only) × 3 attack types (strong, moderate, overlap with D_dom > 0) × ε ∈ {0, .02, .05, .10, .20}. Inference clusters = scenarios. Headline table: `docs/recal_study_summary.csv` (generated from the full config).

### Results of the full run (5 seeds × 100 replicates per cell; α = 0.05; ε ≥ 0.02 pooled unless stated)
| drift family | method | FPR/α | P(FPR > 1.5α) | recall |
|---|---|---|---|---|
| location / scale / affine | static | 5.25 | 1.00 | 0.65 |
| | naive (raw-window recalibration) | 0.63 | 0.00 | 0.31 |
| | **bar_auto** | 0.78 | 0.011 | 0.39 |
| | oracle (labels) | 1.00 | 0.00 | 0.46 |
| power (non-affine) | naive / **bar_auto** / oracle | 0.58 / 0.96 / 1.00 | 0.00 / 0.069 / 0.00 | 0.35 / 0.51 / 0.53 |
| tail-only | naive / bar_auto / bar_auto_guard | 0.71 / 1.61 / 1.40 | 0.00 / 0.71 / 0.53 | 0.24 / 0.43 / 0.40 |

At ε = 0.20 under affine-family drift: BAR recall 0.30 vs naive 0.15 (2.0×) with FPR/α 0.58 (safe side).
Other facts: `trim_known_eps` and `trim_2x_eps` (need the contamination level) violate the FPR target badly (P(FPR>1.5α) up to 1.0); `robust_alpha` is infeasible for ε ≥ α (recall 0); `bar_location` fails on scale drift; `bar_loglinear` fails on location drift — hence `auto`. Attack type `overlap` (D_dom > 0): BAR FPR/α 1.30 vs naive 1.04 at ε = 0.10, i.e. BAR is *not* safe when many attacks score below benign traffic.
Verdict: H6 holds for affine-family drift; it fails for tail-only drift (as proved, P3) and marginally for power drift (6.9% > 5%). The paper should present BAR as a safe-ish, recall-preserving alternative **under a stated drift class**, not as universally better than naive recalibration — naive is always the safer (more conservative) choice.
