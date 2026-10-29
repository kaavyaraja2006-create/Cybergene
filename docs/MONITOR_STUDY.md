# Monitor study (RQ7): triggers, delay prediction, certificate — controlled score-level simulation

Run: `python -m cgresearch.cli monitor-study -c configs/v2_monitor_study.yaml`. Numbers below: 2 seeds, 2 base laws (gamma, lognormal), 15 replicates/scenario, 6 harmful drifts (scale 1.1–2.0, location +0.5/+1.0), 2 harmless drifts (scale 0.8/0.6), stream length 4000, onset at 1000, reference n = 2000. Baselines are tuned to ≤ 1% false alarms on drift-free tuning streams. Raw tables: `docs/monitor_study_*.csv`. Synthetic score streams, **not** real traffic.

## A. Trigger comparison
| trigger | false alarm (no drift) | detect rate (ε=0 / 0.10) | median-of-mean delay, samples (ε=0 / 0.10) | alarm under HARMLESS drift (ε=0 / 0.10) |
|---|---|---|---|---|
| conformal martingale (ours) | 0.6% (bound δ = 1%) | 0.97 / 0.99 | 468 / 127 | 0.02 / 0.33 |
| certified-KS window (ours) | 0 | 0.94 / 1.00 | 418 / 239 | 0.00 / 0.52 |
| KS test (two-sided) | 0.8% | 0.97 / 1.00 | 386 / 222 | 1.00 / 1.00 |
| ADWIN (simplified) | 0 | 0.72 / 0.87 | 529 / 448 | 0.79 / 0.50 |
| Page-Hinkley (one-sided) | 0 | 0.94 / 1.00 | 564 / 335 | 0.00 / 0.00 |
| DDM on alert rate | 1.7% | 0.69 / 0.98 | 859 / 276 | 0.00 / 0.76 |

Reading, honestly:
- The martingale **is not uniformly the fastest**. At ε = 0 the two-sided KS test is quicker (386 vs 468); at ε = 0.10 the martingale is quickest (127). What it adds is an **any-time false-alarm guarantee (Ville)** that the tuned baselines only match empirically on the streams they were tuned on, and **one-sidedness**: it ignores harmless (downward) drift (2% alarms) where the two-sided KS test and ADWIN raise 100% / 79%.
- Page-Hinkley is also one-sided, never alarms on harmless drift and is competitive; it is slower and its false-alarm control is empirical only.
- **All label-free monitors fire when attacks arrive** (ε = 0.10 makes harmless-drift alarm rates 0.33–1.0, except Page-Hinkley): a mixture shift is indistinguishable from benign drift without labels. State this as a limitation: the monitor says "the score distribution moved up", not "benign traffic drifted".
- The original DDM-on-alert-rate had a degenerate minimum (fires on 20% of drift-free streams); fixed with Laplace smoothing and documented in the code. Report the baseline as modified.

## B. Delay prediction
`predicted_delay = ln(1/δ)/λ̂`, λ̂ = max over ε of the mean log-growth of the power martingale computed on a 1000-point post-drift window against the reference (label-free). Realised/predicted ratio, medians over 12 scenarios: **1.31 (range 0.94–1.91)** — the prediction is within a factor 2 but optimistic by ≈ 30%. This is a Wald-type heuristic; no tight proof is claimed.

## C. Certificate (`fpr_certificate`, R6)
Upper bound on the deployed **benign** FPR of a static conformal threshold from unlabelled data, valid with probability ≥ 1−δ (δ = 0.05) given an operator bound ε_max on the attack share of the window. It is the minimum of
(a) an **alert-rate bound** CP_upper(alerts, n)/(1−ε_max) — no distributional assumption, and
(b) a **shift bound** α_cal,ub + (D⁺_mix + DKW slack + ε_max·D_dom)/(1−ε_max).
Coverage over all drift × attack × ε cells: **≥ 0.979 (ε_max exact) and ≥ 0.985 (ε_max = 2ε+0.02)**, including `overlap` attacks that violate dominance. Tightness (bound − true FPR): +0.018 at ε = 0, +0.03–0.14 at ε = 0.10 (depends on attack strength: strong attacks raise the alert rate and loosen the bound), up to +0.29 at ε = 0.20.
Honest limitation: at n ≈ 1000–2000 the **shift bound is very loose (≈ +0.22–0.44 above truth)**, so the certificate's practical content is the alert-rate bound; the D⁺ decomposition remains the *explanation* of inflation (it predicts it, see `shift-monitor`), not a tight certificate. Do not oversell it in the paper.

*Re-run note:* after correcting the DKW constant (ln(2/δ) → ln(4/δ), see `docs/paper/theory_appendix.tex`) the study was re-run: certificate coverage stays ≥ 0.979 (exact ε_max) / ≥ 0.985 (loose), shift-bound slack +0.23 to +0.44; trigger results are unaffected (they do not use the slack except `cert_ks`, whose FPR/detection numbers did not change at the reported precision).
