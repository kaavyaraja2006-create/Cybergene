# Pre-registration — CYBERGENE: Calibration under Shift

Freeze this file (commit + tag `prereg-v1`) **before** running any real-data experiment. Anything not listed here is exploratory and must be labelled so in the paper.

## Question
When a threshold is calibrated on benign scores to hit a target false-positive rate α, does the FPR survive (i) duplicates, (ii) unseen attack families, (iii) time? Can a label-free monitor tell, and does acting on it help?

## Design (fixed)
- Datasets (reported separately, never pooled): UNSW-NB15, CIC-IDS2017, optionally CIC-IDS2018 and NF-UNSW-NB15-v2.
- Roles: fit / tune / cal / test are disjoint; calibration data is never used for fitting or selection (enforced by `Fold.check`).
- Dedup before sampling (`dedup_first: true`) for headline numbers; dedup-after as the sensitivity arm.
- Detectors (unsupervised): IsolationForest, OCSVM, LOF, autoencoder, PCA, HBOS, ECOD, KitNET-style. Supervised references: logreg, LightGBM.
- Tuning: random grid n=12, label-free pseudo-anomaly criterion (headline); v1 first-k labelled tuning only in RQ5.
- α ∈ {0.01, 0.05}; methods: empirical (v1), split-conformal (R1), PAC-conformal δ=0.05 (R2). Seeds 0–9.
- **Unit of inference = cluster** (day, attack family, dataset), never seed. Cluster-bootstrap 95% CIs (10 000 resamples); paired sign-flip tests.
- Violation = realised benign FPR > α; *material* violation = FPR > 1.5α.

## Hypotheses
- **H1 (transfer fails under shift).** Under random splits all three methods have violation rate ≈ their nominal level (conformal ≤ ~0.5 marginally, PAC ≤ δ). Under temporal and leave-one-family-out windows the violation rate and mean FPR/α are higher than under random splits for the same detector (paired cluster test, one-sided, α=0.05).
- **H2 (the gap explains it).** The identity FPR = α_cal + shift term holds to numerical precision (max abs error < 1e-9) and the unlabelled D⁺ is positively rank-correlated with FPR inflation (Spearman ρ > 0.5, pooled over windows within detector).
- **H3 (monitor helps).** `martingale` and `ks_certified` reduce the temporal violation rate relative to `static` without labels, at a recall cost no larger than `unlabeled_periodic`. The martingale false-alarm rate on exchangeable streams is ≤ δ=0.01 (Ville).
- **H4 (contamination).** For detectors with D_dom ≈ 0, contamination ε ≤ 0.10 does not raise FPR above α (only lowers recall); the measured FPR stays ≤ α + ε·D_dom + DKW slack in ≥ 95% of cells. Detectors with D_dom > 0 are expected to violate it as ε grows.
- **H5 (protocol matters).** The v1 protocol (first-k grid, labelled tuning) changes the winner or Kendall τ of detector rankings versus the label-free random-grid protocol in at least one scenario/dataset (reported as rank-stability, Friedman/Nemenyi).
- **H6 (label-free recalibration without the recall collapse).** In the controlled score-level study (`recal-study`), for the affine-family drifts (location/scale/affine) and ε ∈ [0.02, 0.20], `bar_auto` has |FPR − α|/α smaller than `static` and a material-violation rate (FPR > 1.5α) ≤ 5%, with mean recall ≥ 1.3× `naive` at ε = 0.20. Expected failure, stated in advance: tail-only drift (bulk-invisible), where the guard is the only defence. On real data, `martingale_bar` is compared with `martingale` (naive) on FPR violation rate and recall (paired cluster test over windows).
- **H7 (monitor).** (i) The martingale's false-alarm rate on drift-free streams is ≤ δ = 0.01; (ii) its detection rate is ≥ 0.9 for scale drift ≥ 1.25 and its delay is within a factor 2 of `predicted_delay`; (iii) its alarm rate under *harmless* (downward) drift is ≤ 0.05, lower than the two-sided KS test and ADWIN; (iv) it is not required to be faster than the best baseline (reported either way).
- **H8 (certificate).** `fpr_certificate` covers the true deployed benign FPR in ≥ 95% of cells for every attack type (including dominance-violating ones) when ε_max ≥ the true attack share; the shift-bound component is expected to be loose and is reported as such.

## Falsification / reporting rules
- Null or contrary results are reported as such. No hypothesis is dropped after seeing data.
- Deviations (changed grid, excluded detector, reduced seeds) are listed in a "Deviations" table in the paper.
- SMOKE_* outputs are synthetic plumbing checks and are never cited.
- The Sentinel case study (backend simulator) is a mechanism demonstration only.
