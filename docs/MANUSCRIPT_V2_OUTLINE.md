# Manuscript outline (title keeps CYBERGENE)
**CYBERGENE: Does a Benign-Calibrated Intrusion-Detection Threshold Survive Duplicates, Unseen Attacks and Time?** (working title)

1. **Introduction.** Contribution list: (a) exact shift decomposition + monitorable gap D⁺ (R3); (b) finite-sample/PAC thresholds (R1–R2); (c) contamination safety condition (R4); (d') bulk-anchored recalibration (R5): label-free threshold transport that keeps recall when the alarm window contains attacks, with an explicit impossibility result for tail-only drift (P3); (d'') label-free FPR certificate (alert-rate + shift bounds) and a predicted-delay formula for the monitor; (d) conformal test martingale with Ville false-alarm bound as label-free recalibration trigger; (e) leakage-safe four-role evaluation harness; cluster-level inference; (f) multi-dataset measurement + CYBERGENE prototype integration.
2. **Related work.** Dos and Don'ts of ML in computer security (Arp et al., 2022); CICIDS2017 problems (Engelen et al., 2021); NetFlow-v2 datasets (Sarhan et al.); Kitsune (Mirsky et al., 2018); conformal outlier testing (Bates et al., 2023); conformal under covariate shift (Tibshirani et al., 2019); testing exchangeability online / plug-in martingales (Vovk et al.; Fedorova et al., 2012). *Verify every citation and add 2024–2026 work before submission.*
3. **Method.** Roles and scenarios; threshold methods; R1–R4 with proofs in appendix; monitors; policies.
4. **Experimental protocol.** Datasets, dedup-first, detectors, label-free tuning, clusters, statistics, preregistration link.
5. **Results.** (add: RQ6 controlled recalibration study table + per-drift figure.) RQ1/2 Table (violation rate, FPR/α by scenario×method, CI); D⁺ vs inflation figure; RQ3 policy table + false-alarm control; RQ4 ε-curves and D_dom per family; RQ5 rank stability; harness-bug sensitivity (v1 vs v2 protocol).
6. **CYBERGENE case study.** CalibratedSentinelAgent vs original on simulator with drift (mechanism demo; states the recall cost when recalibrating on contaminated pools).
7. **Limitations / threats.** Score-level (not feature-level) shift; exchangeability inside windows; benign labels assumed clean in cal; dataset artefacts; simulator is synthetic.
8. **Conclusion; artifact statement** (code, prereg, configs, hashes, Zenodo DOI).
Appendices: proofs R1–R4, full tables, deviations from preregistration.
