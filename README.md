# CYBERGENE v2 — Calibration under Shift

v2 adds: `cgresearch/calibration.py` (conformal / PAC thresholds, shift decomposition, contamination bound), `monitor.py` (conformal test martingale, KS monitor, recalibration policies), `scenarios.py` (fit/tune/cal/test roles; random / leave-one-family-out / temporal), `stats_cluster.py` (cluster-level inference), new detectors (PCA, HBOS, ECOD, KitNET-style), label-free tuning, and five experiments:
`calib-transfer | shift-monitor | contamination | protocol-sens | sentinel-case | recal-study | monitor-study`; the new label-free **bulk-anchored recalibration** is documented in `docs/BAR_METHOD.md`; monitor baselines, delay prediction and the FPR certificate in `docs/MONITOR_STUDY.md`.
Proofs: `docs/paper/theory_appendix.pdf` (LaTeX source alongside); related-work notes with verified references: `docs/paper/related_work.md`; Windows runner `reproduce_v2.ps1`, sanity checker `tools/check_run.py`, results packer `tools/pack_results.py`. Start with `docs/RUNBOOK.md`; hypotheses in `docs/PREREGISTRATION.md`; run `./reproduce_v2.sh --smoke` first. Smoke outputs are synthetic and are not results.

---
# CYBERGENE research harness

Reproducible, auditable experiments for the CYBERGENE paper. **No result in this repository is
real until it was produced by a non-`SMOKE_` run on the real dataset; `--smoke` and all unit tests use
synthetic data and must never be quoted.** Every run writes `manifest.json` (experiment ID, UTC time,
git commit, Python/package versions, OS, CPU/GPU, dataset path/version/SHA-256, seeds, CLI args,
config SHA-256, runtime) plus raw per-seed tables (CSV+JSON) and artifacts.

## Quick start
**Linux / macOS (bash)**
```bash
cd research
pip install -r requirements.txt           # optional extras: pip install -r requirements-optional.txt
python -m pytest -q tests                 # 39 tests
./reproduce.sh --smoke                    # tiny SYNTHETIC run of everything incl. report export (~6 min)
./reproduce.sh                            # real run (needs the data below)
```
**Windows (PowerShell)**
```powershell
cd research
pip install -r requirements.txt           # hnswlib/xgboost are optional (hnswlib needs MSVC build tools)
python -m pytest -q tests
.\reproduce.ps1 -Smoke                    # plumbing check on synthetic data
.\reproduce.ps1                           # real run
```
Data (not redistributed): `data/UNSW-NB15/{UNSW-NB15_1.csv, NUSW-NB15_features.csv, UNSW_NB15_training-set.csv}`
and `data/CIC-IDS2017/*.csv`.

## Status (what exists vs. what is still required)
| # | Item | Code | Tested on synthetic/fixtures | Real-data evidence |
|---|------|------|---|---|
| 1,2 | 30 paired seeds, all metrics, bootstrap/Wilcoxon-Holm/Cliff, FPR sweep + 5 curve types | done | yes | **pending** (run `unsw_sampleA/B_detection`) |
| 3 | IF diagnosis: orientation, histograms, per-seed ROC/PR, top-50 records, port variants (raw / separate_std / drop / one-hot), rule-based verdict | done | yes | **pending** |
| 4 | Leakage audit (exact + near, cross-split), dedup rerun, original-vs-dedup | done | yes | **pending** |
| 5 | LogReg, RF, LightGBM (+XGBoost optional), autoencoder, rule-based, flag-all/none | done | yes | **pending** |
| 5b | Kitsune | **not implemented** (needs packet-level features); autoencoder is a documented stand-in | - | future |
| 6 | CIC-IDS2017/2018 loader with explicit alias/unit alignment, same pipeline | done | yes (fixture CSVs in both header styles) | **pending**, needs CIC files |
| 7 | Mutation: 7 policies, 5 reasons, paired episodes, independent ground-truth env | done (simulation) | yes | simulation only |
| 8 | Retrieval: synthetic (circular) AND weak-label real-flow retrieval | done | yes | **pending**; expert labels still future |
| 9 | ANN scaling to 100k (hnswlib; faiss if installed) | done | smoke only | **pending** |
| 10 | Drift / poisoning / adversarial telemetry | done (simulated) | yes | **pending** |
| 11 | Manifests, JSON/CSV, requirements, README, LICENSE placeholder, reproduce.sh | done | yes | n/a |
| 12 | LaTeX tables + PDF/PNG figures with captions/labels/traceability comments | done | compiled with pdflatex | n/a (renders real runs once they exist) |

## Aligning with the draft paper
* Sample A = stratified 20,000-row sample of `UNSW-NB15_1.csv` (`sample_n`, `sample_seed`; **confirm the seed
  used originally**, it is not recorded in the draft); Sample B = `UNSW_NB15_training-set.csv` (no ports).
* Legacy protocol (draft Eq. 4): threshold = (1-c) quantile of benign TRAIN scores, c=0.15
  (`legacy_contamination`), reported next to the validation-calibrated thresholds, with measured AND implied FPR
  (Eq. 5) and a paired legacy-vs-calibrated test table.
* RQ2 standardisation ablation: `variants` command (paired by seed).
* Flag-all / flag-none reference rules (draft notes flag-all F1 = 0.810 on the 68.1% release).
* Split protocol differs from the draft (it used a stratified `train_test_split`): here benign 60/20/20 and
  attacks 40/20/40 with a benign validation set for calibration. State this when comparing numbers.
* Mutation: the draft describes response parameters (aggressiveness, isolation_duration, monitoring_intensity) that are
  NOT in the v2 backend; this harness mutates DNA genes via `EvolutionEngine`. If your newer backend has the response
  configuration, swap the gene<->parameter map in `experiments/mutation.py` (`_dna_from_u` / `_u_from_dna`).

## Layout
```
research/
  README.md requirements.txt LICENSE reproduce.sh .gitignore
  configs/            <dataset>_{detection,if_diagnostics,dedup,standardization,retrieval_real}.yaml, mutation, retrieval, ann, robustness
  cgresearch/
    cli.py            single entry point (+ `index` command -> results/INDEX.json)
    provenance.py     env/git/dataset SHA-256/config hash
    runlog.py         ExperimentRun: manifest + tables + artifacts
    metrics.py        orientation contract, benign-val threshold calibration, metrics, curves
    stats.py          bootstrap CI, Wilcoxon, Holm-Bonferroni, Cliff's delta
    preprocess.py     identical preprocessing, port_mode raw|drop|onehot|onehot_scaled
    data/             unsw.py cic.py (loaders) synthetic.py (SMOKE ONLY) audit.py splits.py
    report.py         LaTeX tables + PDF/PNG figures -> paper/tables, paper/figures (+INDEX.csv)
    detectors/        models.py (IF, OCSVM, LOF, AE, rule, LR, RF, LGBM, XGB) tuning.py
    experiments/      detection variants if_diag dedup mutation corpus retrieval retrieval_real ann_scaling robustness
  tests/              orientation/threshold, splits/audit/stats, integration
  results/<experiment>/<experiment_id>/{manifest.json,config.json,tables/,artifacts/}
```
Backend changes (backward compatible): `EvolutionEngine(birth_penalty=0.05)`; mutation branches for
`high_business_impact`, `dependency_failure`, `repeated_identical_failure`; `novel_action_seen` now
actually changes the DNA (it previously only recorded a label); `AttackDNA.vector(genomes=...)`
for genome ablation (default vector unchanged, covered by a test).

## Commands (all accept `--seeds 0-29`, `--set key.sub=value`, `--out DIR`, `--smoke`)
```
python -m cgresearch.cli if-diagnostics -c configs/unsw_sampleA_if_diagnostics.yaml           # run on ORIGINAL data first
python -m cgresearch.cli if-diagnostics -c configs/unsw_sampleA_if_diagnostics.yaml --set dedup=exact
python -m cgresearch.cli audit          -c configs/unsw_sampleA_dedup.yaml                    # leakage report only
python -m cgresearch.cli dedup          -c configs/unsw_sampleA_dedup.yaml                    # none/exact/near full reruns
python -m cgresearch.cli detection      -c configs/unsw_sampleA_detection.yaml                # detection + baselines + FPR sweep
python -m cgresearch.cli detection      -c configs/unsw_sampleB_detection.yaml
python -m cgresearch.cli detection      -c configs/cic2017_detection.yaml                     # second dataset, reported separately
python -m cgresearch.cli variants       -c configs/unsw_sampleA_standardization.yaml          # RQ2
python -m cgresearch.cli mutation       -c configs/mutation.yaml
python -m cgresearch.cli retrieval      -c configs/retrieval.yaml                             # synthetic ground truth (circular)
python -m cgresearch.cli retrieval-real -c configs/unsw_sampleA_retrieval_real.yaml           # weak real labels
python -m cgresearch.cli ann            -c configs/ann.yaml
python -m cgresearch.cli robustness     -c configs/robustness.yaml
python -m cgresearch.cli index  --out results                                                 # results/INDEX.json
python -m cgresearch.cli report --out results --paper paper                                   # LaTeX + figures
```
`report` skips SMOKE/synthetic runs unless `--include-smoke` (then every caption says NOT A RESULT). Each
`.tex` begins with `% experiment_id`, `% config_sha256`, `% sources`; `paper/tables/INDEX.csv` maps every
table/figure to its run, so any number in the paper traces to a manifest.

## Protocol guarantees (enforced in code/tests)
* Split depends only on (labels, seed): identical for every model (`split_hash` stored per row).
  Benign 60/20/20 train/val/test, attacks 40/20/40. Unsupervised fit on benign train only.
* Dedup happens **before** splitting; `leakage_report.json` quantifies cross-split copies.
* Alert rule `score > threshold`; **higher score = more anomalous** for every model; sklearn's
  `predict()` is never used for evaluation. Threshold = benign-validation quantile; measured FPR on test
  and tie fraction at the threshold are reported.
* Hyperparameters chosen by validation PR-AUC only (`tune()` has no test argument).
* Statistics per metric: mean, SD, median, 95% percentile bootstrap CI (10,000), per-seed values;
  paired Wilcoxon (seeds intersected), Holm-Bonferroni within each metric/FPR family, Cliff's delta.
  Degenerate cases yield p=1 with a note, never invented values.

## Isolation Forest: hypotheses the diagnostic tests (not conclusions)
AUC 0.777 with recall 0.019 @5% FPR and negative MCC means attacks sit mid-distribution, above the
benign median but below the benign 95th percentile, or the alert rule disagrees with the AUC score.
`if-diagnostics` records, per seed and port treatment: AUC for all four sklearn orientations
(`score_samples`, `-score_samples`, `decision_function`, `-decision_function`); whether `predict()`
(contamination threshold learned on train) differs from the benign-validation threshold; tie mass at
the threshold; class score quantiles; an "INVERTED" counterfactual showing what a sign bug looks like;
top-scoring benign/attack rows. Candidate causes to confirm or reject: (a) raw port numbers as
magnitudes, (b) dense repeated attack flows (DoS/generic) looking "normal" to IF, (c) contamination-
based `predict()` used for the reported recall/MCC, (d) heavy-tailed unscaled features,
(e) duplicates across splits inflating benign density. Note: IF is invariant to per-feature affine
scaling, so `onehot` and `onehot_scaled` should match for IF; a difference means a bug.

## Mutation experiment: caveats you must state in the paper
Outcomes come from a hidden ground-truth simulator independent of the Response Engine/Digital Twin
(the engine only names the action in the log). Results depend on `direction_prior` (probability the
failure reason points the right way) and `reason_noise`; the sensitivity sweep (0.5/0.7/0.9) is part
of the experiment. With prior 0.5 "targeted" has no informational advantage by construction. Report
fitness with and without the 0.05 birth penalty (both columns are logged).

## Result schemas (columns)
* `manifest.json`: experiment_id, experiment, timestamp_utc, git{commit,dirty}, environment{python,os,cpu,gpu,packages},
  dataset{path,version,sha256,files,synthetic}, seeds, cli_args, config_sha256, dedup, status, runtime_seconds, result_files.
* `tables/per_seed_fpr_sweep`: seed, model, split_hash, params, supervised, target_fpr, threshold, precision, recall, f1, mcc,
  measured_fpr, alerts, alert_rate, tp, fp, fn, tn, roc_auc, pr_auc, tie_fraction_at_threshold_test, latency_us_per_record.
* `tables/per_seed_threshold_free`: seed, model, split_hash, params, roc_auc, pr_auc, fit_seconds, batch_score_us_per_record,
  single_record_latency_us_{median,p95}, n_test, n_test_attack.
* `tables/summary_*`: metric, [target_fpr], model, n, mean, std, median, ci_low, ci_high, per_seed, seeds.
* `tables/comparisons_*`: metric, [target_fpr], model_a, model_b, n_pairs, mean_diff_a_minus_b, median_diff, diff_ci_low/high,
  wilcoxon_stat, p_value, p_holm, cliffs_delta, cliffs_magnitude, note.
* `tables/generation_log` (mutation): strategy, seed, scenario, episode, true_reason, reported_reason, action, generation,
  params_before, params_after, mutation_applied, failure_reason_used, p_success_true, simulated_outcome_true, observed_success,
  containment_status, fitness_with_birth_penalty, fitness_without_birth_penalty, direction_prior.
* `tables/episodes`, `per_seed_policy`, `by_reason`, `summary`, `comparisons_vs_retry` (mutation).
* `tables/per_seed` + `summary` + `comparisons` (retrieval): P@1/5/10, R@5/10, MRR, nDCG@10, latency, memory.
* `tables/per_seed_method`, `per_query_latency`, `summary_by_method_size` (ANN): latency median/p95/p99, qps, build_seconds,
  index_memory_bytes, overlap_at_10, recall_vs_exact_at_10, n_queries.
* `tables/drift|poisoning|adversarial` and `summary_*`, `comparisons_poisoning` (robustness).
* `artifacts/`: split_seed*.npz, curves_seed*.npz (ROC, PR, recall/precision/F1-vs-FPR), score histograms, PNG curve bands.

## Checklist before journal submission
- [ ] Obtain UNSW-NB15 and CIC-IDS2017; record versions/SHA-256 (manifest); confirm Sample A seed and the original split/test fraction.
- [ ] Run `if-diagnostics` on original data; explain the IF anomaly with the recorded evidence; fix and rerun.
- [ ] Leakage report on the full data; rerun everything on the deduplicated data; report both.
- [ ] 30+ paired seeds for every detector/baseline; all four target FPRs; stats tables.
- [ ] Hyperparameter grids justified and validation-only; document final grids.
- [ ] Add Kitsune (or justify the autoencoder) and state exactly what was run.
- [ ] Run everything on CIC-IDS2017 too and report per dataset (no merging); check CIC label/flow-construction issues (Engelen 2021).
- [ ] Replace generator-labelled retrieval ground truth with expert/real incident labels, or state the limitation.
- [ ] Mutation: report sensitivity to direction_prior/reason_noise; call it simulation; consider a real response testbed.
- [ ] ANN: run full 1k-100k on the target hardware; record hardware; >=200 queries per config.
- [ ] Robustness runs on real data (drift needs timestamps); adaptive-attacker caveats stated.
- [ ] Every number in the paper maps to `experiment_id` + config hash + seeds + result file (use `results/INDEX.json`).
- [ ] Choose license; tag the git commit used for the final runs (repo is currently not under git).
