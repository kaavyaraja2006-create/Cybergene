# Data guide
| Key | Files | Time info used for temporal scenario | Notes |
|---|---|---|---|
| unsw_raw | UNSW-NB15_1.csv + NUSW-NB15_features.csv | `stime`, 5 equal-count blocks | headerless CSV; features file supplies names |
| unsw_train | UNSW_NB15_training-set.csv | none (row order) | official partition; random + LOFO only for claims |
| cic2017 | all day CSVs | source file → `day` (Mon…Fri); Monday is benign-only → fit block | state which release (GeneratedLabelledFlows vs MachineLearningCSV); blank-label rows dropped and counted |
| cic2018 | processed CSVs | file → `day` | repeated header rows removed |
| nf_unsw | NF-UNSW-NB15-v2 csv | `FLOW_START_MILLISECONDS` if present | untested on real files — run the fixture test, inspect `info.alignment` |
Dedup is applied to the **full file before sampling** (`dedup_first: true`); the reported duplicate rate then describes the dataset, not the subsample.
