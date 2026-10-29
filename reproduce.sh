#!/usr/bin/env bash
# Reproduce every experiment. Usage: ./reproduce.sh [--smoke]
# Real runs need the datasets (not redistributed):
#   data/UNSW-NB15/{UNSW-NB15_1.csv,NUSW-NB15_features.csv,UNSW_NB15_training-set.csv}
#   data/CIC-IDS2017/*.csv
# --smoke uses tiny SYNTHETIC data: plumbing check only, outputs are named SMOKE_* and are NOT results.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SMOKE="${1:-}"
python -m pytest -q tests
cg() { python -W ignore -m cgresearch.cli "$@" ${SMOKE:+--smoke}; }
# dataset-free experiments
cg mutation  -c configs/mutation.yaml
cg retrieval -c configs/retrieval.yaml
cg ann       -c configs/ann.yaml
run_dataset() {  # $1 = config prefix
  cg if-diagnostics -c configs/$1_if_diagnostics.yaml
  cg if-diagnostics -c configs/$1_if_diagnostics.yaml --set dedup=exact --set name=$1_if_diagnostics_dedup
  cg dedup          -c configs/$1_dedup.yaml
  cg detection      -c configs/$1_detection.yaml
  [[ -f configs/$1_standardization.yaml ]] && cg variants -c configs/$1_standardization.yaml
  [[ -f configs/$1_retrieval_real.yaml ]] && cg retrieval-real -c configs/$1_retrieval_real.yaml
  return 0
}
if [[ -n "$SMOKE" || -d data/UNSW-NB15 ]]; then
  run_dataset unsw_sampleA; run_dataset unsw_sampleB
  cg robustness -c configs/robustness.yaml
fi
if [[ -n "$SMOKE" || -d data/CIC-IDS2017 ]]; then
  cg if-diagnostics -c configs/cic2017_if_diagnostics.yaml
  cg detection -c configs/cic2017_detection.yaml
fi
python -m cgresearch.cli index --out results
python -m cgresearch.cli report --out results --paper paper ${SMOKE:+--include-smoke}
