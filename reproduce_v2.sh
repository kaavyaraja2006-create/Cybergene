#!/usr/bin/env bash
# CYBERGENE v2 -- calibration under shift. Usage: ./reproduce_v2.sh [--smoke] [dataset ...]
# datasets: unsw_raw unsw_train cic2017 cic2018 nf_unsw   (default: those whose data/ files exist)
# --smoke = tiny SYNTHETIC data, outputs named SMOKE_*, NOT results.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SMOKE=""; DS=()
for a in "$@"; do [[ "$a" == "--smoke" ]] && SMOKE="--smoke" || DS+=("$a"); done
python -m pytest -q tests
cg() { python -W ignore -m cgresearch.cli "$@" $SMOKE; }
if [[ -n "$SMOKE" ]]; then
  for c in calib-transfer shift-monitor contamination protocol-sens sentinel-case recal-study monitor-study; do cg $c; done; exit 0
fi
have() { case "$1" in unsw_raw) [[ -f data/UNSW-NB15/UNSW-NB15_1.csv ]];; unsw_train) [[ -f data/UNSW-NB15/UNSW_NB15_training-set.csv ]];;
  cic2017) [[ -d data/CIC-IDS2017 ]];; cic2018) [[ -d data/CIC-IDS2018 ]];; nf_unsw) [[ -f data/NF-UNSW-NB15-v2.csv ]];; esac; }
[[ ${#DS[@]} -eq 0 ]] && DS=(unsw_raw unsw_train cic2017 cic2018 nf_unsw)
for d in "${DS[@]}"; do
  have "$d" || { echo "skip $d (data missing)"; continue; }
  cg calib-transfer -c configs/v2_${d}_calib_transfer.yaml
  cg shift-monitor  -c configs/v2_${d}_shift_monitor.yaml
  cg contamination  -c configs/v2_${d}_contamination.yaml
  cg protocol-sens  -c configs/v2_${d}_protocol_sens.yaml
done
cg recal-study -c configs/v2_recal_study.yaml
cg monitor-study -c configs/v2_monitor_study.yaml
cg sentinel-case -c configs/v2_sentinel_case.yaml
python tools/check_run.py results || true
python tools/pack_results.py results cybergene_results_to_send.zip
