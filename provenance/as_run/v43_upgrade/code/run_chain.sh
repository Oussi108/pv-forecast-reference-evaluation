#!/usr/bin/env bash
# Sequential v43 training chain (frozen protocol). Each step is resumable.
set -e
W="E:/master in china/s2/research/solar energy forecasting"
R="$W/reports/paper1_v43_upgrade_20260926"
S="$W/WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting/scripts"
export PV_FORECAST_PROTOCOL_PATH="$R/protocol/protocol_v43_pvdaq9068.json" PYTHONDONTWRITEBYTECODE=1
echo "STEP train_point $(date -u +%FT%TZ)"
python "$S/train_corrected_study.py" --private-root "$W/tmp/paper1_v43_private/prepared" --stage point --beds PVDAQ9068
echo "STEP train_quantile $(date -u +%FT%TZ)"
python "$S/train_corrected_study.py" --private-root "$W/tmp/paper1_v43_private/prepared" --stage quantile --beds PVDAQ9068
echo "STEP updates $(date -u +%FT%TZ)"
python "$R/code/u01_updates_9068.py"
echo "STEP tuning $(date -u +%FT%TZ)"
python "$R/code/t01_neural_tuning.py" PVDAQ9068 DKASC HKUST
echo "CHAIN_COMPLETE $(date -u +%FT%TZ)"
