#!/bin/bash
# Re-runs the unit commitment of every case whose hourly results are published in results/.
# Each case is a full year (366 days); two cases run in parallel. On a recent desktop a case takes
# from about one hour to several hours; the whole set takes a few days.
#
# Usage, from the repository root, after setup.sh:
#   PY=python3 bash run_cases.sh              all cases
#   PY=python3 bash run_cases.sh rocof1_dyn_mp  one case, by the name of its results file
# New results are written as resultados_red/<case>.csv and can be compared with results/<case>.csv.gz.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
[ -d RTS-GMLC/RTS_Data ] && [ -f PDSA-RTS-GMLC/3-DynData/RTS.par ] || { echo "run setup.sh first"; exit 1; }

# results file -> command line of uc_mof1_red.py
declare -A CASES=(
  [base_dyn_mp]="base --dyn --suceso=mayor"
  [muon8_dyn_mp]="muon8 --dyn --suceso=mayor"
  [snsp75_dyn_mp]="snsp75 --dyn --suceso=mayor"
  [ekin13781_dyn_mp]="ekin13781 --dyn --suceso=mayor"
  [rocof1_dyn_mp]="rocof1 --dyn --suceso=mayor"
  [ekin25410_dyn_mp]="ekin25410 --dyn --suceso=mayor"
  [rocof1_dynsabot_mp]="rocof1 --dynsabot --suceso=mayor"
  [base_dyn_mp_nuclibre]="base --dyn --suceso=mayor --nuclibre"
  [base_dyn_mp_sinnuclear]="base --dyn --suceso=mayor --sinnuclear"
  [rocof1_dyn_mp_sinnuclear]="rocof1 --dyn --suceso=mayor --sinnuclear"
  [ekin25410_dyn_mp_sinnuclear]="ekin25410 --dyn --suceso=mayor --sinnuclear"
  [ekin11661_dyn_mp_sinnuclear]="ekin11661 --dyn --suceso=mayor --sinnuclear"
  [muon8_dyn_mp_sinnuclear]="muon8 --dyn --suceso=mayor --sinnuclear"
  [snsp75_dyn_mp_sinnuclear]="snsp75 --dyn --suceso=mayor --sinnuclear"
  [base_dyn_gap0.1_mp]="base --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [rocof1_dyn_gap0.1_mp]="rocof1 --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [ekin25410_dyn_gap0.1_mp]="ekin25410 --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [base_dyn_gap0.1_mp_sinnuclear]="base --dyn --gap=0.001 --tlim=300 --suceso=mayor --sinnuclear"
  [rocof1_dyn_gap0.1_mp_sinnuclear]="rocof1 --dyn --gap=0.001 --tlim=300 --suceso=mayor --sinnuclear"
  [ekin25410_dyn_gap0.1_mp_sinnuclear]="ekin25410 --dyn --gap=0.001 --tlim=300 --suceso=mayor --sinnuclear"
  [muon8_dyn_gap0.1_mp]="muon8 --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [sinc2485_dyn_mp]="sinc2485 --dyn --suceso=mayor"
  [sinc2485_dyn_gap0.1_mp]="sinc2485 --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [ekin12584_dyn_mp]="ekin12584 --dyn --suceso=mayor"
  [ekin12584_dyn_gap0.1_mp]="ekin12584 --dyn --gap=0.001 --tlim=300 --suceso=mayor"
  [base_dyn_hmin_mp]="base --dyn --hmin --suceso=mayor"
  [rocof1_dyn_hmin_mp]="rocof1 --dyn --hmin --suceso=mayor"
)

run() {
  local args=${CASES[$1]}
  [ -n "$args" ] || { echo "unknown case: $1"; return 1; }
  mkdir -p logs
  echo "$(date '+%F %T') start $1"
  $PY uc_mof1_red.py $args > "logs/$1.log" 2>&1 && echo "$(date '+%F %T') done  $1" || echo "$(date '+%F %T') FAILED $1 (see logs/$1.log)"
}
if [ $# -gt 0 ]; then for c in "$@"; do run "$c"; done; exit; fi
for c in "${!CASES[@]}"; do echo "$c"; done | sort | xargs -P 2 -I{} bash "$0" {}
