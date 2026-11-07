#!/bin/bash
#SBATCH -J pcf80sw -N1 -c 32 --mem=150G -t 02:00:00
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/polar_combfix80/logs/%x_%j.out
# Self-resubmitting pcf80 sweep (coordinator 10-29). Every ~2 h (debug-cpu or download, whichever starts first): every polar box
# (Np80 / N-90) with state/<BOX>/.done and no current pcf80 (missing, or marker source md5 != current _deseam.tif) is staged + verified +
# written beside _deseam.tif (pcf80_boxes.sbatch logic), then gets its QA record (qa_box_tag_v3_des.sh -> downstream/qa_pcf80/).
# Partition (Slurm here refuses multi-partition jobs): code/pick_part.sh (debug-cpu; download too small: 1 job, 8 CPU, 32 GB per user,
# held by another session). The next run is queued at the START (begin now+2h, afterany this job) so a killed run does not break the chain; it is cancelled when
# all polar boxes are current or $PC/STOP exists. One line per run in logs/landed.log.
G=/home/gridsan/groups/planetary_dtm/mars_dem; PC=$G/global_run/v3_2026-10/polar_combfix80; V3=ta_implicit_d1_spline100s1_e098v3; cd $PC
unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PC; PY=~/miniforge/envs/isis/bin/python3
export NPROC=${NPROC:-28} TAG=$V3 QA_DIR=$G/global_run/v3_2026-10/downstream/qa_pcf80 NP=8; mkdir -p $QA_DIR
T0=$(date +%s); BUDGET=${BUDGET:-5700}; J=$SLURM_JOB_ID
[ -f $PC/STOP ] && { echo "$(date '+%F %T') job $J: STOP file, sweep ended" | tee -a $PC/logs/landed.log; exit 0; }
NEXT=$(sbatch --parsable -p $(bash $PC/code/pick_part.sh) --begin=now+2hours --dependency=afterany:$J $PC/code/pcf80_sweep.sbatch); echo "next run $NEXT"; echo $NEXT > $PC/logs/next_job.txt
$PY code/pcf80_state.py > $PC/logs/state_$J.txt; PENDING=$(awk '$2=="pending"{print $1}' $PC/logs/state_$J.txt)
echo "$(date +%T) polar boxes: $(wc -l < $PC/logs/state_$J.txt); current $(grep -c ' current' $PC/logs/state_$J.txt); pending $(echo $PENDING | wc -w); not done $(grep -c notdone $PC/logs/state_$J.txt)"
touch $PC/logs/installed_$J.txt; declare -a QP=()
for B in $PENDING; do
  [ -f $PC/STOP ] && { echo "STOP file"; break; }
  [ $(( $(date +%s) - T0 )) -gt $BUDGET ] && { echo "$(date +%T) time budget reached, rest next run"; break; }
  n0=$(wc -l < $PC/logs/installed_$J.txt); BOXES=$B SLURM_JOB_ID=$J bash code/pcf80_boxes.sbatch | grep -v "^.*done; installed"
  if [ $(wc -l < $PC/logs/installed_$J.txt) -gt $n0 ]; then
    while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 5; done
    ( DES=$G/global_run/boxes/$B/box_run/box_dem_${V3}_deseam_pcf80.tif bash code/qa_box_tag_v3_des.sh $B > $PC/logs/qa_$B.txt 2>&1 ) &
  fi
done; wait
L=$(cat $PC/logs/installed_$J.txt | tr '\n' ' '); NL=$(echo $L | wc -w); pass=0; rev=0; revl=""
for B in $L; do v=$(grep -ho "VERDICT [A-Z]*" $QA_DIR/$B.txt 2>/dev/null | cut -c9-); [ "$v" = PASS ] && pass=$((pass+1)) || { rev=$((rev+1)); revl="$revl $B:${v:-none}"; }; done
$PY code/pcf80_state.py > $PC/logs/state_${J}_end.txt; NT=$(wc -l < $PC/logs/state_${J}_end.txt); NC=$(grep -c ' current' $PC/logs/state_${J}_end.txt)
FAILED=$(grep -E "FAILED|NOT installed" $PC/logs/pcf80sw_$J.out | awk '{print $2}' | sort -u | tr '\n' ' ')
[ $NL -gt 0 ] || [ -n "$FAILED" ] && echo "$(date '+%F %T') job $J: landed $NL [$L] QA PASS $pass REVIEW/other $rev [$revl ]; failed [$FAILED]; current $NC/$NT" >> $PC/logs/landed.log
if [ -z "$NEXT" ] && ! { [ $NC -eq $NT ] && [ $NT -ge 72 ]; } && [ ! -f $PC/STOP ]; then    # the start-of-run resubmission failed: retry once
  NEXT=$(sbatch --parsable -p $(bash $PC/code/pick_part.sh) --begin=now+2hours $PC/code/pcf80_sweep.sbatch); echo $NEXT > $PC/logs/next_job.txt
  [ -n "$NEXT" ] || echo "$(date '+%F %T') job $J: CHAIN BROKEN (could not queue the next run)" >> $PC/logs/landed.log; fi
if [ $NC -eq $NT ] && [ $NT -ge 72 ]; then scancel $NEXT; echo "$(date '+%F %T') job $J: all $NT polar boxes current, sweep ended" | tee -a $PC/logs/landed.log; fi
[ -f $PC/STOP ] && { scancel $NEXT; echo "$(date '+%F %T') job $J: STOP file, sweep ended" | tee -a $PC/logs/landed.log; }
echo "$(date +%T) run done: landed $NL, current $NC/$NT"
