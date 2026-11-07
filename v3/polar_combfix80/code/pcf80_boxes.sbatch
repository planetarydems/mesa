#!/bin/bash
#SBATCH -J pcf80 -p debug-cpu -N1 -c 48 --mem=170G -t 02:00:00
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/polar_combfix80/logs/%x_%j.out
# User 10-29: "before the mosaic generation, run this filter above 80N and below 80S". Polar comb filter (combfix v2, frozen copy, see
# code/FROZEN.md5), latitude ramp 0 at 80 deg -> 1 at 81 deg, on every FINISHED v3 box (downstream/state/<BOX>/.done) of rows Np80 + N-90,
# run LAST on the box's current _deseam.tif. Idempotent, keyed on the source md5: a box whose pcf80 marker matches the current
# _deseam.tif md5 is skipped; a rewritten box is re-staged. Steps: box_correct.py -> staging $PC/boxes/<BOX>/, verify_pcf80.py, then
# install_pcf80.py writes box_run/box_dem_<TAG>_deseam_pcf80.tif + .json BESIDE _deseam.tif (never replaces it; coordinator 10-29).
G=/home/gridsan/groups/planetary_dtm/mars_dem; PC=$G/global_run/v3_2026-10/polar_combfix80; V3=ta_implicit_d1_spline100s1_e098v3
ST=$G/global_run/v3_2026-10/downstream/state; cd $PC
unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PC; PY=~/miniforge/envs/isis/bin/python3
export CF_TAG=pcf80 NPROC=${NPROC:-44}; PARAMS=$(cat $PC/code/params.json)
LIST=${BOXES:-$(cd $G/global_run/boxes && ls -d E*_Np80 E*_N-90)}; touch $PC/logs/installed_$SLURM_JOB_ID.txt
for B in $LIST; do
  BD=$G/global_run/boxes/$B/box_run; S=$BD/box_dem_${V3}_deseam.tif; MK=$BD/box_dem_${V3}_deseam_pcf80.json
  [ -f $ST/$B/.done ] && [ -f $S ] || { echo "$(date +%T) $B not finished by the fleet yet"; continue; }
  PRE=$(md5sum $S | cut -c1-32)
  if [ -f $MK ] && [ -f $BD/box_dem_${V3}_deseam_pcf80.tif ] && [ "$($PY code/pcf80_state.py $B | awk '{print $2}')" = current ]; then echo "$(date +%T) $B up to date"; continue; fi
  if ! { [ -f $PC/boxes/$B/verify.json ] && grep -q "\"source_md5\": \"$PRE\"" $PC/boxes/$B/verify.json && grep -q '"ok": true' $PC/boxes/$B/verify.json && \
         [ $PC/boxes/$B/verify.json -nt $PC/code/params.json ] && [ $PC/boxes/$B/verify.json -nt $PC/code/combfix.py ] && [ $PC/boxes/$B/verify.json -nt $PC/code/box_correct.py ]; }; then
    echo "$(date +%T) $B filter"; mkdir -p $PC/boxes/$B; rm -f $PC/boxes/$B/verify.json
    CF_SRC=$S $PY code/box_correct.py $PC/boxes $B "$PARAMS" > $PC/logs/box_$B.txt 2>&1 || { echo "$(date +%T) $B FAILED filter"; tail -3 $PC/logs/box_$B.txt; continue; }
    $PY code/verify_pcf80.py $B > $PC/logs/verify_$B.txt 2>&1 || { echo "$(date +%T) $B VERIFY FAILED"; tail -c 600 $PC/logs/verify_$B.txt; continue; }
    echo "$(date +%T) $B verified $(sed -E 's/"(source|output)_md5": "[0-9a-f]*", //g; s/"source_(size|mtime)": [0-9.]*, //g' $PC/logs/verify_$B.txt | cut -c1-240)"
  fi
  $PY code/install_pcf80.py $B $PRE && echo $B >> $PC/logs/installed_$SLURM_JOB_ID.txt || echo "$(date +%T) $B NOT installed"
done; echo "$(date +%T) done; installed this run: $(wc -l < $PC/logs/installed_$SLURM_JOB_ID.txt)"
