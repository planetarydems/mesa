#!/bin/bash
#SBATCH -J pcf80lim -p debug-cpu -N1 -c 48 --mem=170G -t 02:00:00
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/polar_combfix80/logs/%x_%j.out
# SCRATCH only ($PC/limit_test): (1) repair the notch before/after scoring of the 9 boxes (job 6084985 died: partial params json),
# (2) pre-notch corrections of the 3 refused boxes that have none, (3) limit_test.py on the 6 refused + 5 normal boxes.
G=/home/gridsan/groups/planetary_dtm/mars_dem; PC=$G/global_run/v3_2026-10/polar_combfix80; W=$G/global_run/v3_2026-10/polar_mola_correction; V3=ta_implicit_d1_spline100s1_e098v3
E=$PC/notch_eval; O=$PC/limit_test; mkdir -p $O/c0 $O/logs; cd $PC
unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WIN_LAT_MIN=80; PY=~/miniforge/envs/isis/bin/python3
echo "$(date +%T) repair notch_eval"
for B in $(ls -d $E/E* | xargs -n1 basename); do cp prenotch/$B/pcf80.json $E/$B/pcf80.json; cp boxes/$B/pcf80.json $E/$B/cfn80.json; done; n=0
for B in $(ls -d $E/E* | xargs -n1 basename); do
  for T in pcf80 cfn80; do ( OMP_NUM_THREADS=2 $PY $W/code/box_eval.py $E $B $T > $E/$B/eval_$T.txt 2>&1 || echo "$B $T eval FAILED" ) & n=$((n+1)); [ $((n % 12)) -eq 0 ] && wait; done
done; wait; $PY code/summ_notch_pmc.py $E > $E/summary.md 2>&1; echo "$(date +%T) notch_eval done"
for B in E-070_N-90 E-070_Np80 E-100_N-90; do
  CF_TAG=pn NPROC=44 CF_SRC=$G/global_run/boxes/$B/box_run/box_dem_${V3}_deseam.tif $PY code/box_correct.py $O/c0 $B '{"LAT0": 80.0, "LAT1": 81.0}' > $O/logs/c0_$B.txt 2>&1 || echo "$B c0 FAILED"; done; echo "$(date +%T) c0 done"
n=0; for B in E-020_Np80 E-050_Np80 E-060_N-90 E-070_N-90 E-070_Np80 E-100_N-90 E-010_Np80 E-030_Np80 E-040_Np80 E-010_N-90 E-020_N-90; do
  ( OMP_NUM_THREADS=2 $PY code/limit_test.py $O $B > $O/logs/lim_$B.txt 2>&1 || echo "$B limit FAILED" ) & n=$((n+1)); [ $((n % 4)) -eq 0 ] && wait
done; wait; echo "$(date +%T) limit_test done"
