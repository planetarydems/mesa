#!/bin/bash
#SBATCH -p xeon-p8 -N 1 -c 48 --mem 180G -t 12:00:00 -J seam_sweep --exclude=d-17-12-4,d-5-14-3,d-14-2-1,d-18-1-1,d-6-8-2
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/seam_sweep_%j.out
# Re-run of the combined mif stage (infill rule v2.2 -> seam stage -> relaxation) and the QA for every box the v3 fleet finished (state/<B>/.done) whose surface marker (<deseam>.mif.json) does not name the installed
# scripts' md5s and the band settings, or whose QA record is older than its surface (a lane that dies between the stage and the QA; the stage then skips and the QA re-runs). 2026-10-30, from mif22_sweep.sbatch, infill-overwrite-fix session.
# The stage rebuilds the surface out of place and swaps it in once (stamped). Resumable. A stage stuck past 2 h (the worker's own limit; a killed Pool child hangs imap_unordered) is killed and reported as rc=124.
# Job name without the v3d_ prefix on purpose: the fleet workers yield a node to pending jobs of other names, not to v3d_ ones. FIRST=<file>: boxes to take first, one per line.
# Lanes: SHARD=i NSHARD=n submits one of n lanes; a box belongs to the lane cksum(box) % n, so lanes never touch the same box. Each lane re-lists when done, so it also picks up boxes that finished with the old stage while it
# ran, and stops when its list is empty. Run one final pass after the fleet ends: the worker treats .done as final, only this sweep reads the marker.
V=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}; S=${SHARD:-0}; N=${NSHARD:-1}; PAR=${PAR:-3}
export TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' $V/predict/CAMPAIGN.json) || exit 2
B=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/boxes; echo "host $(hostname) shard $S/$N tag $TAG, $(date)"
for pass in 1 2 3 4 5 6; do
  SIG="mif $(md5sum $V3D/code/mola_infill_fix.py | cut -c1-32) relax $(md5sum $V3D/code/seam_relax_v3.py | cut -c1-32) ramp ${RELAX_RAMP:-32} seam $(md5sum $V3D/code/seam_stage_v3.py | cut -c1-32) band ${SEAM_BAND:-32} feather ${SEAM_FEATHER:-4} lam ${SEAM_LAM:-1e-2}"
  L=$V3D/logs/seam_sweep_${SLURM_JOB_ID}_p$pass.boxes; echo "pass $pass $SIG"
  for d in $V3D/state/*/; do b=$(basename $d); [ -e $d/.done ] || continue; [ $(( $(echo $b | cksum | cut -d' ' -f1) % N )) = $S ] || continue; R=$B/$b/box_run
    grep -qF "\"$SIG\"" $R/box_dem_${TAG}_deseam.tif.mif.json 2>/dev/null && [ ! $R/box_dem_${TAG}_deseam.tif -nt $V3D/qa/$b.txt ] || echo $b; done > $L.all
  { [ -s ${FIRST:-/dev/null} ] && grep -xFf $L.all $FIRST; cat $L.all; } | awk '!s[$0]++' > $L; rm -f $L.all
  echo "pass $pass: $(wc -l < $L) boxes $(date)"; [ -s $L ] || break
  xargs -P $PAR -I{} sh -c 't0=$(date +%s); RELAX_NP=${RNP:-14} timeout -k 120 7200 bash $V3D/stages/mif_box_tag_v3.sh {} > $V3D/logs/seam_sweep_{}.log 2>&1; r=$?; t1=$(date +%s);
    NP=10 bash $V3D/stages/qa_box_tag_v3.sh {} > /dev/null 2>&1; q=$?; echo "{} mif rc=$r $((t1-t0))s qa rc=$q $(( $(date +%s) - t1 ))s $(grep -o "VERDICT [A-Z]*" $V3D/qa/{}.txt)"' < $L
done
echo "done $(date)"
