#!/bin/bash
#SBATCH -p xeon-p8 -N 1 -c 48 --mem 180G -t 08:00:00 -J v3d_remif
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/remif_%j.out
# Re-run of the mif stage (infill rule v2.1 + seam relaxation 32 px) and the QA for every box the v3 fleet finished
# (state/<B>/.done) whose v3 surface marker (<deseam>.mif.json) does not name the installed scripts' md5s (2026-10-27).
# The stage rebuilds the surface out of place and swaps it in once (stamped). Resumable; PAR boxes at a time (default 3).
V=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}
export TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' $V/predict/CAMPAIGN.json) || exit 2
SIG="mif $(md5sum $V3D/code/mola_infill_fix.py | cut -c1-32) relax $(md5sum $V3D/code/seam_relax_v3.py | cut -c1-32) ramp ${RELAX_RAMP:-32}"
B=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/boxes
for d in $V3D/state/*/; do b=$(basename $d); [ -e $d/.done ] || continue
  grep -qF "\"$SIG\"" $B/$b/box_run/box_dem_${TAG}_deseam.tif.mif.json 2>/dev/null || echo $b; done > $V3D/logs/remif_$SLURM_JOB_ID.boxes
echo "re-mif $(wc -l < $V3D/logs/remif_$SLURM_JOB_ID.boxes) boxes, tag $TAG, $SIG, $(date)"
xargs -P ${PAR:-3} -I{} sh -c 't0=$(date +%s); RELAX_NP=14 bash $V3D/stages/mif_box_tag_v3.sh {} > $V3D/logs/remif_{}.log 2>&1; r=$?; t1=$(date +%s);
  NP=12 bash $V3D/stages/qa_box_tag_v3.sh {} > /dev/null 2>&1; q=$?; echo "{} mif rc=$r $((t1-t0))s qa rc=$q $(( $(date +%s) - t1 ))s $(grep -o "VERDICT [A-Z]*" $V3D/qa/{}.txt)"' \
  < $V3D/logs/remif_$SLURM_JOB_ID.boxes
echo "done $(date)"
