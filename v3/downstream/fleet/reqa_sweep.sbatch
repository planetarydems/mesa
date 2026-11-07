#!/bin/bash
#SBATCH -p debug-cpu -N 1 -c 48 --mem 170G -t 02:00:00 -J v3d_reqa
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/reqa_%j.out
# Re-QA sweep (2026-10-27): runs the installed stages/qa_box_tag_v3.sh (churn-tolerant PEDR rule + outlier report) on every box
# the v3 fleet finished (state/<B>/.done) whose QA record does not yet carry the new rule ("[rule 2026-10-27]"). QA only: it writes
# $V3D/qa/<B>.txt and $V3D/qa/outliers/, never a product. Resumable (re-submit to continue); 4 boxes at a time.
V=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}
export TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' $V/predict/CAMPAIGN.json) || exit 2
for d in $V3D/state/*/; do b=$(basename $d); [ -e $d/.done ] || continue
  grep -q "rule 2026-10-27" $V3D/qa/$b.txt 2>/dev/null || echo $b; done > $V3D/logs/reqa_$SLURM_JOB_ID.boxes
echo "re-QA $(wc -l < $V3D/logs/reqa_$SLURM_JOB_ID.boxes) boxes, tag $TAG, $(date)"
xargs -P ${PAR:-4} -I{} sh -c 'NP=12 bash $V3D/stages/qa_box_tag_v3.sh {} > /dev/null 2>&1; echo "{} rc=$? $(grep -o "VERDICT [A-Z]*" $V3D/qa/{}.txt) $(grep -o "^OUTLIERS [^;]*" $V3D/qa/{}.txt | cut -c1-90)"' \
  < $V3D/logs/reqa_$SLURM_JOB_ID.boxes
echo "done $(date)"
