#!/bin/bash
#SBATCH -J pcf80qa -p debug-cpu -N1 -c 48 --mem=170G -t 02:00:00
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/polar_combfix80/logs/%x_%j.out
# Coordinator 10-29: QA record of the pcf80 surface for the polar boxes = downstream/stages/qa_box_tag_v3.sh (copy with DES overridable,
# code/qa_box_tag_v3_des.sh) with DES = box_run/box_dem_<TAG>_deseam_pcf80.tif and QA_DIR = downstream/qa_pcf80. Only boxes whose pcf80
# marker matches the current _deseam.tif md5. 3 boxes at a time, 16 eval workers each. (frame_outlier_v3.py is report-only and reads by tag.)
G=/home/gridsan/groups/planetary_dtm/mars_dem; PC=$G/global_run/v3_2026-10/polar_combfix80; V3=ta_implicit_d1_spline100s1_e098v3
export TAG=$V3 QA_DIR=$G/global_run/v3_2026-10/downstream/qa_pcf80 NP=16; mkdir -p $QA_DIR; cd $PC
LIST=${BOXES:-$(cat $PC/logs/installed_*.txt 2>/dev/null | sort -u)}; n=0
for B in $LIST; do
  BD=$G/global_run/boxes/$B/box_run; S=$BD/box_dem_${V3}_deseam.tif; MK=$BD/box_dem_${V3}_deseam_pcf80.json
  grep -q "\"source_md5\": \"$(md5sum $S | cut -c1-32)\"" $MK 2>/dev/null || { echo "$B pcf80 missing or stale, no QA"; continue; }
  ( DES=$BD/box_dem_${V3}_deseam_pcf80.tif bash code/qa_box_tag_v3_des.sh $B > $PC/logs/qa_$B.txt 2>&1; echo "$(date +%T) $B $(grep -h VERDICT $QA_DIR/$B.txt | cut -c1-230)" ) &
  n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
done; wait; echo "$(date +%T) QA done"
