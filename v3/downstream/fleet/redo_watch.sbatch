#!/bin/bash
#SBATCH -p xeon-p8 -c 2 --mem 8G -t 12:00:00 -J v3d_redo
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/redo_%j.out
# Redo watcher (2026-10-27): for every box of redo_targets.txt ("weights BOX STEM" = the frame's weight is now 0.0;
# "ctx BOX STEM" = the frame's queue context was reverted to the R1 VRT and it is being re-predicted), once the change is LIVE
# (weights row present / queue ctx = box_run/aligned_pedr), runs redo_box_v3.sh (DRY=0) until the box is unaffected,
# not started, or set up for the fleet's redo. Busy boxes (claimed, temp dir or lock) are retried every 10 min; 11.5 h max.
V=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}
T=${TARGETS:-$V3D/redo_targets.txt}; DONE=$V3D/logs/redo_20261027.handled; touch $DONE; END=$(( $(date +%s) + 41400 ))
while :; do
  left=0
  while read -r kind b s; do
    [ -z "$b" ] && continue; grep -qx "$b" $DONE && continue
    if [ "$kind" = weights ]; then grep -q "^$s,0.0,pedr_net_harmful_20261027" $V3D/weights/$b.csv || { echo "wait $b: weights row for $s not installed"; left=$((left+1)); continue; }
    else awk -F'\t' -v b=$b -v s=$s '$2==b && $3==s {print $4}' $V/predict/queue_v3.tsv | grep -q "/global_run/boxes/$b/box_run/aligned_pedr/${s}_aligned.vrt" \
         || { echo "wait $b: queue ctx of $s not yet the R1 VRT"; left=$((left+1)); continue; }; fi
    DRY=0 bash $V3D/redo_box_v3.sh $b; rc=$?
    case $rc in 0) echo "$b" >> $DONE;; 10) left=$((left+1));; *) echo "ERROR $b rc=$rc (left for review)"; echo "$b" >> $DONE;; esac
  done < $T
  [ $left -eq 0 ] && { echo "all targets handled $(date)"; break; }
  [ $(date +%s) -gt $END ] && { echo "time limit: $left targets still waiting; resubmit"; break; }
  echo "--- $left waiting, next pass in 10 min ($(date +%T))"; sleep 600
done
