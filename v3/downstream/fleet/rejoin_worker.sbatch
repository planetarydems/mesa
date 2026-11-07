#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=186G
#SBATCH -t 12:00:00
#SBATCH -J v3d_rejoin
#SBATCH --exclude=d-5-14-3
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/rejoin_%j.out
# Re-join worker (2026-10-26): claims boxes the v3 fleet finished (state/<B>/.done, no .claim) whose FINAL.merge.json does not name
# the installed seamline_merge_v3.py md5 -> stages/rejoin_box_v3.sh (out-of-place join + infill rule, swap) -> stages/qa_box_tag_v3.sh.
# LANES lanes per node (default 2; one equatorial join peaks ~70-90 GB). Claims: rejoin/claim/<B> (mkdir); one attempt per box:
# rejoin/rc/<B> = "<rejoin rc> <qa rc> <secs> <job> <date>" (remove it to retry). CHECK_BOX (default E-010_N-10) runs the before/
# after join check. A lane exits after two passes with nothing to do (10 min apart: catches boxes whose fleet mif began
# before the install). touch rejoin/STOP = no lane starts another box. No lane starts a box with less than 40 min of the job left. Submit with -p debug-cpu -t 2:00:00 for a debug node.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
G=/home/gridsan/groups/planetary_dtm/mars_dem; V=$G/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}
export TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "$V/predict/CAMPAIGN.json") || exit 2
JMD5=$(md5sum "$V3D/code/seamline_merge_v3.py" | cut -c1-32); C=$V3D/rejoin; mkdir -p "$C/claim" "$C/rc"
LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; if (NF==4) s=$1*86400+$2*3600+$3*60+$4; print s}')
END=$(( $(date +%s) + ${LEFT:-7200} ))
echo "rejoin worker $SLURM_JOB_ID $(hostname) tag=$TAG join=$JMD5 lanes=${LANES:-2} until $(date -d @$END +%T)"
lane() {
  local idle=0
  while [ $idle -lt 2 ]; do
    local n=0
    for B in $( { echo "${CHECK_BOX:-E-010_N-10}"; ls "$V3D/state"; } | awk '!s[$0]++'); do
      d=$V3D/state/$B; F=$G/global_run/boxes/$B/box_run/box_dem_${TAG}_FINAL.tif
      [ -e "$d/.done" ] && [ ! -e "$d/.claim" ] && [ ! -e "$d/.fail" ] || continue
      grep -q "\"$JMD5\"" "$F.merge.json" 2>/dev/null && ! [ "$F" -nt "$F.merge.json" ] && continue
      [ -e "$C/rc/$B" ] && continue
      [ $(( END - $(date +%s) )) -gt 2400 ] || { echo "lane $1: < 40 min left, stop"; return; }
      [ -e "$C/STOP" ] && { echo "lane $1: $C/STOP, stop"; return; }
      mkdir "$C/claim/$B" 2>/dev/null || continue
      echo "$SLURM_JOB_ID lane$1 $(hostname) $(date +%FT%T)" > "$C/claim/$B/by"; n=$((n+1)); t0=$(date +%s)
      CHECK=$([ "$B" = "${CHECK_BOX:-E-010_N-10}" ] && echo 1) bash "$V3D/stages/rejoin_box_v3.sh" "$B" > "$V3D/logs/rejoin_${B}.log" 2>&1; rc=$?
      qrc=-
      if [ $rc -eq 0 ]; then NP=16 bash "$V3D/stages/qa_box_tag_v3.sh" "$B" >> "$V3D/logs/rejoin_${B}.log" 2>&1; qrc=$?; fi
      echo "$rc $qrc $(( $(date +%s) - t0 )) $SLURM_JOB_ID $(date +%FT%T)" > "$C/rc/$B"; rm -rf "$C/claim/$B"
      echo "lane $1 $B rejoin rc=$rc qa rc=$qrc $(( $(date +%s) - t0 ))s | $(grep -h 'merge-v3\]\|\[E\|\[Ep\|VERDICT' "$V3D/logs/rejoin_${B}.log" | cut -c1-160 | tr '\n' ' ')"
    done
    if [ $n -eq 0 ]; then idle=$((idle+1)); [ $idle -lt 2 ] && sleep 600; else idle=0; fi
  done
  echo "lane $1: nothing left $(date +%T)"
}
for l in $(seq 1 ${LANES:-2}); do lane $l & sleep 120; done; wait
echo "rejoin worker done $(date)"
