#!/bin/bash
# v3 downstream fleet, CPU node (2026-10-24): one worker_v3d.py (level -> CPU fill -> mif -> qa), one box at a time, claims from
# $V3D/boxes.tsv (queue_v3.tsv order; a box is claimed only once all its predictions are final). Job prefix v3d_.
# Resubmits itself on 10 (deadline), 11 (FS errors), 12 (yield); on 75 / a crash / a run shorter than MINRUN it resubmits
# excluding this host, up to MAXCRASH times; stops on 0, 20 (drain), 21 (space), 30 (failed boxes) and when STOP / DRAIN_<partition>
# exists in $V3D. Env: V3D (default global_run/v3_2026-10/downstream), V3D_NO_YIELD=1 (do not yield to pending v3p_ jobs),
# LIST, STATE, STAGES, MAX_BOXES, NO_RESUBMIT.
#SBATCH -p xeon-p8
#SBATCH --job-name=v3d_p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=186G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=d-5-14-3
#SBATCH --output=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/v3d_p8_%j.out
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
F=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}; export V3D=$F
EXCLUDE=${EXCLUDE:-d-5-14-3}
CRASHES=${CRASHES:-0}; MAXCRASH=${MAXCRASH:-3}; MINRUN=${MINRUN:-600}
cd /home/gridsan/groups/planetary_dtm/mars_dem || exit 1
START=$(date +%s)
if python3 -c "import numpy, scipy, rasterio"; then
  DEADLINE=$(python3 -c "import time; print(time.time() + 23.3*3600)")
  python3 "$F/code/worker_v3d.py" --fill cpu --jobs 48 --deadline "$DEADLINE" ${MAX_BOXES:+--max-boxes $MAX_BOXES} \
    ${LIST:+--list $LIST} ${STAGES:+--stages $STAGES} ${STATE:+--state $STATE}
  rc=$?
else
  echo "import check FAILED on $(hostname)"; rc=75
fi
RUNTIME=$(( $(date +%s) - START ))
echo "worker exit=$rc runtime=${RUNTIME}s host=$(hostname) $(date)"
if [ -e "$F/STOP" ] || [ -e "$F/DRAIN_xeon-p8" ] || [ -n "${NO_RESUBMIT:-}" ]; then echo "drain / NO_RESUBMIT: not resubmitting"; exit $rc; fi
resubmit_same() { export EXCLUDE CRASHES=0; sbatch --exclude="$EXCLUDE" "$F/worker_v3d_p8.sbatch"; }
resubmit_excl() {
  if [ "$CRASHES" -lt "$MAXCRASH" ]; then export EXCLUDE="$EXCLUDE,$(hostname -s)" CRASHES=$((CRASHES+1)); sbatch --exclude="$EXCLUDE" "$F/worker_v3d_p8.sbatch"
  else echo "crash budget spent ($CRASHES): not resubmitting"; fi
}
if [ "$RUNTIME" -lt "$MINRUN" ] && [ "$rc" != 0 ] && [ "$rc" != 20 ] && [ "$rc" != 30 ]; then resubmit_excl
else case $rc in 0|20|21|30) echo "finished (rc=$rc)";; 10|11|12) resubmit_same;; *) resubmit_excl;; esac; fi
exit $rc
