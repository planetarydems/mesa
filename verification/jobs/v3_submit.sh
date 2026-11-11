#!/bin/bash
# Submit the v3 verification jobs once (run with nohup on a login node). A job is skipped when a job of the same name
# is queued or running, or when its submission record exists ($O/submitted/<name>). Usage: v3_submit.sh NAME [NAME ...]
# with NAME in gpu cpu tiles images hirise tilesenv recheck; cpu and tiles wait for gpu (they read the exported weights).
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
J=$C/verification/jobs; mkdir -p $O/submitted $O/logs; cd $O/logs
# NAME_rK (e.g. hirise_r2) resubmits the job of NAME once more under its own record (finished parts are reused).
for arg in "$@"; do
  n=${arg%%_r[0-9]*}; rec=$arg
  name=mesa_v3_$arg
  if [ -e $O/submitted/$rec ]; then echo "$rec: already submitted ($(cat $O/submitted/$rec))"; continue; fi
  q=$(squeue -u $USER -h -n $name -o %i) || { echo "$rec: squeue failed, not submitted"; continue; }
  if [ -n "$q" ]; then echo "$rec: $name already queued"; continue; fi
  dep=""
  if [ $n = cpu ] || [ $n = tiles ]; then
    g=$(cut -d' ' -f1 $O/submitted/gpu 2>/dev/null)
    [ -n "$g" ] && [ ! -s $WT/mesa_e098.pt ] && dep="--dependency=afterok:$g"
  fi
  id=$(sbatch --parsable -J $name $dep $J/v3_$n.sbatch) && echo "$id $(date +%F_%T) $dep" > $O/submitted/$rec && echo "$rec: job $id $dep"
done
