#!/bin/bash
#SBATCH -p xeon-p8 -N 1 -c 48 --mem 180G -t 12:00:00 -J pass2_sweep --exclude=d-17-12-4,d-5-14-3,d-14-2-1,d-18-1-1,d-6-8-2
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/pass2_sweep_%j.out
# DRAFT (2026-10-30, seam-remaining; not installed). Additive second seam pass + QA on every finished box (state/<B>/.done) whose first-pass marker (<deseam>.mif.json, key pass1) has no "pass2" with the current
# signature run on the current pass1, or whose QA record is older than its surface (r1-seam-install's marker scheme: pass2 is added to the mif marker, sig/pass1 untouched). From seam_sweep.sbatch: shards by cksum(box) % NSHARD, PAR lanes, re-lists per pass, no v3d_ prefix so fleet workers yield. One final pass after the fleet ends.
V=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10; export V3D=${V3D:-$V/downstream}; S=${SHARD:-0}; N=${NSHARD:-1}; PAR=${PAR:-3}
export TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' $V/predict/CAMPAIGN.json) || exit 2
B=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/boxes; echo "host $(hostname) shard $S/$N tag $TAG, $(date)"
P2=$V3D/code/seam_pass2.py; SIG="pass2 $(md5sum $P2 | cut -c1-32) band ${P2_BAND:-64} feather ${P2_FEATHER:-8} lam ${P2_LAM:-1e-3} trim ${P2_TRIM:-32} taper ${P2_TAPER:-0} auth ${P2_AUTH:-0} minoff ${P2_MINOFF:-20}"
for pass in 1 2 3 4 5 6; do
  L=$V3D/logs/pass2_sweep_${SLURM_JOB_ID}_p$pass.boxes; echo "pass $pass $SIG"
  for d in $V3D/state/*/; do b=$(basename $d); [ -e $d/.done ] || continue; [ $(( $(echo $b | cksum | cut -d' ' -f1) % N )) = $S ] || continue; R=$B/$b/box_run; D=$R/box_dem_${TAG}_deseam.tif
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); p1=d.get("pass1", d.get("sig")); sys.exit(2 if p1 is None else (0 if d.get("pass2")==sys.argv[2] and d.get("pass2_on")==p1 else 1))' $D.mif.json "$SIG" 2>/dev/null; rc=$?
    [ $rc = 2 ] && continue   # no first-pass marker: not ours to touch
    [ $rc = 0 ] && [ ! $D -nt $V3D/qa/$b.txt ] || echo $b; done > $L
  echo "pass $pass: $(wc -l < $L) boxes $(date)"; [ -s $L ] || break
  xargs -P $PAR -I{} sh -c 't0=$(date +%s); P2_NP=${RNP:-14} timeout -k 120 5400 bash $V3D/stages/pass2_box_tag_v3.sh {} > $V3D/logs/pass2_sweep_{}.log 2>&1; r=$?; t1=$(date +%s);
    NP=10 bash $V3D/stages/qa_box_tag_v3.sh {} > /dev/null 2>&1; q=$?; echo "{} pass2 rc=$r $((t1-t0))s qa rc=$q $(( $(date +%s) - t1 ))s $(grep -o "VERDICT [A-Z]*" $V3D/qa/{}.txt)"' < $L
done
echo "done $(date)"
