#!/usr/bin/env bash
# redo_box_v3.sh <BOX> : make a v3 box whose LEVEL inputs changed (frame weights CSV, or a frame's prediction / queue context)
# re-run by the downstream fleet (2026-10-27). Idempotent. DRY=1 (default) only reports.
# Decision per box (input identity = code/v3_input_guard.py check, the level stage's own test):
#   unaffected  guard passes (published level matches the current queue, predictions, weights, solver) -> nothing
#   not-started no levelled mosaic and no .done/.fail -> nothing (the fleet levels it with the new inputs when it claims it)
#   busy        state/<BOX>/.claim, a level/fill lock, or a level/fill/mif/rejoin temp dir changed within 2 h -> run again later
#   REDO        guard fails and the box is idle -> move aside (<file>.stale.<epoch>): the levelled mosaic + its frame list
#               (+ input_identity.json), the fresh_tag descendants of model/filled/prov (fresh_tag_v3.py quarantine: model,
#               filled, prov, FINAL, deseam, report ... + stamps), the surface's companions and markers (deseam_mask.tif,
#               _mif.json, _relax.json, deseam.tif.mif.json, FINAL.tif.merge.json); state .done/.fail -> .done.redo_<epoch>;
#               the QA record -> qa/<BOX>.txt.redo_<epoch>. The fleet then re-claims the box once its predictions are final
#               (readiness gate) and runs level -> fill -> mif -> qa from scratch. No product is deleted.
# Exit 0 = nothing to do / redone; 10 = busy (retry later); 2 = error.
# Env: TAG (default from CAMPAIGN.json), V3D, QUEUE_TSV (default $V/predict/queue_v3.tsv), BOXES_ROOT, DRY (1).
set -uo pipefail
B=${1:?box}; BASE=/home/gridsan/groups/planetary_dtm/mars_dem; V=$BASE/global_run/v3_2026-10
V3D=${V3D:-$V/downstream}; export V3D
TAG=${TAG:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' $V/predict/CAMPAIGN.json)} || exit 2
Q=${QUEUE_TSV:-$V/predict/queue_v3.tsv}; OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$B/box_run; ST=$V3D/state/$B; DRY=${DRY:-1}
P=${V3_PY:-python3}; LEV=$OUT/box_dem_${TAG}_leveled.tif; DES=$OUT/box_dem_${TAG}_deseam.tif; FINAL=$OUT/box_dem_${TAG}_FINAL.tif
say() { echo "[redo $(date '+%F %T')] $B: $*"; }
[ -d "$OUT" ] || { say "no box_run"; exit 2; }
for x in "$ST/.claim" "$OUT/.level_${TAG}.lock" "$OUT/.fill_${TAG}.lock"; do
  [ -e "$x" ] && { say "busy ($(basename $x)), retry later"; exit 10; }
done
# temp dirs: busy only if something in them changed within 2 h (the fill leaves .fill_tmp_<TAG>/fine.guard.json behind for good)
for x in "$OUT/.level_tmp_$TAG" "$OUT/.fill_tmp_$TAG" "$OUT/.mif_tmp_$TAG" "$OUT/.rejoin_tmp_$TAG"; do
  [ -d "$x" ] && [ -n "$(find "$x" -maxdepth 1 -mmin -120 2>/dev/null | head -1)" ] && { say "busy ($(basename $x) active), retry later"; exit 10; }
done
if [ ! -s "$LEV" ] && [ ! -e "$ST/.done" ] && [ ! -e "$ST/.fail" ]; then say "not started: nothing to do"; exit 0; fi
if [ -s "$LEV" ]; then
  G=$($P "$V3D/code/v3_input_guard.py" "$B" --tag "$TAG" --queue "$Q" --out "$OUT" 2>&1 | tail -1)
  if echo "$G" | grep -q '"ok": true'; then say "unaffected (level inputs match): nothing to do"; exit 0; fi
  say "level inputs changed: $(echo "$G" | sed 's/.*"reason": "//; s/"}$//' | cut -c1-160)"
else
  say "no levelled mosaic but state $(ls -A $ST 2>/dev/null | grep '^\.\(done\|fail\)$' | tr '\n' ' ')"
fi
E=$(date +%s)
mvaside() { for f in "$@"; do [ -e "$f" ] || [ -L "$f" ] || continue
  if [ "$DRY" = 1 ]; then say "would move $(basename $f) -> .stale.$E"; else mv -n "$f" "$f.stale.$E" && say "moved $(basename $f) -> .stale.$E"; fi; done; }
mvaside "$LEV" "${LEV%.tif}.frames.txt" "${LEV%.tif}.input_identity.json"
if [ "$DRY" = 1 ]; then
  say "would quarantine (fresh_tag): $($P $V3D/code/fresh_tag_v3.py list $OUT $TAG 2>/dev/null | awk '$1!="ABSENT" {print $2}' | grep -v '^leveled$\|^ctx$\|ckpt\|^frames$' | tr '\n' ' ')"
else
  $P "$V3D/code/fresh_tag_v3.py" quarantine "$OUT" "$TAG" model filled prov 2>&1 | tail -1 | sed "s/^/[redo] $B: /"
fi
mvaside "${DES%.tif}_mask.tif" "${DES%.tif}_mif.json" "${DES%.tif}_relax.json" "$DES.mif.json" "$FINAL.merge.json"
for s in .done .fail; do [ -e "$ST/$s" ] || continue
  if [ "$DRY" = 1 ]; then say "would move state $s -> $s.redo_$E"; else mv -n "$ST/$s" "$ST/$s.redo_$E" && say "state $s -> $s.redo_$E"; fi; done
QR=${QA_DIR:-$V3D/qa}/$B.txt
[ -e "$QR" ] && { if [ "$DRY" = 1 ]; then say "would move qa record -> $B.txt.redo_$E"; else mv -n "$QR" "$QR.redo_$E" && say "qa record -> $B.txt.redo_$E"; fi; }
say "$([ "$DRY" = 1 ] && echo 'DRY RUN, nothing changed' || echo 'REDO set up: the fleet re-claims the box when its predictions are final')"
exit 0
