# Polar comb filter for release v3 (poleward of 80 deg)

User 2026-10-29: "okay, I think this looks good enough for now, before the mosaic generation, run this filter above 80N and below 80S.
then after the mosaic, ctx, and uncertainty are all complete, run a test of correcting the MOLA input and repredict (use filter and try
correct with shots)."

- Filter = combfix v2 (polar-mola-correction session, `revision_work/polar_mola_correction_2026-10-29/code/`), frozen here
  (`code/FROZEN.md5`); only change: latitude ramp 0 at 80 deg -> 1 at 81 deg (sandbox runs used 74 -> 76). The fit does not use the ramp,
  so poleward of 81 deg the correction is identical to the tested one; equatorward of 80 deg nothing changes (verified per box, exactly 0).
- Basis: `revision_work/pedr_failure_census_2026-10-23/v3_check/compare_1029/README.md` (combfix beats mog everywhere; 16 polar
  HiRISE DTMs comb-band misfit 2.51 -> 2.13 m, 12 closer, none farther).
- SuperCloud: `$G/global_run/v3_2026-10/polar_combfix80/` (code/, boxes/<BOX>/ staging outputs + verify.json, logs/).
- Stage (`code/pcf80_boxes.sbatch`, debug-cpu, 44 workers): rows Np80 + N-90 (72 boxes), only boxes with the fleet's
  `downstream/state/<BOX>/.done`, run LAST on the current `_deseam.tif`; idempotent and keyed on the source md5 (a rewritten box is re-staged).
  box_correct.py -> staging `$PC/boxes/<BOX>/`, verify_pcf80.py (same grid + mask, exactly 0 change below 80 deg, |C| <= 80 m), then
- Install (coordinator 10-29: never swap `_deseam.tif`, the fleet owns it): `code/install_pcf80.py` writes
  `box_run/box_dem_<TAG>_deseam_pcf80.tif` BESIDE `_deseam.tif` plus the marker `box_dem_<TAG>_deseam_pcf80.json` (source md5 / mtime /
  size, output md5, code md5s, ramp, verify result). Render, CTX/uncertainty and website use the pcf80 file only while the marker's
  source_md5 equals the current `_deseam.tif` md5 (v3 render spec).
- QA (`code/qa_pcf80.sbatch`): `downstream/stages/qa_box_tag_v3.sh` (copy `code/qa_box_tag_v3_des.sh`, only change: DES overridable) on the
  pcf80 surface, records in `downstream/qa_pcf80/`.
- Sweep (coordinator 10-29: no fleet hook): `code/pcf80_sweep.sbatch` re-queues itself every ~2 h on debug-cpu (download's per-user
  limits, 1 job / 8 CPU / 32 GB, are too small), stages + QAs every polar box with `.done` and no current pcf80 (`code/pcf80_state.py`),
  ends when all 72 are current or `$PC/STOP` exists; one line per run with landings in `logs/landed.log`; next job id in `logs/next_job.txt`.
  First run 6084633 (nothing pending), next 6084636 (16:31). "Current" = marker source md5 matches AND marker params / code md5s equal
  `code/params.json` / `code/*.py`, so a method change re-stages every box.
- 10-29 15:05 infill notch APPLIED by the coordinator after the user's "sure redo it ." (`pcf80_infill_notch.diff`, params.json adds
  `"NOTCH_FRAC": 0.5`; code now combfix.py 6a5dd490 / box_correct.py cece2493; backup in backup_20261029_prenotch/, INSTALL_LOG.txt). The
  notch removes half of what is left in the oriented band on infill pixels (feathered prov == 1 mask from `_filled.tif.prov.tif`, gated by
  MOLA band amplitude), times the same ramp (still 0 below 80 deg). Sweep run 6084636 re-stages all finished boxes; pre-notch corrections
  saved in prenotch/<BOX>/; before/after scoring `code/notch_eval.sbatch` (job 6084985) -> notch_eval/summary.md.
- Jobs: 6083989 (cancelled before start, swap design), 6084022 stage + 6084023 QA (11 finished boxes, 10-29).
- Part 2 (after mosaic + CTX mosaic + uncertainty): cleaned-MOLA re-prediction test (filter-cleaned MOLA input; shot-corrected MOLA input).
