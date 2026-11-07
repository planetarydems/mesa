# The v3 chain, as run (release tag `ta_implicit_d1_spline100s1_e098v3`)

This directory holds the scripts that produced the v3 box surfaces, copied verbatim from the
cluster (`../PROVENANCE_v3.md` gives the source path and md5 of every file). The scripts keep
their cluster paths and were run on SuperCloud with the environments named below. They are not
a package: they document the chain as it ran and let one box be replayed bit for bit
(`verification/check_box_v3.sbatch`, result in `../VERIFY_v3.md`). The `mesa` package in the
parent directory is the readable form of the same pipeline for epoch 88 (release 1). The v3
stages below replace its prediction model, join, infill rule and seam repair.

## Order of the stages

| stage | script | input | output |
|---|---|---|---|
| 1. prediction | `predict/predict_queue_v3.py` -> `predict/predict_frame_v3.py` | registered CTX image, MOLA 463 m mosaic, PEDR shots, model `ta_impl_spline100_s1_fp32_e098` | per-frame `pred_shot_anchor.tif`, `pred_feather.tif`, `mola_on_grid.tif`, `ctx_on_grid.tif`, `.done` record |
| 2. leveling | `downstream/stages/level_box_tag_v3.sbatch` -> `build_box_final_tag_v3.sh level_only` -> `code/level_frames_box_v3.py` | the box's frames, `downstream/weights/<BOX>.csv`, orbit screen | `box_dem_<TAG>_leveled.tif`, `leveled_<TAG>/*_leveled.tif` and `*_w.tif` |
| 3. CTX layer and infill | `downstream/stages/ctx_fill_{gpu,cpu}_tag_v3.sbatch` -> `ctx_fill_box_tag_v3.sh` -> `build_box_final_tag_v3.sh fill` | leveled tile, Global CTX Mosaic, infill networks (release 1 weights) | `ctx_robbins_18m_<TAG>.tif`, `box_dem_<TAG>_filled.tif` (+ `.model.tif`, `.prov.tif`) |
| 4. join | `downstream/code/seamline_merge_v3.py` (called by stage 3, `--buffer 32 --feather 8 --min-hole-px 30`) | leveled tile, infill model surface | `box_dem_<TAG>_FINAL.tif`, marker `FINAL.tif.merge.json` |
| 5. infill rule, seam stage, relaxation | `downstream/stages/mif_box_tag_v3.sh` -> `code/mola_infill_fix.py` (v2.2), `code/seam_stage_v3.py`, `code/seam_relax_v3.py` | FINAL, provenance, leveled frames and weights, MOLA, PEDR shots | `box_dem_<TAG>_deseam.tif`, `_deseam_mask.tif`, `box_seam_<TAG>_{domid,mask,offset}.tif`, markers `_deseam.tif.mif.json`, `_deseam_{mif,seam,relax}.json` |
| 6. seam pass 2 | `downstream/stages/pass2_box_tag_v3.sh` -> `code/seam_pass2.py` | the stage-5 surface | `box_dem_<TAG>_deseam.tif` (rewritten), `_deseam_pass2.json`, keys `pass2*` in the stage-5 marker |
| 7. QA | `downstream/stages/qa_box_tag_v3.sh` -> `code/eval_box.py`, `code/lattice_box.py`, `code/frame_outlier_v3.py` | the box surface, the release-1 tile, PEDR census | `downstream/qa/<BOX>.txt` (verdict PASS or REVIEW, never a product change) |
| 8. polar comb correction (boxes of rows Np80 and N-90 only) | `polar_combfix80/code/pcf80_boxes.sbatch` -> `box_correct.py`, `verify_pcf80.py`, `install_pcf80.py` | `box_dem_<TAG>_deseam.tif`, MOLA | `box_dem_<TAG>_deseam_pcf80.tif` beside the source plus the marker `_deseam_pcf80.json` |
| 9. reviewed local corrections | `corrections/apply_global.py` (writes a release), `corrections/resolve_global.py` (reads one at render time) | reviewed candidate surfaces, the box surface | per-release `MANIFEST.json`, patch GeoTIFFs, `<BOX>_reviewed.vrt`, pointer `LATEST.json` |

The fleet worker `downstream/fleet/worker_v3d.py` ran stages 2, 3, 5 and 7 per box (`.done` in
`downstream/state/<BOX>/`). Stages 4 to 7 were re-run on the finished boxes by the sweeps in
`downstream/fleet/` each time a stage changed (`rejoin_worker.sbatch` on 2026-10-26,
`remif_sweep.sbatch` on 10-27, `reqa_sweep*.sbatch` on 10-27 and 10-28, `seam_sweep.sbatch`
and `pass2_sweep.sbatch` from 10-31). The markers written by the stages name the md5 of the
scripts and the settings. A sweep re-runs a box whose marker does not name the installed
versions. The versions in this directory are the ones the markers of the finished boxes name.

## Settings of the chain

* Prediction: the production adapter of release 1 (`src/testing/predict_single_ctx_mosaic_frame.py`,
  a file of the main repository that is not part of this release) imported by
  `predict_frame_v3.py`, which (a) builds the network from `config.json` with `upsample_mode`
  and `down_mode` as constructor keys and stops if either setting does not change the
  architecture, (b) queries the PEDR shots over all longitudes when a frame spans 360 deg of
  longitude or reaches within 2 deg of a pole (the release-2 south-pole failure), and
  (c) `predict_queue_v3.py` fails a frame whose confidence input has a mean below 0.05.
  Windows of 384 px with stride 192 and a Hann blend, PEDR confidence `--conf-d0 2000`,
  shot anchoring (`--methods shot_anchor,feather`).
* Leveling: `level_frames_box_v3.py` with feathered demotion (`--demote-ramp 512`), the raster
  border counted as a data edge, taper 512 px, frame weights from `downstream/weights/<BOX>.csv`
  (factors of hazy frames, and 0 for frames placed with a non-translational error), orbit screens
  of the equatorial tiles, and the release-1 leveled tile as the coverage reference.
* Infill: the release-1 coarse and fine networks (`fill_holes_coarse.py`, `fill_holes_fine.py` of
  the main repository), unchanged.
* Join: `seamline_merge_v3.py`: alphas of every eligible gap computed against the original leveled
  tile, combined by maximum and composited once (order independent). Gaps under 30 px are shifted
  by the edge residual.
* Infill rule v2.2 (`mola_infill_fix.py`): infill pixels whose smoothed departure from the MOLA
  mosaic exceeds 30 m poleward of 60 deg and 150 m elsewhere are replaced by edge-matched MOLA with
  a smooth weight (sigma 32 px). Each region writes only its own pixels (the window overwrite of
  v2.1 is fixed).
* Seam stage (`seam_stage_v3.py`): seam lines from the dominant-frame map only, Poisson solve
  in a 32 px band (`--band-px 32 --feather 4 --lam 1e-2 --halo 64`), PEDR datum weight
  `--authority 0.01`, guard on seam offset, cliffs, reach and shots. Hard joints (0 to 1 px
  overlap) are not solved.
* Relaxation (`seam_relax_v3.py --ramp 32`): hard hand-overs between frames get a half-offset
  correction blended with the leveler's weights.
* Pass 2 (`seam_pass2.py`): lines whose census offset is still 20 m or more, band 64 px,
  `--lam 1e-3 --feather 8 --trim 32 --taper 0 --authority 0`, halo 96 px.
* QA: PASS when straight-line cells are not more than the release-1 tile's, step pixels are
  within max(20, 10 %) of the release-1 count, the PEDR-failing cells do not significantly
  outnumber release 1's (one-sided sign test) and the shot MAE is not worse by more than 0.25 m.
* Polar comb correction (`polar_combfix80`): combfix v2 with rule v3 (`code/params.json`:
  ramp 0 at 80 deg to 1 at 81 deg, infill notch 0.5, k gate 0.45 to 0.6, 60 m bound). The output
  is a separate file. `_deseam.tif` is never replaced. A consumer uses the pcf80 file only while
  the marker's `source_md5` equals the md5 of the current `_deseam.tif`.
* Reviewed local corrections: `apply_global.py` publishes an immutable release directory
  (`site_corrections_global_2026-11-03/releases/<batch>/`) with a `MANIFEST.json` that binds
  the baseline surface, each patch and the output VRT by SHA256, and rewrites `LATEST.json` to
  point at it with the manifest's SHA256. A patch GeoTIFF has the corrected elevation in band 1,
  the weight in band 2 (positive = corrected pixel) and the signed height change in band 3.
  The 2026-11-06 override (line 144 onward): a card with automatic numeric holds passes only
  when its review is the user's own call that records the same holds
  (`user_override_numeric_hold`, reviewer `user (...)`), logged as `USER_OVERRIDE_NUMERIC_HOLD`.
  `resolve_global.py BOX BASELINE --pointer COPY_OF_LATEST.json` returns the reviewed VRT of a
  corrected box, or the baseline, after checking every hash. It raises on any mismatch.

## Environments on SuperCloud

* Stage scripts and `fresh_tag_v3.py`: `module load anaconda/2023a-pytorch` (Python 3.9, the
  environment of the fill and of the join).
* `mola_infill_fix.py`, `seam_stage_v3.py`, `seam_relax_v3.py`, `seam_pass2.py`, the QA tools and
  `polar_combfix80`: `~/miniforge/envs/isis/bin/python3` (Python 3.11.6, numpy 1.26.4,
  scipy 1.16.2, rasterio 1.4.4, GDAL 3.7.1), with `OMP_NUM_THREADS=1` and
  `GDAL_NUM_THREADS=1`.
* Prediction: PyTorch 2.0 (the module above) on V100 GPUs and Xeon CPUs.

## Replaying one box

`verification/check_box_v3.sbatch BOX [POLAR_BOX]` copies the box's leveled tile, CTX layer,
infill outputs, joined tile and leveled frames into a scratch root, restores the provenance
stamps, replays the join, stage 5, stage 6 and the QA from this directory, and compares the
md5 of every output raster with production. With a polar box it also replays the comb
correction on that box's surface. Nothing is written outside the scratch root. The record of
the run is `../VERIFY_v3.md`. The comb correction blends its tiles in the order the worker
processes return them (`box_correct.py`, `Pool.imap_unordered`), with float32 sums in the
overlaps. Its output is reproducible pixel by pixel to within 0.5 mm, not byte by byte
(`verification/compare_pcf80.sbatch`).

## Files not in this directory

* The model weights (`checkpoints/best.pt`, 287.6 MB, md5 `96de121dea9e271212a93df39d6b7f45`)
  and the training code copied with the model (`code/ablation/`, md5s in `model/.../MD5SUMS`).
* The release-1 scripts that the stages call by their cluster paths: the prediction adapter,
  `warp_preds_to_eqc_tag.py`, `preflight_tag.py`, `build_box_ctx_from_murray_tag.sh`,
  `fill_holes_coarse.py`, `fill_holes_fine.py`, and `tag_guard.sh` and `fresh_tag.py` (these two
  are copied to `downstream/helpers/` for reading, and the stages source them from the cluster
  path).
* The inputs: frames, MOLA mosaic, PEDR shot tables, orbit screens, PEDR census shot tables,
  the Global CTX Mosaic, the release-1 tiles that the QA compares against, and the review cards
  and candidate surfaces that the corrections were built from.
