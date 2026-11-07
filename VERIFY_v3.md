# Verification of the v3 chain: one production box replayed from `v3/`

Purpose. A fresh checkout of `v3/` must reproduce a finished production box bit for bit. The
check replays the per-box stages after the infill (join, infill rule v2.2, seam stage,
relaxation, seam pass 2, QA) from the copied scripts on a scratch copy of the box's inputs and
compares the md5 of every output raster with the production file. A second, polar box checks
the polar comb correction the same way. Production paths are only read.

## Box choice

* `Ep040_N-50` (equatorial band, 97 frames, fleet job 6110811 on 2026-10-31). Its surface was
  last written by the pass-2 sweep (job 6119710, 2026-10-31 23:50 EST) and has not been rebuilt
  by the seam campaign since. The production file is therefore the output of exactly the stages
  in `v3/`: join `seamline_merge_v3.py` (md5 4bc43e49, marker `FINAL.tif.merge.json`), infill rule
  012ef280, seam stage cf033b08, relaxation 8b8f9501 with ramp 32 (marker
  `_deseam.tif.mif.json`, job 6119689), pass 2 5a477b15 (one line solved, 5.4 km accepted).
  QA verdict PASS.
* `E-130_Np80` (polar, 18 frames) for the comb correction: its `_deseam_pcf80.json` marker
  names the current code (`pcf80_state.py`: current) and the installed output md5.

## Stage-in

The level, CTX, infill and join products of the box were copied (`cp -p`, times kept) into
`$G/release_check/v3/boxes/Ep040_N-50/box_run/` together with `leveled_<TAG>/` (196 files,
4.2 GB). The provenance stamps of the chain name the inode of every input. The copies were
therefore re-stamped there with `fresh_tag_v3.py stamp` before the stages ran. The stage scripts were
pointed at the copy with `BOXES_ROOT`, at the released code with `V3D`, and at a scratch QA
directory with `QA_DIR`. The worker counts of the production sweeps were kept (14).

## Command

```bash
# on SuperCloud; $G = /home/gridsan/groups/planetary_dtm/mars_dem
tar czf - v3 | ssh mit 'tar xzf - --strip-components=1 -C $G/release_check/v3/code'
# every file of the copy against its cluster source
awk -F"\t" 'NR>1{print $3"  "$2}' $G/release_check/v3/code/SOURCES.tsv | md5sum -c --quiet -
cd $G/release_check/v3/code && sbatch verification/check_box_v3.sbatch Ep040_N-50 E-130_Np80
```

## Result (job 6185182, xeon-p8 node d-6-6-2, 2026-11-06 19:50 to 20:21 EST, 31 min)

Box `Ep040_N-50`, every output identical to production
(`$G/release_check/v3/results/Ep040_N-50/RESULT.txt`):

| step | time | compared | result |
|---|---|---|---|
| stage-in and re-stamp | 2 min | copies of the inputs (14 GB) | stamped fresh |
| join (`seamline_merge_v3.py`) | 696 s | `box_dem_<TAG>_FINAL.tif` | md5 `788c9836aad1031d7b92e8e4919bb5ca`, bitwise |
| infill rule v2.2, seam stage, relaxation (`mif_box_tag_v3.sh`) | 598 s | `box_dem_<TAG>_deseam.tif` after pass 2, `box_seam_<TAG>_{domid,mask,offset}.tif` | md5 `cab3bdbcc10ac79deddd29682d91d004`, `27501af3...`, `998347c5...`, `fa71ee0a...`, all bitwise |
| seam pass 2 (`pass2_box_tag_v3.sh`) | 163 s | the same surface (one line, 5.4 km) | included above |
| markers | | `FINAL.tif.merge.json`, `_deseam.tif.mif.json`, `_deseam_{mif,seam,relax,pass2}.json` | same values (paths, dates, job ids and timings removed) |
| QA (`qa_box_tag_v3.sh`) | 144 s | verdict line | identical: PASS, lines 0 vs 1, step px 0 vs 951, PEDR cells 43 vs 49, shot MAE 3.41 vs 3.42 m |

Box `E-130_Np80`, polar comb correction: the replayed `_deseam_pcf80.tif` differs in md5 from
the installed file (`c4e6ce66...` vs `3e8a815a...`). See the section below.

## Polar comb correction: run-to-run rounding

`box_correct.py` blends its 1024 px tiles (192 px cosine overlap) into the correction raster in
the order in which the worker processes return them (`Pool.imap_unordered`, line 96). The sums
in the overlap bands are float32 and depend on that order. Two runs of the same code therefore
differ by float32 rounding in the overlaps, and the output md5 is not reproducible. The
equivalence of the comb correction is checked pixel by pixel instead
(`v3/verification/compare_pcf80.sbatch`: two replays with 44 and 28 workers against the
installed file, headers compared field by field).

Result (job 6185457, debug-cpu, 2026-11-06 20:49 to 20:54 EST,
`$G/release_check/v3/results/E-130_Np80/pcf80_compare.txt`):

| comparison | differing pixels (of 125,148,077) | largest difference |
|---|---|---|
| installed file vs replay with 44 workers | 460 (0.0004 %) | 0.49 mm (one float32 step at these elevations) |
| installed file vs replay with 28 workers | 461 | 0.49 mm |
| replay 44 vs replay 28 workers | 547 | 0.49 mm |
| source `_deseam.tif` vs installed (the correction itself) | 65,261,307 (52 %) | 56.9 m, median 0.08 m |

Driver, data type, grid, CRS, nodata, tags and band descriptions are identical. The replayed
correction is the installed correction to within the rounding of the tile order, and two runs
of the released code differ from each other by the same amount. The checked correction spans
latitudes 80.64 to 86.28 deg in this box. The comb correction is therefore reproduced pixel by
pixel but not byte by byte.

## Summary

* The per-box chain after the infill (join, infill rule v2.2, seam stage, relaxation, seam
  pass 2) reproduces the production surface of `Ep040_N-50` bit for bit from the scripts in
  `v3/`, together with the seam rasters, the stage markers and the QA verdict.
* The polar comb correction reproduces the installed `_deseam_pcf80.tif` of `E-130_Np80` to
  within 0.5 mm on 0.0004 % of the pixels (float32 rounding of the parallel tile order).
* Not replayed here: prediction, leveling and the GPU infill of a box (release 1 verified the
  same leveling and infill code bitwise and within 1.5 mm, `verification/README.md`), and the
  publication of a correction release (`apply_global.py` needs the review cards and the
  publication lock of the campaign).
* Scratch copies were removed after the run. Kept under `$G/release_check/v3/`: `code/` (the
  released tree), `sources_check.txt`, `results/`, `logs/`, `qa/`.
