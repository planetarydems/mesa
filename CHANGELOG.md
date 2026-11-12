# Changelog

Versions before 1.0.0 are pre-releases. Until 2026-11-12 they were numbered 1.0.0 (published as
MADER 1.0.0, now 0.1.0), 1.1.0 (now 0.2.0) and the v3 chain (tag `mesa-v3-code`, now 0.3.0), and the
version of the paper was 3.0.0 (now 1.0.0). In the same way, version 1 of the global DEM is the first
public release. It was built as v3, after the pre-releases R1 and R2, and keeps that name in the file
names, tags and paths of production.

## 1.0.0 (November 2026, tag `v1.0.0`): the version of the paper

* The model of the paper: the epoch-98 network with the B-spline decoder
  (`mesa_e098.pt`, md5 `997f5f87f3b94c3b304f82fa9e7504b4`, exported from run
  `ta_impl_spline100_s1_fp32_e098`, checkpoint md5
  `96de121dea9e271212a93df39d6b7f45`), the model of version 1 of the global
  DEM. `mesa.model` gains the B-spline upsampler (`SplineUpsample2d`, a fixed
  degree-4 B-spline x2 followed by a learned 3 x 3 convolution,
  `upsample_mode = spline_conv`) and builds the network from a configuration
  (`mesa/configs/mesa_e098.json`, `mesa_e088.json`; `mesa.model.build`,
  `read_config`). `load_model` takes the configuration from its `config`
  argument, from the weight file, from a JSON file beside it or from the
  packaged configuration of a released file name, and refuses weights of a
  decoder other than the transposed convolution without one: the B-spline
  and bilinear decoders have the same parameter shapes, so the weights cannot
  tell them apart. The epoch-88 network loads as before.
* `mesa.train --config` (default `mesa_e098.json`); the configuration is
  stored in every checkpoint.
* `mesa.predict` applies the two rules of the prediction of version 1: PEDR shots over
  all longitudes for a frame that spans 360 deg of longitude or reaches within
  2 deg of a pole (`PedrDatabase(full_lon_near_poles=True)`), and no
  prediction when the shot-distance input averages below 0.05
  (`--min-conf-mean`). New option `--config`.
* Evaluation of the paper. `data/test_images.csv` holds the co-registration of
  each test image with the full stereo DEM of its pair (the transforms of the
  manuscript's scoring; methods `tile_region_rigid` and
  `tile_region_shift` added to `mesa.evaluate_images`), and the whole images
  are scored against the full stereo DEMs. `data/hirise_*.csv` hold the
  windows of pre-release R1 with their version-1 predictions, the outcome of their image
  co-registration in R1 (`registered_r1`, honoured by
  `mesa.evaluate_hirise`) and the test pairs over the full stereo DEMs
  (149 listed, 132 scored at 98 HiRISE DEMs).
* Verification of 1.0.0 (`verification/README.md`, section "Version 1.0.0"):
  the network, 4 training steps and 3 product frames against production, and
  Table 1 (tiles and whole images) and Tables 3 and 4 against the
  manuscript's scoring. New scripts
  `make_test_images_v3.py`, `make_hirise_inputs_v3.py`,
  `run_tiles_chunked.py` and the jobs `verification/jobs/v3_*.sbatch`.
  `check_model.py`, `check_predict.py`, `check_train_steps.py`,
  `check_evaluate.py`, `check_images.py` and `check_hirise.py` take the
  model configuration and the references of version 1; the R1 job scripts pass
  the R1 settings explicitly.
* Documentation: README, `docs/evaluation.md` and `docs/pipeline.md` describe
  the model of version 1 and the commands behind the numbers of the paper.

## 0.3.0 (November 2026, tag `v0.3.0`, first tagged `mesa-v3-code`): the production chain of version 1

* New directory `v3/` with the scripts of the production chain of version 1 (built as v3, tag
  `ta_implicit_d1_spline100s1_e098v3`), copied verbatim from the cluster:
  prediction adapter and queue runner, model metadata (config, checksums), the
  per-box stages (leveling, CTX layer and infill, join, infill rule v2.2, seam
  stage, relaxation, seam pass 2, QA), the fleet worker and sweeps, the per-box
  frame weights, the polar comb correction and the reviewed-correction tools.
  `PROVENANCE_v3.md` lists the source path and md5 of every file;
  `v3/README.md` the order and settings of the stages; the README section
  "Production chain of version 1" the render rules, the error names and the product classes.
* `VERIFY_v3.md`: replay of one production box (and the polar comb correction
  of a second) from the copied scripts on a scratch copy of the inputs, with
  the md5 of every output compared against production
  (`v3/verification/check_box_v3.sbatch`).
* Nothing in `mesa/`, `data/` or `docs/` changed in 0.3.0.

## 0.2.0 (October 2026, not tagged; numbered 1.1.0 until 2026-11-12)

Working tree after the author's decisions of 2026-10-14, committed on branch
`v3-code` (380da6e) and included in 1.0.0.

* Uncertainty layer calibrated on the assembled mosaic (2026-10-16, revision
  Task 21). `mesa.uncertainty` gains `calibrate-mosaic`, `mosaic-lut`, `remap`
  and the validation baseline `mean_baseline`. The stage-1 single-image table
  (`data/uncertainty_lut_single_image.json`) now gives a score; the released
  values are the median and 90th percentile of the error of the lattice-filtered
  mosaic against the 359 test stereo DEMs after the mosaic's own image
  co-registration, by score bin, number of blended images and infill distance
  class (`data/uncertainty_lut_mosaic.json`, transforms in
  `data/uncertainty_mosaic_coreg.csv`). New per-tile files
  `<tile>_median_error_m.tif` and `<tile>_p90_error_m.tif` replace
  `<tile>_expected_error_m.tif`; `<tile>_uncertainty_class.tif` keeps its
  thresholds. The reserved HiRISE validation cohort is
  `data/uncertainty_hirise_cohort.csv`. `verification/check_uncertainty_mosaic.py`
  compares the package with the production run.
* Renamed to MESA (Mars Elevation from Shading and Altimetry). The package is
  `mesa` (`python -m mesa.<module>`), the network class `MESA`, the loss
  `MesaLoss`, the distribution `mesa` 0.2.0 (then numbered 1.1.0), the expected weight file
  `mesa_e088.pt`. Version 0.1.0 was published as "mader" 1.0.0 (repository
  `planetarydems/mader`, release v1.0.0, now v0.1.0, with the asset `mader_e088.pt`);
  `mesa.model.load_model` accepts either name and falls back to the other in
  the same directory, so the published assets still load. The GitHub repository was renamed
  `planetarydems/mesa` on 2026-11-09. Left as they are: the production tag
  `ta_implicit_d1_implrel100s1_e088`, every file name of the products, the
  cluster paths in `verification/jobs/*.sbatch` (`$R/mader`, the on-disk
  weight copies `$O/mader_e088.pt`) and the row names of the reference tables
  parsed by `verification/check_hirise.py`.
* Shot-anchoring footprint 168 m. `anchor.FOOTPRINT_M` is 168 m (Section 2.5
  of the paper; the leveling always used 168 m). As of 2026-10-16,
  `anchor.PRODUCT_FOOTPRINT_M` aliases that constant, and prediction,
  evaluation and the verification scripts all use it. Stored products and
  scores have not been regenerated by this configuration change. The
  historical production settings and the measured effect of the change are
  retained in `verification/README.md`.
* `data/images.csv` now lists the 53,702 placements (45,570 images) that are
  in the released mosaic, the count of the paper. The 20 queued placements
  whose prediction failed (an image part narrower than one 384 px window, or
  no window with 25 % valid image) are in `data/images_not_in_mosaic.csv` with
  their reasons (docs/pipeline.md, step 4).
* New module `mesa.footprints`: the footprint product of a tile from its work
  directory (dominant-image index COG, overlap-depth COG, image table with
  product id, acquisition time, solar geometry and tile shares, footprint
  polygons in a GeoPackage, summary) and the merge of the per-tile
  GeoPackages into one (docs/pipeline.md, step 10). Needs Shapely and Fiona.
* Documentation in American spelling (leveling, leveled); the command-line
  option `--levelled` of `mesa.uncertainty layer` is unchanged.

## 0.1.0 (2026-10-09; published as MADER 1.0.0)

Published as MADER: `github.com/planetarydems/mader` (now `planetarydems/mesa`), tag v1.0.0 (tag
v0.1.0 since 2026-11-12), weights
`mader_e088.pt`, `infill_coarse.pt`, `infill_fine.pt`,
`infill_coarse_pretrained.pt`. Verified bitwise against the production code
and products (`verification/README.md`).
