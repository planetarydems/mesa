# Verification

Each script runs this code and the code that produced the paper and the mosaic
on the same inputs, on the cluster where the products were made, and compares
the outputs. "Bitwise" means identical arrays of floating-point numbers.
`jobs/` holds the Slurm scripts with the paths and environments used
(PyTorch 2.0 on V100 GPUs and Xeon CPUs; the registration in the environment of
the original registration, OpenCV 4.7).

The first section records the checks of version 1.0.0, the version of the
paper: the epoch-98 network with the B-spline decoder (`mesa_e098.pt`), the
frames of version 1 of the global DEM and the evaluation of the paper. The
sections after it record the checks of pre-release R1 (epoch 88, version 0.1.0)
and of version 0.2.

## Version 1.0.0 (November 2026): the model and the evaluation of the paper

The model of the paper is run `ta_impl_spline100_s1_fp32_e098` (checkpoint
md5 `96de121dea9e271212a93df39d6b7f45`), exported by `check_model.py` to
`mesa_e098.pt` (md5 `997f5f87f3b94c3b304f82fa9e7504b4`; the weights without
the unused `mola_weight` scalar and the configuration
`mesa/configs/mesa_e098.json`). The references are the run's own training
code (copied with the model), the frames of version 1 of the global DEM
(`global_run/boxes/<tile>/box_run/preds_ta_implicit_d1_spline100s1_e098v3/`)
and the manuscript's evaluation (`experiments/v3_manuscript_results_2026-11-05`,
results of 2026-11-05). Jobs `jobs/v3_*.sbatch` (SuperCloud jobs
6202163-6202166, 6202550 and the diagnostic 6203158 on 2026-11-09;
6213216-6213218 for `hirise_r2`, `tilesenv` and `recheck` on 2026-11-10),
submitted once by `jobs/v3_submit.sh`, paths in `jobs/v3_env.sh`.

| step | script (job) | what was compared | result |
|---|---|---|---|
| network | `check_model.py` (`v3_gpu`) | released network against the network built as the run's `eval_cells.load_model` builds it (config.json, `upsample_mode = spline_conv`, `down_mode = maxpool`): random inputs and 8 test samples, CPU and V100 | bitwise |
| training | `check_train_steps.py`, `MODEL_CONFIG=mesa_e098` (`v3_gpu`) | loss of each of 4 optimiser steps (CPU) and every weight after them, against the run's trainer | bitwise |
| prediction | `check_predict.py`, `FOOTPRINT_M=150` (`v3_gpu`, `v3_cpu`) | 3 product frames, re-predicted on the kind of node production used, 5 threads: P18_007966_1361 (Ep040_N-50, V100), P18_007968_2023 (E-020_Np20, Xeon), J06_046962_0925 (E-060_N-90, 86-88 S, PEDR query over all longitudes; Xeon); the DEM, the DEM without the shot fit, and the image and MOLA grids | bitwise (all 12 rasters; the polar frame, 441 windows of which 415 fitted to shots, took 93 min on 5 threads) |
| Table 1 (tiles) | `run_tiles_chunked.py`, `check_evaluate.py` (`v3_tiles`) | every measure of the 6,396 test samples, CPU, 4 threads per process, against the manuscript's sample table; the Table 1 tile cells, the incidence classes and the consistency between views | same 6,396 samples, the same anchoring and shot count at every sample. In the Python of the manuscript's tile scoring (conda environment `planetary_dtm`: PyTorch 2.5.1, GDAL 3.6.2; `v3_tilesenv`, job 6213217) every sample measure equals the reference at its rounding at all 6,396 samples (labelMAE, alignedMAE, `<1 km` r and amplitude, incidence), the MOLA measures are within 3e-13 and the 33 Table 1 values within 1.1e-14. In the PyTorch 2.0 module environment (`v3_tiles`) 6,391 (labelMAE), 6,171 (alignedMAE), 6,394 (r) and 6,389 (amplitude) of the 6,396 samples match at the reference's rounding and the 33 values are within 3e-5; the production scorer run in that environment gives this code's values (diagnostic job 6203158), so the remaining differences come from the environment |
| Table 1 (whole images) | `make_test_images_v3.py`, `check_images.py` (`v3_images`) | `data/test_images.csv` rebuilt from the manuscript's co-registration records; grids, every measure of the 359 pairs, the Table 1 cells and the Section 3.1 statistics | `data/test_images.csv` equals the table rebuilt from the records (`v3_recheck`, job 6213218). Stereo DEM and MOLA grids identical. Prediction grids: 121 pixels in 13 of the 359 pairs differ by at most two float32 spacings (at most 0.49 mm). The manuscript's grids were warped by `gdal.Warp` under GDAL 3.13.2, this code warps with rasterio's bilinear `reproject` under GDAL 3.9.3. At the 6 pairs with the most such pixels, 62 of the 81 lie in the scored area (8.7 to 20.0 million scored pixels per pair) and the in-place MAE agrees to the 9th decimal. Every measure within 2e-12 (relative above 1); every Table 1 cell and Section 3.1 statistic within 1e-12. The verdict line of `check_images.py` reads `IMAGES CHECK DIFF` only because its grid rule allows one float32 spacing |
| Tables 3 and 4 | `make_hirise_inputs_v3.py`, `check_hirise.py` (`v3_hirise`) | the HiRISE input tables rebuilt from the manuscript's records; every window of the 619 HiRISE DEMs, the per-DEM values, every cell of Tables 3 and 4 and the Section 3.2 numbers | identical: at all 1,172 windows the status, gate, co-registration method, stereo DEM and surfaces are the same and every window measure equals the reference (image shifts within 4e-12 m, ground shifts within 1e-7 m); the 3,273 per-DEM rows equal the reference at its rounding (5e-5); Table 3 (27 cells), Table 4 (119 cells) and the 25 Section 3.2 statements: 0 differ (`hirise_r2`, job 6213216, GDAL 3.6.2). Job 6202550 had scored 1,171 windows; the one window of DTEPC_025901_2460_026455_2460_A01 ran out of memory with 40 workers on the node and was scored alone in `hirise_r2` |

Values of the paper and of this code (medians; m unless stated):

| quantity | paper (manuscript) | this code | difference |
|---|---|---|---|
| tiles: samples / tiles / pairs | 6,396 / 2,878 / 359 | 6,396 / 2,878 / 359 | same samples |
| tiles: labelMAE | 7.176 | 7.176 | 0 |
| tiles: alignedMAE | 10.377 | 10.377 | < 1e-4 |
| tiles: `<1 km` r / amplitude | 0.7726 / 0.7823 | 0.7726 / 0.7823 | < 1e-4 |
| tiles: error / relief (%) | 4.609 | 4.609 | < 1e-4 |
| tiles: MOLA MAE, r, amplitude, error / relief | 16.678, 0.336, 0.368, 7.65 % | 16.678, 0.336, 0.368, 7.65 % | < 1e-12 |
| whole images: pairs | 359 | 359 | same pairs, images and transforms |
| whole images: labelMAE / alignedMAE / deployMAE | 11.2851 / 11.6484 / 11.6744 | 11.2851 / 11.6484 / 11.6744 | < 1e-11 |
| whole images: MOLA deployMAE | 18.9749 | 18.9749 | < 1e-11 |
| whole images: `<1 km` r / amplitude, error / relief | 0.7303 / 0.8067, 1.341 % | 0.7303 / 0.8067, 1.341 % | < 1e-11 |
| whole images: better than MOLA | 342 of 359 | 342 of 359 | 0 |
| HiRISE: in place, 618 DEMs, deployMAE vs MOLA | 8.5579 vs 17.8215 | 8.5579 vs 17.8215 | 0 at the reference's rounding |
| HiRISE: co-registered, 581 DEMs, labelMAE / alignedMAE / deployMAE vs MOLA | 8.0224 / 8.5912 / 8.6214 vs 18.5148 | 8.0224 / 8.5912 / 8.6214 vs 18.5148 | 0 at the reference's rounding |
| HiRISE: 98 DEMs with a CTX stereo DEM, deployMAE (model / stereo / MOLA) | 11.2 / 10.0 / 21.7 | 11.2 / 10.0 / 21.7 | Table 3 cells: 0 of 27 differ |

With the default footprint of 168 m the tile medians are labelMAE 7.176,
alignedMAE 10.387, `<1 km` r 0.7726 and amplitude 0.7828, error / relief
4.606 % (job 6202166): alignedMAE +0.010 m and amplitude +0.0005 against the
150 m of the paper's scores.

## Pre-release R1 (epoch 88; version 0.1.0, October 2026)

Configuration note of 2026-10-16: prediction, evaluation, leveling and
verification use the same nominal 168 m FWHM footprint.
`anchor.PRODUCT_FOOTPRINT_M` is now an alias of `anchor.FOOTPRINT_M`.
The results below record previous runs; this configuration edit does not
regenerate their inputs or establish new bitwise agreement. The R1
job scripts pass the settings of pre-release R1 explicitly (`FOOTPRINT_M=150`,
`RELEASE1_RULES=1`, `EMBED_CONFIG=0`). Use a fresh output directory when
rerunning checks so cached scores are not reused.

| step | script | what was compared | result |
|---|---|---|---|
| network | `check_model.py` | outputs on random inputs and on 8 test samples, CPU and GPU | bitwise |
| training samples | `check_data.py` | 64 validation and 64 augmented training samples | bitwise |
| training | `check_train_steps.py` | loss of each of 4 optimiser steps (CPU) and every weight after them | bitwise |
| prediction | `check_predict.py` | 3 whole images (GPU and CPU): DEM, rescaled DEM, image and MOLA grids | bitwise |
| registration | `check_register.py` | 10 images of an equatorial and a polar tile, both stages | within 5 µm (similarity), 1e-17 (refinement) |
| PEDR field | `check_field.py` | Global CTX Mosaic corrections, 379 tile vectors, 300 shifted images | identical |
| leveling | `check_tile_chain.py level` | offsets, leveled DEMs, weights and leveled mosaic of 2 tiles (57 and 138 DEMs) | bitwise |
| infill | `check_tile_chain.py infill` | infill surface and provenance of 2 tiles | provenance bitwise; surface within 1.5 mm |
| join and de-seam | `check_tile_chain.py seams` | joined tile, seam mask, dominant-DEM index, de-seamed tile | bitwise |
| global mosaic | `check_mosaic.py` | cross-tile leveling field and rendered DEM and mask of 4 tiles | bitwise |
| uncertainty | `check_uncertainty.py` | look-up table; calibration of 8 test pairs, and of 8 more with infilled pixels; expected-error and class rasters of one tile | identical (bitwise) |
| Table 1 (tiles), Table 2 | `check_evaluate.py` | every measure of 6,396 test samples; per-sample bin sums of 6,684 crops; every cell | identical (samples at the reference's rounding; bin sums and cells bitwise) |
| Table 1 (whole images) | `check_images.py` | stereo DEMs, grids and every measure of the 359 pairs | identical; fine-scale measures within 1e-13 |
| Tables 3 and 4 | `check_hirise.py` | 1,172 windows over 619 HiRISE DEMs, per-DEM values, every cell and the Section 3.2 numbers | identical (windows bitwise; per-DEM values at the reference's rounding) |

Notes.

* Infill. The infill networks run on the GPU with non-deterministic
  convolution algorithms: two runs of this code on the same tile differ from
  each other as much as from the product (at most 1 mm, on 6 % of the pixels),
  so the remaining difference is run-to-run rounding.
* Registration. The matcher draws its random trees from OpenCV's generator,
  which `register.sift_matches` resets for every image, as when each image is
  registered in its own process. The similarity then agrees to within 5 µm (the
  rest is the order of floating-point operations).
* Whole images. The grids are bit-identical when the bilinear warp keeps its
  kernel at four pixels (GDAL warp options `XSCALE=1`, `YSCALE=1`, GDAL 3.6 or
  later); without them GDAL widens the kernel where the grids' scales differ
  and the predictions differ by about 2 cm.
* Uncertainty. Some product tiles of the epoch-98 mosaic were rebuilt after the
  layer was calibrated; the infill calibration of a test pair is compared only
  where its tiles are older than the calibration.
* HiRISE. The image shift of the stereo DEMs and of the predictions at the 76
  DEMs with a stereo DEM is computed in ground metres (52 and 66 m); the
  value of 67 m in the text was computed in projected metres.
* Environments. The uncertainty calibration and the HiRISE scoring are checked
  in the environment they were run in (rasterio 1.4.3 with GDAL 3.6.2); the
  rest in PyTorch 2.0 with GDAL 3.9.3.

### Version 0.2 (October 2026)

* Name. The package is `mesa` (the product was renamed from MADER to MESA);
  the checks above were run on version 0.1.0 as `mader`. Every module of
  version 0.2 imports and prints its command-line help in the environments of
  the checks (`import mesa.<module>`, `python -m mesa.<module> --help`,
  revision job rev8_check 5923013 on xeon-p8, 2026-10-14: all 20 modules import
  and print their help in the `planetary_dtm` environment (Python 3.11.13,
  rasterio 1.4.3, GDAL 3.6.2) and in the `anaconda/2023a-pytorch` module
  environment (Python 3.9.16, PyTorch 2.0.0+cu117)).
* Shot-anchoring footprint. Version 0.1.0 anchored each prediction window to
  the PEDR shots through a 150 m footprint (`anchor.FOOTPRINT_M = 150`); the
  paper states 168 m and the leveling used 168 m. The released frames (epoch
  88, October 2026) and the paper's tile scoring were computed with 150 m.
  The initial version 0.2 verification scripts explicitly passed the old
  150 m value to reproduce those comparisons. As of 2026-10-16 both scripts
  use the common 168 m value; their next results must be recorded separately.
  Effect of the change, measured on 3 whole test images predicted on CPU
  nodes (the network run once, the windows anchored with each footprint, the
  frames scored against the stereo DEM with `mesa.evaluate_images`, the
  tile-merged labels and the registration of `data/test_images.csv`; job
  5923013, 12 threads per image): B01_009956_2169 (Ep050_Np30, 549 windows,
  18.0 M pixels) mean |168 m - 150 m| 0.049 m, median 0.031 m, 99th percentile
  0.28 m, largest 0.80 m; deployMAE 13.252 -> 13.250 m, alignedMAE 13.083 ->
  13.092 m, labelMAE 12.468 -> 12.473 m. B01_010092_2003 (E-050_Np20, 209
  windows, 7.7 M pixels) mean 0.024 m, median 0.012 m, p99 0.19 m, largest
  0.59 m; deployMAE 11.146 -> 11.142 m, alignedMAE 11.006 -> 11.001 m,
  labelMAE 10.596 -> 10.589 m. V15_085759_1292 (E-070_N-60, 292 windows,
  9.4 M pixels) mean 0.014 m, median 0.008 m, p99 0.10 m, largest 0.16 m;
  deployMAE 17.176 -> 17.171 m, alignedMAE 15.724 -> 15.724 m, labelMAE 12.821
  -> 12.826 m. The mean signed change is below 0.003 m on each image and every
  window of the three images was fitted with both footprints. The 150 m
  frames of this run differ from the production frames by at most 0.0012 m
  (12 threads here against 8 in production; `check_predict.py` with the
  production thread count is bitwise).
* Footprint product. `mesa.footprints` was checked on the three tiles
  E-020_N-30 (eqc, 118 images), Ep040_N-50 (sinu, 93) and E-020_Np80 (sinu,
  7) against the per-tile census of the revision (Task 3,
  `tile_frames_csv.py`, job 5923013): identical. For every image of the three
  tiles the index, covered, sole and dominant pixel counts are the same, the
  dominant and weight shares agree within 1e-9, and the product ids,
  acquisition times and solar geometry are equal; the summaries agree
  (1,094,998,893 / 788,641,373 / 49,998,675 predicted pixels, 65,198 / 58,773 /
  32,230 without a dominant image, dominant shares summing to 0.99994 /
  0.99993 / 0.99936, 50.8 / 52.7 / 13.2 % of predicted pixels from two or
  more images, 0 images without an index date). Run time 359 s, 307 s and
  30 s per tile on one node.
