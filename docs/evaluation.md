# Reproducing the evaluation

This page gives, for each table of the paper, the command that computes it and
the inputs it needs. The network is `mesa_e098.pt`, the epoch-98 network with
the B-spline decoder (v3); the test set is the 359 stereo pairs of
`data/test_pairs.txt`. Medians are over samples (tiles), test pairs (whole
images) or HiRISE DEMs, as in the table captions. The commands below reproduce
the numbers of the paper; `verification/README.md` records the check of every
number against the manuscript's own scoring.

Anchoring footprint. The paper gives 168 m for the full width at half maximum
of the footprint through which a prediction is fitted to the PEDR shots, and
168 m is the default of `mesa.predict`, `mesa.evaluate tiles` and
`mesa.evaluate bins`. The frames of the v3 product, from which the
whole-image and HiRISE rows are scored, and the tile scores of the paper were
computed with 150 m. The commands below pass `--footprint 150` where the
footprint enters; with 168 m the medians change by a few millimetres
(`verification/README.md`).

Resampling with GDAL is part of several steps, and GDAL's bilinear warp has
changed between versions where the two grids differ in scale. The numbers of
the paper were computed with GDAL 3.9 (tiles), 3.13 (the grids of the whole
images; `mesa.evaluate_images` reproduces them with GDAL 3.6 or later) and
3.6 (HiRISE and the uncertainty calibration); with another version the
resampled values can differ by about a centimetre.

## Table 1, tile rows

Inputs: the training archive (`dem/`, `ctx/`, `mola_bilinear/`, `d1_shot/`,
`shot/` and `ctx_overlapping_95pct_polar.json`), the test manifests in `data/`
and the MOLA gridded DEM as distributed.

```bash
python -m mesa.evaluate tiles --model mesa_e098.pt --root mars_ctx_dem_v4_pedr \
    --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \
    --pairs data/test_pairs.txt --out tiles.csv --footprint 150
python -m mesa.evaluate mola-tiles --root mars_ctx_dem_v4_pedr --samples tiles.csv \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out tiles_mola.csv
python -m mesa.evaluate tables --samples tiles.csv --mola-tiles tiles_mola.csv --pairs data/test_pairs.txt \
    --out tables_tiles.json
```

`tiles` scores the central 384 px window of every test tile under every CTX
image that covers it (6,396 samples of the 359 pairs, 2,878 tiles). For each
sample it records labelMAE, alignedMAE, the `<1 km` correlation and amplitude
ratio, the relief of the stereo DEM over the window and how the window was
anchored to the shots. A test tile's image was registered to its stereo DEM
when the archive was built, so the surface scored on a tile is already aligned
and is not the deployed product: the tile rows have no deployMAE, and their
error / relief is alignedMAE over the relief. `mola-tiles` scores MOLA as
distributed on the same windows, once per tile. `tables` prints the Table 1
tile rows, the per-sample change from alignedMAE to labelMAE, the relief, the
error by the solar incidence of the image (medians over samples) and the
consistency between views of a tile (standard deviation of labelMAE between
the images of a tile). On a CPU the 6,396 samples take about 8 h on four
threads; `verification/run_tiles_chunked.py` runs them in parallel processes.

Expected values (v3): labelMAE 7.18 m, alignedMAE 10.38 m, `<1 km` r 0.773 and
amplitude 0.782, error 4.61 % of relief; MOLA 16.68 m, 0.336, 0.368, 7.65 %.

## Table 2

`mesa.evaluate bins` reproduces the crops of the validation loader exactly:
384 px at a random position drawn by Python's generator, seed 0, batch 16,
eight loader workers each seeded with its index, over all 375 pairs of the
test manifests (pairs outside the 359 are drawn but not counted). Changing the
batch size or the number of workers changes the crops.

```bash
for band in eq mid polar; do
  python -m mesa.evaluate bins --model mesa_e098.pt --root mars_ctx_dem_v4_pedr \
      --manifest data/test_manifest_$band.csv --pairs data/test_pairs.txt \
      --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out bins_$band.npz --footprint 150
done
python -m mesa.evaluate tables --samples tiles.csv --mola-tiles tiles_mola.csv --pairs data/test_pairs.txt \
    --bins bins_eq.npz bins_mid.npz bins_polar.npz --out tables_tiles.json
```

Every pixel's labelMAE and alignedMAE errors are binned by

* the disagreement between the MOLA input (the half-cell corrected grid of
  `mola_bilinear/`) and the stereo DEM, over all valid pixels;
* the distance from the nearest shot, recovered from the network's input
  `d_shot = exp(-d1 / 2000 m)`;
* the slope, the aspect relative to the solar azimuth (folded to 0-180 deg) and
  the local incidence angle of the predicted surface (the prediction rescaled
  with the stereo DEM's statistics), by central differences; aspect and local
  incidence only on pixels the predicted surface turns to the Sun.

A cell is the median over samples of each sample's MAE in the bin, each sample
weighted by its pixels in the bin. The MOLA row of the shot-distance block is
MOLA as distributed, warped bilinearly onto each crop. Bin edges belong to the
lower bin (a pixel at exactly 20 m of disagreement is in 0-20 m). The binning
was checked against the manuscript's scoring of epoch 88
(`verification/README.md`); the v3 manuscript did not re-score the bins.

## Table 1, whole-image rows

Inputs: the full stereo DEMs of the test pairs (`<pair>.tif`, part of the
evaluation data of the paper: the tied DEM of the stereo pipeline without hole
filling, or the filled DEM masked to the tied DEM's pixels, cropped to its
valid extent), the whole-image predictions of the images named in
`data/test_images.csv` (`mesa.predict --footprint 150`, one per pair, written
to `preds/<tile>/<image>/`; these are the frames of the v3 product), and the
MOLA gridded DEM as distributed.

```bash
python -m mesa.evaluate_images score --images data/test_images.csv --labels stereo_dems_full --preds preds \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out images.csv
python -m mesa.evaluate_images tables --scores images.csv --out tables_images.json
```

The image of each pair is the one selected for release 1 and kept for the
paper. The co-registration of each image with its stereo DEM was measured from
the images alone and does not depend on the model, so it is given in
`data/test_images.csv` and applied as it is: a rotation and shift (319 pairs,
and 2 pairs with the transform of the region of the test tiles; cubic-spline
resampling), or a shift in whole pixels (35 consensus shifts of the windows,
2 shifts of the whole overlap, 1 shift of the region of the test tiles). The
prediction and MOLA are warped onto the stereo DEM's grid with GDAL's bilinear
kernel held at four pixels (`XSCALE=YSCALE=1`, GDAL 3.6 or later), which is
what the grids of the paper used. A whole image is scored as deployed, so its
error / relief is deployMAE over the relief. MOLA is scored on every valid
pixel of the stereo DEM within the grid. `tables` prints the whole-image rows
of Table 1 and the statistics of Section 3.1 (the change with
co-registration, the level in place, the pairs poleward of 80 deg and the
classes of relief).

Expected values (v3), all latitudes: labelMAE 11.29 m, alignedMAE 11.65 m,
deployMAE 11.67 m, `<1 km` r 0.730 and amplitude 0.807, error 1.34 % of relief;
MOLA 18.97 m, 0.313, 0.368, 2.29 %. The model is better than MOLA at 342 of
the 359 pairs.

Release 1 scored the whole images against stereo DEMs merged from the test
tiles (`mesa.evaluate_images labels`), which cover only the test tiles of a
pair, with transforms fitted on those tiles.

## Tables 3 and 4, and the HiRISE numbers of Section 3.2

Inputs, laid out as the tables in `data/` expect (paths relative to the
working directory): the HiRISE DTMs and their orthoimages from the PDS
(`hirise_dtm/`, `hirise_ortho/`, each JP2 with its `.LBL`), the predictions of
the CTX images over them (`preds/<tile>/<image>/`, from `mesa.predict
--footprint 150`; the frames of the v3 product) and the registered images
they were predicted from (`aligned_pedr/<tile>/`), the full stereo DEMs of the
test pairs (`stereo_dems_full/`, as for the whole images) and their tied
orthoimages (`orthoimages/<stereo directory>/`), and the MOLA gridded DEM as
distributed.

```bash
python -m mesa.evaluate_hirise score --dems data/hirise_dems.csv --windows data/hirise_windows.csv \
    --stereo data/hirise_stereo_pairs.csv --test-pairs data/test_pairs.txt \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out hirise_scores.csv --jobs 24
python -m mesa.evaluate_hirise tables --dems data/hirise_dems.csv --scores hirise_scores.csv \
    --out tables_hirise.json
```

Every HiRISE DEM is compared with each prediction that covers it (the 1,172
windows of release 1 over 619 DEMs; 618 have a window with enough pixels). The
CTX image of a window is co-registered with the HiRISE orthoimage by masked
normalised cross-correlation of high-passed images, with a coarser band-pass
estimate as fallback. The paper keeps the outcome of this co-registration
from release 1 (column `registered_r1` of `data/hirise_windows.csv`): 1,088
windows at 581 DEMs are co-registered. A DEM's value is the mean over its
windows, and Table 3 gives medians over DEMs. Table 4 pools the
difference-of-Gaussian band statistics over the DEMs. The stereo rows compare
the model with the full stereo DEMs of the test pairs over 98 DEMs (132
pairs), each stereo DEM co-registered with HiRISE through its own
orthoimage. `tables` also prints the numbers of Section 3.2 (distance from the
training ground, latitude bands, slope error, image shift, change with
alignment and rescaling, scale, the poles). The scoring takes about an hour
on 36 cores.

Expected values (v3): deployMAE 8.56 m against 17.82 m for MOLA over 618 DEMs;
at the 581 co-registered, labelMAE 8.02 m, alignedMAE 8.59 m, deployMAE
8.62 m, MOLA 18.51 m; at the 98 DEMs with a stereo DEM, labelMAE 9.68 m
(stereo 5.70 m), alignedMAE 10.05 m (8.78 m), deployMAE 11.19 m (9.99 m).
