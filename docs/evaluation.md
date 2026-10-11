# Reproducing the evaluation

This page gives, for each table of the paper, the command that computes it and
the inputs it needs. The network is `mader_e088.pt`; the test set is the 359
stereo pairs of `data/test_pairs.txt`. Medians are over samples (tiles), test
pairs (whole images) or HiRISE DEMs, as in the table captions.

Resampling with GDAL is part of several steps, and GDAL's bilinear warp has
changed between versions where the two grids differ in scale. The numbers of
the paper were computed with GDAL 3.9 (tiles), 3.13 (the grids of the whole
images; `mader.evaluate_images` reproduces them with GDAL 3.6 or later) and
3.6 (HiRISE and the uncertainty calibration); with another version the
resampled values can differ by about a centimetre.

## Table 1, tile rows, and Table 2

Inputs: the training archive (`dem/`, `ctx/`, `mola_bilinear/`, `d1_shot/`,
`shot/` and `ctx_overlapping_95pct_polar.json`), the test manifests in `data/`
and the MOLA gridded DEM as distributed.

```bash
python -m mader.evaluate tiles --model mader_e088.pt --root mars_ctx_dem_v4_pedr \
    --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \
    --pairs data/test_pairs.txt --out tiles.csv
python -m mader.evaluate mola-tiles --root mars_ctx_dem_v4_pedr --samples tiles.csv \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out tiles_mola.csv
for band in eq mid polar; do
  python -m mader.evaluate bins --model mader_e088.pt --root mars_ctx_dem_v4_pedr \
      --manifest data/test_manifest_$band.csv --pairs data/test_pairs.txt \
      --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out bins_$band.npz
done
python -m mader.evaluate tables --samples tiles.csv --mola-tiles tiles_mola.csv --pairs data/test_pairs.txt \
    --bins bins_eq.npz bins_mid.npz bins_polar.npz --out tables_tiles.json
```

`tiles` scores the central 384 px window of every test tile under every CTX
image that covers it (6,396 samples of the 359 pairs, 2,878 tiles). For each
sample it records labelMAE, alignedMAE, the `<1 km` correlation and amplitude
ratio, the relief of the stereo DEM over the window and how the window was
anchored to the shots. A test tile's image was registered to its stereo DEM
when the archive was built, so the surface scored on a tile is already aligned
and is not the deployed product: the tile rows have no deployMAE, and their
error / relief is alignedMAE over the relief. `mola-tiles` scores MOLA as distributed on the same
windows, once per tile. `tables` prints the Table 1 tile rows, the per-sample
change from alignedMAE to labelMAE, the relief, the solar-incidence block of
Table 2 (medians over samples by the incidence of the image) and the
consistency between views of a tile (standard deviation of labelMAE between
the images of a tile).

`bins` reproduces the crops of the validation loader exactly: 384 px at a
random position drawn by Python's generator, seed 0, batch 16, eight loader
workers each seeded with its index, over all 375 pairs of the test manifests
(pairs outside the 359 are drawn but not counted). Changing the batch size or
the number of workers changes the crops. Every pixel's labelMAE and alignedMAE
errors are binned by

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
lower bin (a pixel at exactly 20 m of disagreement is in 0-20 m).

Expected values (epoch 88): Table 1 tiles 7.2 m labelMAE, 10.5 m alignedMAE,
`<1 km` r 0.77 and amplitude 0.79, error 4.6 % of relief; MOLA 16.7 m, 0.34,
0.37, 7.6 %. `verification/check_evaluate.py` compares every sample and every
bin with the scoring behind the paper.

## Table 1, whole-image rows

Inputs: the stereo DEMs of the test pairs, rebuilt from the test tiles of the
archive (`labels`), the whole-image predictions of the images named in
`data/test_images.csv` (`mader.predict`, one per pair, written to
`preds/<tile>/<image>/`), and the MOLA gridded DEM as distributed.

```bash
python -m mader.evaluate_images labels --root mars_ctx_dem_v4_pedr \
    --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \
    --pairs data/test_pairs.txt --out labels
python -m mader.evaluate_images score --images data/test_images.csv --labels labels --preds preds \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out images.csv
python -m mader.evaluate_images tables --scores images.csv --out tables_images.json
```

The co-registration of each image with its stereo DEM was measured from the
images alone and does not depend on the model, so it is given in
`data/test_images.csv` and applied as it is: a rotation and shift (335 pairs,
cubic-spline resampling), or a shift in whole pixels (24 pairs). The
prediction and MOLA are warped onto the stereo DEM's grid with GDAL's bilinear
kernel held at four pixels (`XSCALE=YSCALE=1`, GDAL 3.6 or later), which is
what the grids of the paper used. A whole image is scored as deployed, so its
error / relief is deployMAE over the relief. `tables` prints the whole-image
rows of Table 1 and the statistics of Section 3.1 (the change with
co-registration, the level in place, the pairs poleward of 80 deg and the
classes of relief).

Expected values (epoch 88), all latitudes: labelMAE 9.8 m, alignedMAE 10.6 m,
deployMAE 10.5 m, `<1 km` r 0.75 and amplitude 0.82, error 1.9 % of relief;
MOLA 18.4 m, 0.33, 0.40, 3.2 %.

## Tables 3 and 4, and the HiRISE numbers of Section 3.2

Inputs, laid out as the tables in `data/` expect (paths relative to the
working directory): the HiRISE DTMs and their orthoimages from the PDS
(`hirise_dtm/`, `hirise_ortho/`, each JP2 with its `.LBL`), the predictions of
the CTX images over them (`preds/<tile>/<image>/`, from `mader.predict`) and
the registered images they were predicted from (`aligned_pedr/<tile>/`), the
stereo DEMs of the test pairs (`labels/`, from `mader.evaluate_images labels`)
and their tied orthoimages (`orthoimages/`), and the MOLA gridded DEM as
distributed.

```bash
python -m mader.evaluate_hirise score --dems data/hirise_dems.csv --windows data/hirise_windows.csv \
    --stereo data/hirise_stereo_pairs.csv --test-pairs data/test_pairs.txt \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out hirise_scores.csv --jobs 24
python -m mader.evaluate_hirise tables --dems data/hirise_dems.csv --scores hirise_scores.csv \
    --out tables_hirise.json
```

Every HiRISE DEM is compared with each prediction that covers it (1,172
windows over 619 DEMs; 618 have a window with enough pixels). The CTX image of
a window is co-registered with the HiRISE orthoimage by masked normalised
cross-correlation of high-passed images, with a coarser band-pass estimate as
fallback; 581 DEMs have at least one co-registered window. A DEM's value is
the mean over its windows, and Table 3 gives medians over DEMs. Table 4 pools
the difference-of-Gaussian band statistics over the DEMs. `tables` also prints
the numbers of Section 3.2 (distance from the training ground, latitude bands,
slope error, image shift, change with alignment and rescaling, scale, the
poles). The scoring takes about an hour on 36 cores.

Expected values (epoch 88): deployMAE 8.6 m against 17.8 m for MOLA over 618
DEMs; at the 581 co-registered, labelMAE 7.8 m, alignedMAE 8.5 m, deployMAE
8.7 m.
