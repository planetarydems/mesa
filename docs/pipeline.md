# Reproducing MESA, step by step

This page lists every step from the inputs to the released mosaic and to the
numbers of the paper, with the exact settings used. Paths are examples. Steps
marked *external* use ISIS or the Ames Stereo Pipeline (ASP) and are given as
commands; everything else is in this package.

This page describes the pipeline of pre-release R1 of the mosaic, with the model
of the paper (version 1, epoch 98) in steps 3 and 4. Version 1 of the mosaic
(October 2026) used the same inputs, the epoch-98 network and the stages
under `v3/`; the README section "Production chain of version 1" and `v3/README.md` list what
changed after the prediction (join, infill rule, seam repair, polar comb
correction, reviewed local corrections).

Software used for the release: ISIS 8, ASP 3.6.0, GDAL 3.9, Python 3.9-3.11,
PyTorch 2.0, NumPy 1.24-1.26, SciPy 1.9-1.16, rasterio 1.4, OpenCV 4.7
(registration only), Lightning 2.0 (infill training only), Shapely 2 and
Fiona 1.9 (footprint product only).

## 0. Inputs

| input | source | used by |
|---|---|---|
| CTX EDRs and the CTX cumulative index | PDS Imaging Node | map projection, solar geometry |
| MOLA gridded DEM, 463 m (`Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`) | USGS Astrogeology | network input, leveling, infill, mosaic |
| MOLA PEDR shots, as an SQLite table `mola(LAT, LON, ELEV, ORBIT)` | PDS Geosciences Node | shot distance, anchoring, leveling |
| controlled CTX mosaic quadrangles (Robbins et al., 2023) | USGS | registration within 30 deg of the equator |
| Global CTX Mosaic tiles (Dickson et al., 2024) | Murray Lab | registration poleward of 30 deg; CTX layer of the infill |
| training archive (stereo-DEM tiles, registered images, MOLA, d1, shots, manifests) | released with the paper | training and test tiles |
| `data/` of this repository | this repository | orbit screens, mosaic corrections, gap census, splits |

The PEDR table: one row per shot with planetocentric latitude, east longitude
(0-360), elevation above the areoid in metres and the orbit number, built from
the PEDR products of the PDS (about 595 million rows). The schema used is

```sql
CREATE TABLE mola (LON real, LAT real, ELEV real, RADIUS real, ORBIT integer);
CREATE INDEX molaIndex ON mola(LAT, LON);
```

Only LAT, LON, ELEV and ORBIT are read; the index makes the box queries fast.

## 1. Map projection of a CTX image (*external*)

Each image is ingested, given its SPICE kernels, radiometrically calibrated and
map projected at 6 m onto the MOLA gridded DEM converted to ellipsoid heights:

```bash
mroctx2isis from=B01_009956_2169_XN_36N304W.IMG to=img.cub
spiceinit from=img.cub web=no
ctxcal from=img.cub to=img_cal.cub
# MOLA over the tile (+0.5 deg), areoid -> ellipsoid
gdalwarp -te LON0 LAT0 LON1 LAT1 -te_srs "+proj=longlat +R=3396190 +no_defs" Mars_MGS_MOLA_DEM_mosaic_global_463m.tif mola_areoid.tif
dem_geoid --reverse-adjustment mola_areoid.tif -o mola_tile        # writes mola_tile-adj.tif
mapproject --tr 6 --t_srs "$TSRS" --nodata-value 0 --t_projwin ULX ULY LRX LRY mola_tile-adj.tif img_cal.cub B01_009956_2169_mapped.tif
```

`TSRS` is the projection of the product tile: `+proj=sinu +lon_0=<tile centre>
+x_0=0 +y_0=0 +R=3396190 +units=m +no_defs` for tiles poleward of 30 deg, and
for the equatorial tiles either the same sinusoidal form or the global
equirectangular `+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190
+units=m +no_defs` (both occur in the product; `mesa.level` brings them onto
one grid). The projection window is the image footprint (from the cumulative
index) intersected with the tile, padded by 0.05 deg.

Solar geometry from the cumulative index: incidence `INCIDENCE_ANGLE`, azimuth
`(SUB_SOLAR_AZIMUTH - NORTH_AZIMUTH) mod 360`.

## 2. Registration (`mesa.register`, `mesa.pedr_field`)

Reference crops are built with GDAL (`scripts/reference_crops.sh`): the
controlled mosaic resampled to 24 m on the global equirectangular grid over the
tile plus 0.3 deg, or the Global CTX Mosaic resampled to 24 m in the tile's
sinusoidal projection, then shifted by the tile's correction from
`data/global_ctx_mosaic_corrections.csv` (Text S8: the two mosaics differ by a
median of 88 m).

```bash
python -m mesa.register --image mapped/X_mapped.tif --reference reference_24m.tif \
    --ref-nodata -32768 --ref-offset 32768 --out aligned/X_aligned.vrt              # equatorial
python -m mesa.register --image mapped/X_mapped.tif --reference reference_24m_corrected.vrt \
    --ref-nodata 0 --translation-fallback --out aligned/X_aligned.vrt               # poleward of 30 deg
```

The translation onto the PEDR shots: match registered images to the PEDR-
registered orthoimages of the stereo pairs they overlap (at least 150 km^2),
fit the field, apply it:

```bash
python -m mesa.pedr_field measure --pairs overlaps.csv --out matches.csv
python -m mesa.pedr_field fit --matches matches.csv --out field.json
python -m mesa.pedr_field apply --field field.json --images 'aligned/*_aligned.vrt' --out aligned_pedr
```

The matches of the mosaic (31,040, from 8,176 image-tile pairs) are in
`data/pedr_field_matches.csv`; `fit` on that file gives the field of the
product (379 tile vectors).

## 3. Training (`mesa.train`)

```bash
python -m mesa.train --root mars_ctx_dem_v4_pedr --manifest mars_ctx_dem_v4_pedr/manifest_v4sT_ge10.csv \
    --incidence-json mars_ctx_dem_v4_pedr/ctx_overlapping_95pct_polar.json --out runs/mesa --seed 1
```

The architecture comes from `--config` (default `mesa/configs/mesa_e098.json`,
the B-spline decoder). The model of the paper (`mesa_e098.pt`) is epoch 98 of
a 100-epoch run from scratch with seed 1, the epoch with the lowest validation
MAE (8.76 m); the run was resumed once from its epoch-74 checkpoint
(`--resume`). The model of pre-release R1 (`mesa_e088.pt`, `--config
mesa/configs/mesa_e088.json`) is epoch 88 of the same recipe with the
transposed-convolution decoder (one H200 GPU, about 30 h for 100 epochs;
validation MAE 8.80 m; resumed after epochs 58 and 88). Training is not
bitwise reproducible on a GPU; a retrain from another seed differs by about
0.05 m in labelMAE (Text S5).

## 4. Prediction (`mesa.predict`)

```bash
python -m mesa.predict --image aligned_pedr/X_aligned.vrt --inc 45.45 --az 262.89 --model mesa_e098.pt \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite --out preds/X --footprint 168
```

An image overlapping several product tiles is predicted once per tile, from the
image registered in that tile.

The windows of a prediction are fitted to the PEDR shots through a Gaussian
footprint of 168 m FWHM (`mesa.anchor`), matching Section 2.5, evaluation and
the leveling of step 5. The frames of both product releases were fitted with
150 m (`--footprint 150` reproduces the frames of version 1 bit for bit); the
footprint sensitivity measurements are in `verification/README.md`. Two
rules of the prediction of version 1 apply: PEDR shots are queried over all longitudes
for a frame that spans 360 deg of longitude or reaches within 2 deg of a pole,
and a frame whose shot-distance input averages below 0.05 is not predicted
(the shot query failed).

The prediction list of the mosaic held 53,722 placements (tile, image) of
45,580 CTX images (`data/images.csv` of version 0.1.0). 20 placements failed:
in 18 the image's part inside the tile is narrower than one 384 px window at
18 m, in 2 no window has 25 % valid image (`mesa.predict` raises the same
errors). The mosaic therefore holds 53,702 placements of 45,570 images, the
rows of `data/images.csv`; the 20 are listed with their reasons in
`data/images_not_in_mosaic.csv`. 10 of the 20 images are present in another
tile.

## 5. Product tile (`scripts/run_tile.sh`)

```bash
python -m mesa.level --tile E-030_N-10 --frames 'preds/*/pred_shot_anchor.tif' --mola MOLA.tif --pedr pedr.sqlite \
    --exclude-orbits data/orbit_screen/E-030_N-10.txt --out tiles/E-030_N-10
# CTX layer for the infill: the Global CTX Mosaic averaged onto the leveled grid (gdalwarp, see run_tile.sh)
python -m mesa.infill --dem tiles/E-030_N-10/leveled.tif --ctx tiles/E-030_N-10/ctx_18m.tif --mola MOLA.tif \
    --coarse infill_coarse.pt --fine infill_fine.pt --out tiles/E-030_N-10/infill
python -m mesa.seams merge --dem tiles/E-030_N-10/leveled.tif --model tiles/E-030_N-10/infill.model.tif \
    --out tiles/E-030_N-10/final.tif
python -m mesa.seams deseam --final tiles/E-030_N-10/final.tif --prov tiles/E-030_N-10/infill.prov.tif \
    --frames 'tiles/E-030_N-10/leveled/*_leveled.tif' --out tiles/E-030_N-10/deseam.tif
```

Orbit screens (`data/orbit_screen/`) exist for the 216 equatorial tiles: MOLA
passes whose median difference from the gridded DEM exceeds 25 m over at least
100 shots are left out of the leveling (`mesa.pedr.screen_orbits`).

## 6. Global mosaic (`mesa.mosaic`)

```bash
python -m mesa.mosaic coarse --tiles-root tiles --work global      # every tile
python -m mesa.mosaic level  --tiles-root tiles --work global
python -m mesa.mosaic render --tiles-root tiles --work global --mola MOLA.tif --threads 8
python -m mesa.mosaic cog --work global --out-dem mesa_dem.tif --out-mask mesa_mask.tif
```

## 7. Infill training (`mesa.infill_train`)

Training windows come from 71 leveled tiles (`data/infill_tiles_split.tsv`:
tile, split, stride), the gap shapes from `data/gap_census.npz`.

```bash
python -m mesa.infill_train store --dem tiles/T/leveled.tif --ctx tiles/T/ctx_18m.tif --mola MOLA.tif \
    --out store_coarse --tile T --split train --stride 1024 --coarse-only --min-valid 0.4
python -m mesa.infill_train train --stage coarse --store store_coarse --holes data/gap_census.npz \
    --init infill_coarse_pretrained.pt --epochs 40 --lr 5e-5 --patience 10 --max-hole-m 60000 --max-holes 2 \
    --out runs/infill_coarse
python -m mesa.infill_train store ... --out store_fine --min-valid 0.5          # 18 m level kept
python -m mesa.infill_train train --stage fine --store store_fine --holes data/gap_census.npz \
    --coarse runs/infill_coarse/best.pt --init runs/infill_coarse/best.pt --epochs 25 --lr 1e-4 --patience 8 \
    --max-holes 2 --known-weight 0.25 --out runs/infill_fine
```

Two nodes with two V100 GPUs each, batch 8 per GPU, 16-bit mixed precision.
The coarse network was fine-tuned from a network first trained on one tile
(Margaritifer Terra), released as `infill_coarse_pretrained.pt`.

## 8. Uncertainty layer (`mesa.uncertainty`)

```bash
python -m mesa.uncertainty calibrate --pred grids/PAIR/pred.tif --stereo grids/PAIR/stereo.tif --inc I --az A \
    --pedr pedr.sqlite --out calib/PAIR.npz                                     # every test pair
python -m mesa.uncertainty calib-infill --label labels/PAIR.tif --tiles tiles/T/deseam.tif:tiles/T/infill.prov.tif \
    --out calib_infill/PAIR.npz
python -m mesa.uncertainty lut --pairs 'calib/*.npz' --infill 'calib_infill/*.npz' --out lut.json
python -m mesa.uncertainty layer --lut lut.json --dem tiles/T/deseam.tif --prov tiles/T/infill.prov.tif \
    --domid tiles/T/deseam_seams_domid.tif --suns images.csv --levelled 'tiles/T/leveled/*_leveled.tif' \
    --pedr pedr.sqlite --out stage1/T/T
```

These four steps give the stage-1 score (`stage1/<tile>/<tile>_expected_error_m.tif`;
the production table is `data/uncertainty_lut_single_image.json`). The released
values come from stage 2, which calibrates the score against the assembled
mosaic. Each pixel is grouped by its score (bins 0, 5, 10, 15, 25, 40, 60 m), by
the number of images averaged there (`footprints/<tile>/<tile>_overlap_depth.tif`,
N = 1 or N >= 2) and, if infilled, by its stage-1 distance class. The median and
the 90th percentile of |mosaic - stereo DEM| of each group over the test pairs,
after the image co-registration of the mosaic itself
(`data/uncertainty_mosaic_coreg.csv`), are the released values
(`data/uncertainty_lut_mosaic.json`):

```bash
python -m mesa.uncertainty calibrate-mosaic --pair PAIR --label labels_full/PAIR.tif \
    --dem 'tiles/{tile}/deseam.tif' --prov 'tiles/{tile}/infill.prov.tif' \
    --score 'stage1/{tile}/{tile}_expected_error_m.tif' --depth 'footprints/{tile}/{tile}_overlap_depth.tif' \
    --lut1 data/uncertainty_lut_single_image.json --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif \
    --out calib_mosaic/PAIR.npz                                                 # every test pair
python -m mesa.uncertainty mosaic-lut --pairs 'calib_mosaic/*.npz' --lut1 data/uncertainty_lut_single_image.json \
    --out lut_mosaic.json
python -m mesa.uncertainty remap --lut lut_mosaic.json --dem tiles/T/deseam.tif --prov tiles/T/infill.prov.tif \
    --score stage1/T/T_expected_error_m.tif --depth footprints/T/T_overlap_depth.tif --out tiles/T/T
```

`remap` writes `<tile>_median_error_m.tif` and `<tile>_p90_error_m.tif` (float32
COG, nodata -9999) and `<tile>_uncertainty_class.tif` (uint8 COG: 0 median below
10 m, 1 10-25 m, 2 25 m or more, 3 infilled, 255 nodata). `--dem` names the
released surface of the tile (the lattice-filtered surface in the production
run). The values are empirical quantiles of the absolute elevation error against
reference DEMs over groups of pixels. They are not standard deviations and do not
give the uncertainty of a depth, slope or profile measured from the DEM. The
reserved HiRISE cohort of the validation, with the footprint exclusions, is
`data/uncertainty_hirise_cohort.csv`; `verification/check_uncertainty_mosaic.py`
compares this package with the production run.

## 9. Evaluation

`mesa.evaluate` scores the test tiles (Table 1, tile rows, and Table 2),
`mesa.evaluate_images` the whole CTX images of the test pairs (Table 1, image
rows) and `mesa.evaluate_hirise` the HiRISE DEMs (Tables 3 and 4). The
commands and inputs are in `docs/evaluation.md`.

## 10. Image footprints and per-pixel image provenance (`mesa.footprints`)

The DEM of a tile is a feather blend of the leveled DEMs of its images
(step 5), so a pixel inside an overlap is a weighted mean of predictions from
images of different dates. The footprint product records which images formed
each pixel. It is built per tile from the work directory of step 5 (the
weight rasters `leveled/<image>_w.tif`, the dominant-DEM index
`deseam_seams_domid.tif` written by `mesa.seams deseam` and the provenance
mask `infill.prov.tif`), the solar geometry of `data/images.csv` and the CTX
cumulative index (`cumindex.tab` with its label):

```bash
python -m mesa.footprints tile --tile E-030_N-10 --tile-dir tiles/E-030_N-10 --suns data/images.csv \
    --cumindex cumindex.tab --out footprints/E-030_N-10
python -m mesa.footprints merge --tiles 'footprints/*/*_footprints.gpkg' --out mesa_image_footprints.gpkg
```

Per tile: `<tile>_image_index.tif` (int16 COG, 18 m, no-data -1: the index of
the image with the largest blend weight at the pixel, the first of the tied
images deep inside an overlap, -1 at infilled pixels), `<tile>_overlap_depth.tif`
(int16 COG, 18 m, no-data 0: the number of leveled DEMs with valid data at the
pixel), `<tile>_frames.csv` (one row per index: product id, acquisition time,
the incidence and azimuth given to the network, the index's emission, phase,
solar longitude and orbit, the image's covered, sole, dominant and weight
shares of the tile), `<tile>_footprints.gpkg` (the valid-data extent of each
leveled DEM, traced at 72 m, simplified to 100 m, in planetocentric
longitude and latitude on the Mars sphere, IAU_2015:49900) and
`<tile>_frames_summary.json` (totals and checks). `merge` joins the per-tile
GeoPackages into one. The product's README (written with it) explains how a
pixel's dominant image, date and number of blended images are read. The
production run used the same module on the box work directories
(`--frames-list`, `--weights-dir`, `--weights-suffix`, `--domid`, `--prov`
name the production files).

## Selecting pixels for studies of surface change

Single-image predictions retain an identifiable acquisition date. For analyses
using tile mosaics, select valid elevations within each tile's nominal cell,
with one contributing image, a valid image index, and more than 720 m from
flagged infill. This excludes the measured reach of infill joins within the
audited tile cells; the infill flag marks the original gap, not the full join.
Local comparisons still require co-registration. Infill relief below about
300 m in wavelength remains unvalidated. Exclude low-confidence
reconstructions, whose numerical uncertainty has not been calibrated.
