# Reproducing MADER, step by step

This page lists every step from the inputs to the released mosaic and to the
numbers of the paper, with the exact settings used. Paths are examples. Steps
marked *external* use ISIS or the Ames Stereo Pipeline (ASP) and are given as
commands; everything else is in this package.

Software used for the release: ISIS 8, ASP 3.6.0, GDAL 3.9, Python 3.9-3.11,
PyTorch 2.0, NumPy 1.24-1.26, SciPy 1.9-1.16, rasterio 1.4, OpenCV 4.7
(registration only), Lightning 2.0 (infill training only).

## 0. Inputs

| input | source | used by |
|---|---|---|
| CTX EDRs and the CTX cumulative index | PDS Imaging Node | map projection, solar geometry |
| MOLA gridded DEM, 463 m (`Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`) | USGS Astrogeology | network input, levelling, infill, mosaic |
| MOLA PEDR shots, as an SQLite table `mola(LAT, LON, ELEV, ORBIT)` | PDS Geosciences Node | shot distance, anchoring, levelling |
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
+units=m +no_defs` (both occur in the product; `mader.level` brings them onto
one grid). The projection window is the image footprint (from the cumulative
index) intersected with the tile, padded by 0.05 deg.

Solar geometry from the cumulative index: incidence `INCIDENCE_ANGLE`, azimuth
`(SUB_SOLAR_AZIMUTH - NORTH_AZIMUTH) mod 360`.

## 2. Registration (`mader.register`, `mader.pedr_field`)

Reference crops are built with GDAL (`scripts/reference_crops.sh`): the
controlled mosaic resampled to 24 m on the global equirectangular grid over the
tile plus 0.3 deg, or the Global CTX Mosaic resampled to 24 m in the tile's
sinusoidal projection, then shifted by the tile's correction from
`data/global_ctx_mosaic_corrections.csv` (Text S8: the two mosaics differ by a
median of 88 m).

```bash
python -m mader.register --image mapped/X_mapped.tif --reference reference_24m.tif \
    --ref-nodata -32768 --ref-offset 32768 --out aligned/X_aligned.vrt              # equatorial
python -m mader.register --image mapped/X_mapped.tif --reference reference_24m_corrected.vrt \
    --ref-nodata 0 --translation-fallback --out aligned/X_aligned.vrt               # poleward of 30 deg
```

The translation onto the PEDR shots: match registered images to the PEDR-
registered orthoimages of the stereo pairs they overlap (at least 150 km^2),
fit the field, apply it:

```bash
python -m mader.pedr_field measure --pairs overlaps.csv --out matches.csv
python -m mader.pedr_field fit --matches matches.csv --out field.json
python -m mader.pedr_field apply --field field.json --images 'aligned/*_aligned.vrt' --out aligned_pedr
```

The matches of the mosaic (31,040, from 8,176 image-tile pairs) are in
`data/pedr_field_matches.csv`; `fit` on that file gives the field of the
product (379 tile vectors).

## 3. Training (`mader.train`)

```bash
python -m mader.train --root mars_ctx_dem_v4_pedr --manifest mars_ctx_dem_v4_pedr/manifest_v4sT_ge10.csv \
    --incidence-json mars_ctx_dem_v4_pedr/ctx_overlapping_95pct_polar.json --out runs/mader --seed 1
```

One H200 GPU, about 30 h for 100 epochs. The released model is epoch 88, the
epoch with the lowest validation MAE (8.80 m). The run was interrupted twice
and resumed after epochs 58 and 88 (`--resume`). Training is not bitwise
reproducible on a GPU; a retrain from another seed differs by about 0.05 m in
labelMAE (Text S5).

## 4. Prediction (`mader.predict`)

```bash
python -m mader.predict --image aligned_pedr/X_aligned.vrt --inc 45.45 --az 262.89 --model mader_e088.pt \
    --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite --out preds/X
```

An image overlapping several product tiles is predicted once per tile, from the
image registered in that tile.

## 5. Product tile (`scripts/run_tile.sh`)

```bash
python -m mader.level --tile E-030_N-10 --frames 'preds/*/pred_shot_anchor.tif' --mola MOLA.tif --pedr pedr.sqlite \
    --exclude-orbits data/orbit_screen/E-030_N-10.txt --out tiles/E-030_N-10
# CTX layer for the infill: the Global CTX Mosaic averaged onto the levelled grid (gdalwarp, see run_tile.sh)
python -m mader.infill --dem tiles/E-030_N-10/leveled.tif --ctx tiles/E-030_N-10/ctx_18m.tif --mola MOLA.tif \
    --coarse infill_coarse.pt --fine infill_fine.pt --out tiles/E-030_N-10/infill
python -m mader.seams merge --dem tiles/E-030_N-10/leveled.tif --model tiles/E-030_N-10/infill.model.tif \
    --out tiles/E-030_N-10/final.tif
python -m mader.seams deseam --final tiles/E-030_N-10/final.tif --prov tiles/E-030_N-10/infill.prov.tif \
    --frames 'tiles/E-030_N-10/leveled/*_leveled.tif' --out tiles/E-030_N-10/deseam.tif
```

Orbit screens (`data/orbit_screen/`) exist for the 216 equatorial tiles: MOLA
passes whose median difference from the gridded DEM exceeds 25 m over at least
100 shots are left out of the levelling (`mader.pedr.screen_orbits`).

## 6. Global mosaic (`mader.mosaic`)

```bash
python -m mader.mosaic coarse --tiles-root tiles --work global      # every tile
python -m mader.mosaic level  --tiles-root tiles --work global
python -m mader.mosaic render --tiles-root tiles --work global --mola MOLA.tif --threads 8
python -m mader.mosaic cog --work global --out-dem mader_dem.tif --out-mask mader_mask.tif
```

## 7. Infill training (`mader.infill_train`)

Training windows come from 71 levelled tiles (`data/infill_tiles_split.tsv`:
tile, split, stride), the gap shapes from `data/gap_census.npz`.

```bash
python -m mader.infill_train store --dem tiles/T/leveled.tif --ctx tiles/T/ctx_18m.tif --mola MOLA.tif \
    --out store_coarse --tile T --split train --stride 1024 --coarse-only --min-valid 0.4
python -m mader.infill_train train --stage coarse --store store_coarse --holes data/gap_census.npz \
    --init infill_coarse_pretrained.pt --epochs 40 --lr 5e-5 --patience 10 --max-hole-m 60000 --max-holes 2 \
    --out runs/infill_coarse
python -m mader.infill_train store ... --out store_fine --min-valid 0.5          # 18 m level kept
python -m mader.infill_train train --stage fine --store store_fine --holes data/gap_census.npz \
    --coarse runs/infill_coarse/best.pt --init runs/infill_coarse/best.pt --epochs 25 --lr 1e-4 --patience 8 \
    --max-holes 2 --known-weight 0.25 --out runs/infill_fine
```

Two nodes with two V100 GPUs each, batch 8 per GPU, 16-bit mixed precision.
The coarse network was fine-tuned from a network first trained on one tile
(Margaritifer Terra), released as `infill_coarse_pretrained.pt`.

## 8. Uncertainty layer (`mader.uncertainty`)

```bash
python -m mader.uncertainty calibrate --pred grids/PAIR/pred.tif --stereo grids/PAIR/stereo.tif --inc I --az A \
    --pedr pedr.sqlite --out calib/PAIR.npz                                     # every test pair
python -m mader.uncertainty calib-infill --label labels/PAIR.tif --tiles tiles/T/deseam.tif:tiles/T/infill.prov.tif \
    --out calib_infill/PAIR.npz
python -m mader.uncertainty lut --pairs 'calib/*.npz' --infill 'calib_infill/*.npz' --out lut.json
python -m mader.uncertainty layer --lut lut.json --dem tiles/T/deseam.tif --prov tiles/T/infill.prov.tif \
    --domid tiles/T/deseam_seams_domid.tif --suns images.csv --levelled 'tiles/T/leveled/*_leveled.tif' \
    --pedr pedr.sqlite --out tiles/T/T
```

## 9. Evaluation

`mader.evaluate` scores the test tiles (Table 1, tile rows, and Table 2),
`mader.evaluate_images` the whole CTX images of the test pairs (Table 1, image
rows) and `mader.evaluate_hirise` the HiRISE DEMs (Tables 3 and 4). The
commands and inputs are in `docs/evaluation.md`.
