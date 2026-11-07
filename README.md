# MESA

Mars Elevation from Shading and Altimetry. Code for *MESA: Mars Elevation from
Shading and Altimetry* (Mitchell et al., in review), an 18 m digital elevation
model of Mars from single CTX images: the network, its training, the
prediction of a DEM from one registered CTX image, the assembly of the global
mosaic (leveling against MOLA laser shots, gap infill, seam repair, cross-tile
leveling), the per-pixel uncertainty layer and the image footprint product,
together with the evaluation used in the paper.

The released model is the network at epoch 88 (`mesa_e088.pt`); the global
mosaic was built with it. The product was called MADER until October 2026;
version 1.0.0 of this repository was published under that name and its weight
file is `mader_e088.pt` (`CHANGELOG.md`).

## Layout

```
mesa/
  model.py           network: U-Net with illumination (FiLM) conditioning, DEM and albedo
                     heads, and the fusion with MOLA solved inside the network (Eq. 3)
  data.py            training and test samples from the tiled stereo-DEM archive
  loss.py            training objective (Sec. 2.2)
  train.py           training (Sec. 2.4)
  geo.py             map projections on the Mars sphere, MOLA sampling
  pedr.py            PEDR shot database, shot-distance layer, orbit screen
  anchor.py          fitting a prediction to the PEDR shots (gain and offset)
  predict.py         DEM from one registered CTX image (Sec. 2.5)
  register.py        registration of a map-projected image to a reference mosaic
  pedr_field.py      translation of registered images onto the PEDR shots
  level.py           leveling and blending of the DEMs of a product tile
  infill.py          gap infill: networks and inference (Sec. 2.6)
  infill_train.py    gap census, training windows, infill training
  seams.py           join of the infill, repair of butt-joint steps
  mosaic.py          cross-tile leveling, rendering on the global grid, COG
  uncertainty.py     per-pixel uncertainty layer: single-image score, calibration on the mosaic (Sec. 4.3, Text S13)
  footprints.py      image footprints and per-pixel image provenance of a tile (Sec. 4.4)
  evaluate.py        test tiles: Table 1 (tile rows) and Table 2
  evaluate_images.py whole CTX images against their stereo DEMs: Table 1 (image rows)
  evaluate_hirise.py HiRISE DEMs: Tables 3 and 4
scripts/             shell drivers (reference crops, one product tile)
data/                small inputs the pipeline needs (see below)
docs/pipeline.md     every step from the inputs to the mosaic, with the settings used
docs/evaluation.md   the commands behind each table of the paper
verification/        checks that this code reproduces the results of the paper and the mosaic
v3/                  the scripts of release v3 as run on the cluster (see "v3 chain" below)
PROVENANCE_v3.md     source path and md5 of every file under v3/
VERIFY_v3.md         replay of one production v3 box from the scripts under v3/
CHANGELOG.md         changes since the published version
```

## Installation

```bash
conda create -n mesa python=3.11 numpy scipy pytorch rasterio gdal opencv pyproj lightning shapely fiona -c conda-forge
pip install -e .
```

GDAL command-line tools are needed for the reference crops, the CTX layer of the
infill, the COG and the footprint rasters. ISIS and the Ames Stereo Pipeline
are needed only to map project new CTX images (docs/pipeline.md, step 1).
The results of the paper were computed with PyTorch 2.0, rasterio 1.4.3 and
GDAL 3.9.3 (3.6.2 for the HiRISE scoring and the uncertainty calibration);
other GDAL versions resample slightly differently (docs/evaluation.md).

## Data

External inputs (MOLA gridded DEM, PEDR shots, CTX images and mosaics, the
training archive) are listed in `docs/pipeline.md`. The `data/` directory holds:

| file | content |
|---|---|
| `images.csv` | the 53,702 image placements of the mosaic (45,570 CTX images; an image predicted in several tiles has one row per tile): tile, image, solar incidence and azimuth |
| `images_not_in_mosaic.csv` | the 20 queued placements whose prediction failed, with the reason (docs/pipeline.md, step 4) |
| `test_pairs.txt` | the 359 test stereo pairs |
| `test_manifest_{eq,mid,polar}.csv` | the test samples (tile, image) of the 375 pairs from which the test crops are drawn |
| `test_images.csv` | for each test pair, the CTX image scored whole, its product tile, latitude, and its image-only co-registration with the stereo DEM |
| `hirise_dems.csv`, `hirise_windows.csv`, `hirise_stereo_pairs.csv` | the 619 HiRISE DEMs, the predictions over them and the test pairs over them (Tables 3 and 4) |
| `pedr_field_matches.csv` | the image-to-orthoimage matches from which the PEDR field is fitted |
| `orbit_screen/<tile>.txt` | MOLA passes left out of the leveling of 216 equatorial tiles |
| `global_ctx_mosaic_fits.csv`, `global_ctx_mosaic_corrections.csv` | offsets between the two CTX reference mosaics, measured and applied |
| `gap_census.npz` | shapes of the real gaps of the leveled tiles (infill training) |
| `infill_tiles_split.tsv` | the 71 tiles used to train the infill networks, with their split |

Network weights are attached to the release of this repository:

| file | content | MD5 |
|---|---|---|
| `mesa_e088.pt` | DEM network, epoch 88 (21.8 M parameters). Release v1.0.0 attached this file as `mader_e088.pt`; `mesa.model.load_model` accepts either name and falls back to the other in the same directory | `2c027133a5f95346287015fe2e5b0142` |
| `infill_coarse.pt` | coarse infill network (108 m) | `a3630eceaaaaf5033ab7d7e0eb023e45` |
| `infill_fine.pt` | fine infill network (18 m) | `3b2d9c94557b07191633443dbf13ab68` |
| `infill_coarse_pretrained.pt` | coarse infill network trained on one tile, the start of the coarse training | `c0dbebeee199e457c3285abb7c7a2b64` |

## Quick start

Predict a DEM from one registered CTX image:

```bash
python -m mesa.predict --image B01_009956_2169_aligned.vrt --inc 45.45 --az 262.89 \
    --model mesa_e088.pt --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite \
    --out B01_009956_2169
```

The prediction is fitted to the MOLA shots through a 168 m FWHM Gaussian
footprint, also used by evaluation and leveling. Historical production
comparisons are documented in `verification/README.md`; changing this
configuration does not regenerate stored frames or scores.

Score the network on the test tiles (Table 1, tile rows):

```bash
python -m mesa.evaluate tiles --model mesa_e088.pt --root mars_ctx_dem_v4_pedr \
    --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \
    --pairs data/test_pairs.txt --out tiles.csv
```

The other tables are described in `docs/evaluation.md`. The footprint product
of a tile (dominant-image index, overlap depth, image table, footprint
polygons) is described in `docs/pipeline.md`, step 10.

## Verification

`verification/` holds the checks that compare this code with the code and
products behind the paper, on the same inputs (results in
`verification/README.md`).

## v3 chain

Release v3 of the global DEM (tag `ta_implicit_d1_spline100s1_e098v3`,
predicted in October 2026) was built with the scripts under `v3/`. They were
copied from the cluster without change. The source path and md5 of every file
are listed in `PROVENANCE_v3.md`, the settings of each stage in
`v3/README.md`, and the replay of one production box in `VERIFY_v3.md`. The
`mesa` package remains the readable form of the release-1 pipeline. In v3 the
model and the per-box stages after the infill were changed.

The stages, in order.

1. Prediction (`v3/predict/`). The network is the spline-decoder model of
   epoch 98 (`ta_impl_spline100_s1_fp32_e098`, config and checkpoint md5 in
   `v3/model/`). Its decoder upsamples with a fixed B-spline and a 3 x 3
   convolution in place of the transposed convolutions of epoch 88, which
   removes the 16/8/4/2 px lattice of release 1. PEDR shots are queried over
   all longitudes for frames that span 360 deg or reach within 2 deg of a
   pole. A frame whose confidence input has a mean below 0.05 is recorded as
   failed.
2. Leveling (`v3/downstream/stages/level_box_tag_v3.sbatch`). Frames are
   leveled and blended as in release 1, with feathered demotion of weak
   frames and the raster border counted as a data edge. Per-box frame weights
   are in `v3/downstream/weights/`.
3. CTX layer and infill (`ctx_fill_*_tag_v3.sbatch`). The release-1 networks
   and weights are used.
4. Join (`seamline_merge_v3.py`). All gaps are composited once, in an order
   independent way. Small gaps are shifted to meet the edge.
5. Infill rule v2.2, seam stage and relaxation (`mif_box_tag_v3.sh`). Infill
   that departs from MOLA by more than 30 m poleward of 60 deg (150 m
   elsewhere) is replaced by edge-matched MOLA. Hand-over lines between
   frames are solved in a 32 px band with a weak PEDR datum term and a guard.
   The half-offset at hard joints is removed over 32 px.
6. Seam pass 2 (`pass2_box_tag_v3.sh`). Lines still offset by 20 m or more
   are solved in a 64 px band.
7. QA (`qa_box_tag_v3.sh`). The box is compared with the release-1 tile by
   step pixels, straight-line cells, PEDR cells and shot MAE. The verdict is
   recorded and never enforced.
8. Polar comb correction (`v3/polar_combfix80/`), rows Np80 and N-90 only.
   The N-S comb inherited from the MOLA input near the poles is fitted and
   removed poleward of 80 deg (ramp to 81 deg). The output is a separate file
   beside the box surface, with a marker that names the source md5.
9. Reviewed local corrections (`v3/corrections/`). Height-only patches are
   reviewed site by site and published as immutable releases bound by SHA256
   (`MANIFEST.json`, pointer `LATEST.json`). They are resolved per box at
   render time with `resolve_global.py`.

Render rules of v3 (the render launcher is not in this release). Per box,
`box_dem_<TAG>_deseam_pcf80.tif` is read when it exists and its marker's
source md5 equals the md5 of the current `box_dem_<TAG>_deseam.tif`.
Otherwise `box_dem_<TAG>_deseam.tif` is read. A stale pcf80 file is refused
by the pre-check until it is re-staged. The pointer `LATEST.json` is copied
once at launch and passed to `resolve_global.py` for every box. One release
of corrections therefore applies to the whole render. Short bridges of 10 px
or less are filled from MOLA (mask class 2). Wider gaps with data on both
sides are filled from edge-matched MOLA up to 87 deg of latitude (class 3).
The north polar cap beyond the data is NoData. Blank-image components
(infill where the CTX mosaic has no image, census 048) without a usable
image and with a ridge of 5 m or more on their outline are NoData. Poleward
of 80 deg they are kept and flagged.

Error names used for v3, as in the paper (`docs/evaluation.md`), from the
most corrected to the least:

* labelMAE: the shape error only. The prediction is rescaled with the mean
  and standard deviation of the reference over the compared pixels, after
  co-registration.
* alignedMAE: the delivered surface with its level kept, after the
  horizontal shift that co-registers it with its reference. The shift is
  measured on the images, never on the DEMs. Offsets and tilts are not
  removed. (This error was called registeredMAE in September 2026 drafts.)
* deployMAE: the product unaltered in its own frame. Nothing is shifted or
  removed. MOLA is scored only this way.

Product classes and flags of v3 (provenance band and uncertainty layer):

* NoData: the north polar cap beyond the data, and the blank-image
  components under the rule above.
* Flag, seam 20 to 50 m and seam above 50 m: the seam-stage band around a
  hand-over line whose offset before the seam repair was in that range.
  Frame-infill band steps carry the same flag.
* Flag, polar: pixels poleward of 80 deg, where N-S streaking from the MOLA
  grid is possible. The comb correction reduces it on frames. A residual is
  possible on infill.
* Flag, corrected pixel: pixels replaced by a reviewed local correction
  (patch weight above 0). The calibrated error of the uncertainty layer is
  not valid there.

## License

MIT License (`LICENSE`). It covers the code and the network weights attached
to the release.
