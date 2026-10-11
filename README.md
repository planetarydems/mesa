# MADER

Code for *MADER, an 18 m digital elevation model of Mars from single CTX
images* (Mitchell et al., in review): the network, its training, the prediction
of a DEM from one registered CTX image, the assembly of the global mosaic
(levelling against MOLA laser shots, gap infill, seam repair, cross-tile
levelling) and the per-pixel uncertainty layer, together with the evaluation
used in the paper.

The released model is the network at epoch 88 (`mader_e088.pt`); the global
mosaic was built with it.

## Layout

```
mader/
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
  level.py           levelling and blending of the DEMs of a product tile
  infill.py          gap infill: networks and inference (Sec. 2.6)
  infill_train.py    gap census, training windows, infill training
  seams.py           join of the infill, repair of butt-joint steps
  mosaic.py          cross-tile levelling, rendering on the global grid, COG
  uncertainty.py     calibration and per-pixel uncertainty layer
  evaluate.py        test tiles: Table 1 (tile rows) and Table 2
  evaluate_images.py whole CTX images against their stereo DEMs: Table 1 (image rows)
  evaluate_hirise.py HiRISE DEMs: Tables 3 and 4
scripts/             shell drivers (reference crops, one product tile)
data/                small inputs the pipeline needs (see below)
docs/pipeline.md     every step from the inputs to the mosaic, with the settings used
docs/evaluation.md   the commands behind each table of the paper
verification/        checks that this code reproduces the results of the paper and the mosaic
```

## Installation

```bash
conda create -n mader python=3.11 numpy scipy pytorch rasterio gdal opencv pyproj lightning -c conda-forge
pip install -e .
```

GDAL command-line tools are needed for the reference crops, the CTX layer of the
infill and the COG. ISIS and the Ames Stereo Pipeline are needed only to map
project new CTX images (docs/pipeline.md, step 1). The results of the paper
were computed with PyTorch 2.0, rasterio 1.4.3 and GDAL 3.9.3 (3.6.2 for the
HiRISE scoring and the uncertainty calibration); other GDAL versions resample
slightly differently (docs/evaluation.md).

## Data

External inputs (MOLA gridded DEM, PEDR shots, CTX images and mosaics, the
training archive) are listed in `docs/pipeline.md`. The `data/` directory holds:

| file | content |
|---|---|
| `images.csv` | the 53,722 image placements of the mosaic: tile, image, solar incidence and azimuth |
| `test_pairs.txt` | the 359 test stereo pairs |
| `test_manifest_{eq,mid,polar}.csv` | the test samples (tile, image) of the 375 pairs from which the test crops are drawn |
| `test_images.csv` | for each test pair, the CTX image scored whole, its product tile, latitude, and its image-only co-registration with the stereo DEM |
| `hirise_dems.csv`, `hirise_windows.csv`, `hirise_stereo_pairs.csv` | the 619 HiRISE DEMs, the predictions over them and the test pairs over them (Tables 3 and 4) |
| `pedr_field_matches.csv` | the image-to-orthoimage matches from which the PEDR field is fitted |
| `orbit_screen/<tile>.txt` | MOLA passes left out of the levelling of 216 equatorial tiles |
| `global_ctx_mosaic_fits.csv`, `global_ctx_mosaic_corrections.csv` | offsets between the two CTX reference mosaics, measured and applied |
| `gap_census.npz` | shapes of the real gaps of the levelled tiles (infill training) |
| `infill_tiles_split.tsv` | the 71 tiles used to train the infill networks, with their split |

Network weights are attached to the release of this repository:

| file | content | MD5 |
|---|---|---|
| `mader_e088.pt` | DEM network, epoch 88 (21.8 M parameters) | `2c027133a5f95346287015fe2e5b0142` |
| `infill_coarse.pt` | coarse infill network (108 m) | `a3630eceaaaaf5033ab7d7e0eb023e45` |
| `infill_fine.pt` | fine infill network (18 m) | `3b2d9c94557b07191633443dbf13ab68` |
| `infill_coarse_pretrained.pt` | coarse infill network trained on one tile, the start of the coarse training | `c0dbebeee199e457c3285abb7c7a2b64` |

## Quick start

Predict a DEM from one registered CTX image:

```bash
python -m mader.predict --image B01_009956_2169_aligned.vrt --inc 45.45 --az 262.89 \
    --model mader_e088.pt --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite \
    --out B01_009956_2169
```

Score the network on the test tiles (Table 1, tile rows):

```bash
python -m mader.evaluate tiles --model mader_e088.pt --root mars_ctx_dem_v4_pedr \
    --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \
    --pairs data/test_pairs.txt --out tiles.csv
```

The other tables are described in `docs/evaluation.md`.

## Verification

`verification/` holds the checks that compare this code with the code and
products behind the paper, on the same inputs (results in
`verification/README.md`).

## License

MIT License (`LICENSE`). It covers the code and the network weights attached
to the release.
