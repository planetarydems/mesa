# Verification

Each script runs this code and the code that produced the paper and the mosaic
on the same inputs, on the cluster where the products were made, and compares
the outputs. "Bitwise" means identical arrays of floating-point numbers.
`jobs/` holds the Slurm scripts with the paths and environments used
(PyTorch 2.0 on V100 GPUs and Xeon CPUs; the registration in the environment of
the original registration, OpenCV 4.7).

| step | script | what was compared | result |
|---|---|---|---|
| network | `check_model.py` | outputs on random inputs and on 8 test samples, CPU and GPU | bitwise |
| training samples | `check_data.py` | 64 validation and 64 augmented training samples | bitwise |
| training | `check_train_steps.py` | loss of each of 4 optimiser steps (CPU) and every weight after them | bitwise |
| prediction | `check_predict.py` | 3 whole images (GPU and CPU): DEM, rescaled DEM, image and MOLA grids | bitwise |
| registration | `check_register.py` | 10 images of an equatorial and a polar tile, both stages | within 5 µm (similarity), 1e-17 (refinement) |
| PEDR field | `check_field.py` | Global CTX Mosaic corrections, 379 tile vectors, 300 shifted images | identical |
| levelling | `check_tile_chain.py level` | offsets, levelled DEMs, weights and levelled mosaic of 2 tiles (57 and 138 DEMs) | bitwise |
| infill | `check_tile_chain.py infill` | infill surface and provenance of 2 tiles | provenance bitwise; surface within 1.5 mm |
| join and de-seam | `check_tile_chain.py seams` | joined tile, seam mask, dominant-DEM index, de-seamed tile | bitwise |
| global mosaic | `check_mosaic.py` | cross-tile levelling field and rendered DEM and mask of 4 tiles | bitwise |
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
