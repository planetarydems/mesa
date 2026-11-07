#!/usr/bin/env bash
# One product tile, from the predicted DEMs to the de-seamed tile (Sections 2.5-2.6).
#
#   run_tile.sh TILE PREDS_DIR OUT_DIR
#
# Environment: MOLA (gridded DEM), PEDR (SQLite), GLOBAL_CTX (directory of Global CTX Mosaic tiles:
# the 5 m tiles for tiles within 30 deg of the equator, the same tiles averaged to 18 m with
# `gdal_translate -tr 18 18 -r average` poleward), INFILL_COARSE, INFILL_FINE (network weights),
# DATA (this repository's data/ directory), JOBS.
set -euo pipefail
TILE=$1 PREDS=$2 OUT=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

# 1. levelling and feather blend -> leveled.tif, leveled/, leveled_offsets.csv
SCREEN="$DATA/orbit_screen/$TILE.txt"
python3 -m mesa.level --tile "$TILE" --frames "$PREDS/*/pred_shot_anchor.tif" --mola "$MOLA" --pedr "$PEDR" \
    ${JOBS:+--jobs $JOBS} $( [ -s "$SCREEN" ] && echo --exclude-orbits "$SCREEN" ) --out "$OUT"

# 2. CTX layer of the infill: the Global CTX Mosaic averaged onto the levelled grid (byte, no-data 0)
read -r L B R T W H < <(python3 -c "
import rasterio; s = rasterio.open('$OUT/leveled.tif'); b = s.bounds
print(b.left, b.bottom, b.right, b.top, s.width, s.height)")
python3 -c "import rasterio; print(rasterio.open('$OUT/leveled.tif').crs.to_wkt())" > "$OUT/.tile_crs.wkt"
ls "$GLOBAL_CTX"/MurrayLab_CTX_V01_*_Mosaic.tif > "$OUT/.global_ctx_list.txt"
gdalbuildvrt -srcnodata 0 -vrtnodata 0 -input_file_list "$OUT/.global_ctx_list.txt" "$OUT/.global_ctx.vrt"
gdalwarp -t_srs "$OUT/.tile_crs.wkt" -te "$L" "$B" "$R" "$T" -ts "$W" "$H" -r average -srcnodata 0 -dstnodata 0 \
    -ot Byte -overwrite -multi -wo NUM_THREADS=8 -co TILED=YES -co COMPRESS=DEFLATE -co BIGTIFF=YES \
    "$OUT/.global_ctx.vrt" "$OUT/ctx_18m.tif"

# 3. infill (GPU if available) -> infill.model.tif, infill.prov.tif
python3 -m mesa.infill --dem "$OUT/leveled.tif" --ctx "$OUT/ctx_18m.tif" --mola "$MOLA" \
    --coarse "$INFILL_COARSE" --fine "$INFILL_FINE" --out "$OUT/infill"

# 4. join along a minimum-cost boundary -> final.tif; repair butt-joint steps -> deseam.tif
python3 -m mesa.seams merge --dem "$OUT/leveled.tif" --model "$OUT/infill.model.tif" --out "$OUT/final.tif"
python3 -m mesa.seams deseam --final "$OUT/final.tif" --prov "$OUT/infill.prov.tif" \
    --frames "$OUT/leveled/*_leveled.tif" --out "$OUT/deseam.tif" ${JOBS:+--jobs $JOBS}
