#!/usr/bin/env bash
# Reference image of a product tile for mesa.register (Text S8).
#
#   reference_crops.sh TILE OUT_DIR CONTROLLED_QUADS_DIR GLOBAL_CTX_DIR CORRECTIONS_CSV
#
# Within 30 deg of the equator: the controlled CTX mosaic (Robbins et al., 2023; quadrangles
# MCxx_6mpp.16bit.tif, signed 16-bit, no-data -32768) resampled to 24 m on the global
# equirectangular grid over the tile plus 0.3 deg   -> OUT_DIR/reference_24m.tif
# Poleward: the Global CTX Mosaic (Dickson et al., 2024; 4 x 4 deg tiles, byte, no-data 0)
# averaged to 24 m in the tile's sinusoidal projection over the tile plus 0.3 deg, and a
# copy whose origin is moved by the tile's correction onto the controlled mosaic
#                                                    -> OUT_DIR/reference_24m_corrected.vrt
set -euo pipefail
TILE=$1 OUT=$2 QUADS=$3 GCTX=$4 CORR=$5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
read -r LON LAT < <(python3 -c "
import re,sys; m=re.match(r'E([p-])(\d{3})_N([p-])(\d{2})$', '$TILE')
s=lambda c: -1 if c=='-' else 1; print(s(m.group(1))*int(m.group(2)), s(m.group(3))*int(m.group(4)))")
if [ "$LAT" -ge -30 ] && [ "$LAT" -le 20 ]; then
  # quadrangles touched by the tile (45 deg columns, MC08-15 north / MC16-23 south of the equator)
  read -r QLIST TE < <(python3 - "$LON" "$LAT" "$QUADS" <<'PY'
import math, sys
lon, lat, d = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cols, e = set(), lon
while e <= lon + 10 + 1e-9:
    cols.add(int(((e - 180.0) % 360.0) // 45.0) % 8); e += 2.5
base = 8 if lat + 5 >= 0 else 16
R, B = 3396190.0, 0.3
print(",".join(f"{d}/MC{base + x:02d}_6mpp.16bit.tif" for x in sorted(cols)),
      " ".join(f"{R * math.radians(v):.1f}" for v in (lon - B, lat - B, lon + 10 + B, lat + 10 + B)))
PY
)
  IFS=',' read -ra Q <<< "$QLIST"
  gdalwarp -q -overwrite -t_srs '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs' \
    -te $TE -tr 24 24 -r bilinear -ot Int16 -srcnodata -32768 -dstnodata -32768 -multi -wo NUM_THREADS=4 \
    -co TILED=YES -co COMPRESS=NONE -co BIGTIFF=IF_SAFER --config GDAL_CACHEMAX 2048 "${Q[@]}" "$OUT/reference_24m.tif"
else
  CLON=$((LON + 5))
  TSRS="+proj=sinu +lon_0=$CLON +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs"
  read -r BBOX TILES < <(python3 - "$LON" "$LAT" "$CLON" "$GCTX" <<'PY'
import math, os, sys
lon0, lat0, clon, d = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
R, m = 3396190.0, 0.3
lon1, lat1 = lon0 + 10 + m, min(90.0, lat0 + 10 + m)
lon0, lat0 = lon0 - m, max(-90.0, lat0 - m)
sx = lambda lon, lat: R * math.radians(((lon - clon + 180.0) % 360.0) - 180.0) * math.cos(math.radians(lat))
lats = [lat0, lat1] + ([0.0] if lat0 < 0.0 < lat1 else [])
f4 = lambda v: int(math.floor(v / 4.0) * 4)
name = lambda lo, la: (f"E-{abs(lo):03d}" if lo < 0 else f"E{lo:03d}") + "_" + (f"N-{abs(la):02d}" if la < 0 else f"N{la:02d}")
tiles = [f"{d}/MurrayLab_CTX_V01_{name(((tl + 180) % 360) - 180, tb)}_Mosaic.tif"
         for tl in range(f4(lon0), f4(lon1 - 1e-9) + 4, 4)
         for tb in range(f4(lat0), min(84, f4(lat1 - 1e-9)) + 4, 4) if tb >= -88]
tiles = [t for t in tiles if os.path.exists(t)]
print(",".join(f"{v:.1f}" for v in (min(sx(lon0, la) for la in lats), R * math.radians(lat0),
                                     max(sx(lon1, la) for la in lats), R * math.radians(lat1))), ",".join(tiles))
PY
)
  IFS=',' read -ra T <<< "$TILES"; IFS=',' read -r XMIN YMIN XMAX YMAX <<< "$BBOX"
  gdalwarp -q -overwrite -t_srs "$TSRS" -te "$XMIN" "$YMIN" "$XMAX" "$YMAX" -tr 24 24 -r average -ot Byte \
    -srcnodata 0 -dstnodata 0 -multi -wo NUM_THREADS=8 -co TILED=YES -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER \
    "${T[@]}" "$OUT/reference_24m.tif"
  PYTHONPATH="$ROOT" python3 - "$OUT/reference_24m.tif" "$OUT/reference_24m_corrected.vrt" "$TILE" "$CORR" <<'PY'
import csv, sys
import rasterio
from rasterio.transform import Affine
from mesa.register import write_vrt
ref, vrt, tile, table = sys.argv[1:5]
row = {r['tile']: r for r in csv.DictReader(open(table))}[tile]
with rasterio.open(ref) as r:
    T = r.transform
write_vrt(ref, vrt, Affine(T.a, T.b, T.c + float(row['dx_m']), T.d, T.e, T.f + float(row['dy_m'])))
PY
fi
