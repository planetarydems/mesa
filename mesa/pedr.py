"""MOLA Precision Experiment Data Records (PEDR) and the shot-distance layer.

The PEDR shots are read from an SQLite database with one table, `mola`, with
columns LAT (deg), LON (deg east, 0-360), ELEV (m above the areoid) and ORBIT,
indexed on (LAT, LON); see docs/pipeline.md, step 0.
"""
import sqlite3
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from .geo import lonlat_to_xy, query_box


class PedrDatabase:
    """Read-only access to the PEDR shot table.

    full_lon_near_poles=True applies the rule of the prediction of version 1: a box whose
    longitude span is 360 deg or more, or that reaches within 2 deg of a pole,
    is queried over all longitudes of its latitude band. Without the rule such
    a box (a frame rotated near a pole spans more than 360 deg of longitude)
    folds onto one sector modulo 360, and the frame loses its shots. Shots that
    fall outside the frame are dropped by the callers."""

    def __init__(self, path, full_lon_near_poles=False):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.con = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
        self.full_lon_near_poles = bool(full_lon_near_poles)

    def query(self, lat_min, lat_max, lon_min, lon_max, elev_range=(-8500.0, 22000.0),
              exclude_orbits=None):
        """Shots in a latitude-longitude box -> (lat, lon, elev) arrays.

        Longitudes are taken modulo 360; lon_min > lon_max selects a box that
        wraps through 0. Shots outside elev_range are corrupt records and are
        dropped. exclude_orbits removes whole MOLA passes."""
        if self.full_lon_near_poles and (lon_max - lon_min >= 359.999 or lat_max >= 88.0 or lat_min <= -88.0):
            lon_min, lon_max = 0.0, 359.9999999
        lo, hi = lon_min % 360.0, lon_max % 360.0
        sql = ('SELECT LAT, LON, ELEV FROM mola WHERE LAT BETWEEN ? AND ? AND '
               + ('LON BETWEEN ? AND ?' if lo <= hi else '(LON >= ? OR LON <= ?)')
               + ' AND ELEV BETWEEN ? AND ?')
        params = [lat_min, lat_max, lo, hi, elev_range[0], elev_range[1]]
        if exclude_orbits:
            sql += f" AND ORBIT NOT IN ({','.join(str(int(o)) for o in sorted(exclude_orbits))})"
        rows = self.con.execute(sql, params).fetchall()
        if not rows:
            return np.empty(0), np.empty(0), np.empty(0)
        a = np.asarray(rows, dtype=np.float64)
        return a[:, 0], a[:, 1], a[:, 2]


def shot_distance(pedr, proj, transform, width, height, pixel_size=18.0, buffer_m=25000.0,
                  coarse_step_m=72.0, cap_m=60000.0):
    """Distance d1 (m) from each pixel centre of a grid to the nearest PEDR shot.

    Shots within 25 km of the grid are projected into the grid's projection and
    put in a k-d tree; d1 is evaluated every 72 m (4 pixels), capped at 60 km
    and interpolated bilinearly to every pixel. This reproduces the d1_shot
    layer of the training archive."""
    T = transform
    cols = np.array([0.0, width, 0.0, width])
    rows = np.array([0.0, 0.0, height, height])
    X = T.a * cols + T.b * rows + T.c
    Y = T.d * cols + T.e * rows + T.f
    box = query_box(X.min() - buffer_m, X.max() + buffer_m, Y.min() - buffer_m, Y.max() + buffer_m, proj)
    lat, lon, _ = pedr.query(*box)
    if lat.size == 0:
        return None
    xs, ys = lonlat_to_xy(lon, lat, proj)
    tree = cKDTree(np.column_stack([xs, ys]))

    stride = max(1, int(round(coarse_step_m / pixel_size)))
    cc, rr = np.meshgrid(np.arange(0, width, stride, dtype=np.float64) + 0.5,
                         np.arange(0, height, stride, dtype=np.float64) + 0.5)
    gx = T.a * cc + T.b * rr + T.c
    gy = T.d * cc + T.e * rr + T.f
    d, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    d = np.clip(d.reshape(cc.shape), 0.0, cap_m).astype(np.float32)
    full = zoom(d, (height / d.shape[0], width / d.shape[1]), order=1)
    if full.shape != (height, width):
        full = np.pad(full, ((0, max(0, height - full.shape[0])), (0, max(0, width - full.shape[1]))),
                      mode='edge')[:height, :width]
    return full


def shot_confidence(d1, d0=2000.0):
    """The network's shot-distance input, d_shot = exp(-d1 / d0)."""
    return np.exp(-np.clip(d1, 0.0, None) / float(d0)).astype(np.float32)


def screen_orbits(pedr, lat_min, lat_max, lon_min, lon_max, mola_path, threshold=25.0, min_shots=100):
    """MOLA passes to leave out of the levelling of a tile: orbits with at least
    min_shots shots in the box whose median difference from the gridded DEM
    exceeds threshold metres. The gridded DEM, built from the same shots, is the
    arbiter, so the screen does not depend on our predictions. Returns
    (flagged orbit numbers, [(orbit, n, median, robust std)])."""
    import rasterio
    from rasterio.windows import Window
    from scipy.ndimage import map_coordinates
    from .geo import parse_crs

    lo, hi = lon_min % 360.0, lon_max % 360.0
    sql = ('SELECT LAT, LON, ELEV, ORBIT FROM mola WHERE LAT BETWEEN ? AND ? AND '
           + ('LON BETWEEN ? AND ?' if lo <= hi else '(LON >= ? OR LON <= ?)'))
    a = np.asarray(pedr.con.execute(sql, (lat_min, lat_max, lo, hi)).fetchall(), np.float64)
    if not a.size:
        return [], []
    lat, lon, z, orbit = a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.int64)
    with rasterio.open(mola_path) as ds:
        T = ds.transform
        sx, sy = lonlat_to_xy(lon, lat, parse_crs(ds.crs))
        c0, c1 = max(int((sx.min() - T.c) / T.a) - 4, 0), min(int((sx.max() - T.c) / T.a) + 5, ds.width)
        r0, r1 = max(int((sy.max() - T.f) / T.e) - 4, 0), min(int((sy.min() - T.f) / T.e) + 5, ds.height)
        h = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
        h[h == ds.nodata] = np.nan
    # the gridded DEM's samples sit at the corners, not the centres, of its pixels: no half-pixel shift
    fc = (sx - T.c) / T.a - c0
    fr = (sy - T.f) / T.e - r0
    k = (fr > 1) & (fr < h.shape[0] - 2) & (fc > 1) & (fc < h.shape[1] - 2)
    dz = z[k] - map_coordinates(np.nan_to_num(h), np.vstack([fr[k], fc[k]]), order=1, mode='nearest')
    ok = np.isfinite(dz)
    dz, ob = dz[ok], orbit[k][ok]
    rows = []
    for o in np.unique(ob):
        v = dz[ob == o]
        if v.size >= min_shots:
            med = float(np.median(v))
            rows.append((int(o), int(v.size), med, float(1.4826 * np.median(np.abs(v - med)))))
    return [r[0] for r in rows if abs(r[2]) > threshold], rows
