"""Map projections on the Mars sphere and sampling of the MOLA gridded DEM.

Every raster in the pipeline uses one of three projections of the sphere
R = 3,396,190 m, written here explicitly rather than through PROJ:

    eqc    equirectangular   x = R dlon cos(lat_ts),  y = R lat
    sinu   sinusoidal        x = R dlon cos(lat),     y = R lat
    stere  polar stereographic, true scale at the pole (polar stereo pairs)

A projection is described by the tuple (kind, R, lon0, lat_ts); for stere the
last entry is +90 or -90 and names the pole.
"""
import math

import numpy as np
import rasterio
from rasterio.crs import CRS

R_MARS = 3396190.0
FULL_LON = (0.0, 359.999999)      # "every longitude" for a PEDR query (not 0..360, which wraps to 0..0)


def parse_crs(crs):
    """(kind, R, lon0, lat_ts) of a rasterio CRS, WKT or PROJ string."""
    d = CRS.from_user_input(crs).to_dict()
    kind = d.get('proj')
    R = float(d.get('R') or d.get('a') or R_MARS)
    lon0 = float(d.get('lon_0', 0.0))
    if kind == 'stere':
        lat0 = float(d.get('lat_0', 0.0))
        if abs(abs(lat0) - 90.0) > 1e-6:
            raise ValueError(f'only polar stereographic is supported: {d}')
        return 'stere', R, lon0, (90.0 if lat0 > 0 else -90.0)
    if kind not in ('eqc', 'sinu'):
        raise ValueError(f'unsupported projection: {d}')
    return kind, R, lon0, float(d.get('lat_ts', 0.0))


def lonlat_to_xy(lon, lat, proj):
    """Longitude and latitude (degrees, either convention) -> projected metres."""
    kind, R, lon0, lat_ts = proj
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if kind == 'stere':
        dl = np.radians(lon - lon0)
        if lat_ts > 0:
            rho = 2.0 * R * np.tan(math.pi / 4.0 - np.radians(lat) / 2.0)
            return rho * np.sin(dl), -rho * np.cos(dl)
        rho = 2.0 * R * np.tan(math.pi / 4.0 + np.radians(lat) / 2.0)
        return rho * np.sin(dl), rho * np.cos(dl)
    dlon = np.radians(((lon - lon0 + 180.0) % 360.0) - 180.0)
    lat_r = np.radians(lat)
    if kind == 'sinu':
        x = R * dlon * np.cos(lat_r)
    else:
        x = R * dlon * math.cos(math.radians(lat_ts))
    return x, R * lat_r


def xy_to_lonlat(x, y, proj):
    """Projected metres -> longitude and latitude in degrees (lon0 +- 180)."""
    kind, R, lon0, lat_ts = proj
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if kind == 'stere':
        c = 2.0 * np.arctan(np.hypot(x, y) / (2.0 * R))
        if lat_ts > 0:
            return lon0 + np.degrees(np.arctan2(x, -y)), 90.0 - np.degrees(c)
        return lon0 + np.degrees(np.arctan2(x, y)), -90.0 + np.degrees(c)
    lat_r = y / R
    if kind == 'sinu':
        c = np.cos(lat_r)
        c = np.where(np.abs(c) < 1e-9, 1e-9, c)
        return lon0 + np.degrees(x / (R * c)), np.degrees(lat_r)
    return lon0 + np.degrees(x / (R * math.cos(math.radians(lat_ts)))), np.degrees(lat_r)


def stere_query_box(xmin, xmax, ymin, ymax, R, lon0, pole):
    """(lat_min, lat_max, lon_min, lon_max) covering a rectangle in polar
    stereographic metres. The pole-nearest point of a rectangle is not a corner
    and the longitudes can wrap, so the boundary is sampled; lon_min > lon_max
    means the range wraps through 0/360."""
    dx = max(xmin, 0.0, -xmax)
    dy = max(ymin, 0.0, -ymax)
    rho_min = math.hypot(dx, dy)
    rho_max = max(math.hypot(cx, cy) for cx in (xmin, xmax) for cy in (ymin, ymax))
    c_min = 2.0 * math.atan(rho_min / (2.0 * R))
    c_max = 2.0 * math.atan(rho_max / (2.0 * R))
    if pole > 0:
        lat_min, lat_max = 90.0 - math.degrees(c_max), 90.0 - math.degrees(c_min)
    else:
        lat_min, lat_max = -90.0 + math.degrees(c_min), -90.0 + math.degrees(c_max)
    if xmin <= 0.0 <= xmax and ymin <= 0.0 <= ymax:
        return (lat_min, lat_max) + FULL_LON
    n = 25
    bx = np.concatenate([np.linspace(xmin, xmax, n), np.full(n, xmax), np.linspace(xmax, xmin, n), np.full(n, xmin)])
    by = np.concatenate([np.full(n, ymin), np.linspace(ymin, ymax, n), np.full(n, ymax), np.linspace(ymax, ymin, n)])
    lon, _ = xy_to_lonlat(bx, by, ('stere', R, lon0, pole))
    ang = np.sort(np.mod(lon, 360.0))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    k = int(np.argmax(gaps))
    if gaps[k] < 180.0:
        return (lat_min, lat_max) + FULL_LON
    lo, hi = ang[(k + 1) % ang.size], ang[k]
    return lat_min, lat_max, (lo - 0.05) % 360.0, (hi + 0.05) % 360.0


def query_box(xmin, xmax, ymin, ymax, proj):
    """Longitude-latitude box (lat_min, lat_max, lon_min, lon_max) of the
    corners of a projected rectangle (stereographic: of the whole rectangle)."""
    kind, R, lon0, lat_ts = proj
    if kind == 'stere':
        return stere_query_box(xmin, xmax, ymin, ymax, R, lon0, lat_ts)
    lon, lat = xy_to_lonlat(np.array([xmin, xmax, xmin, xmax]), np.array([ymin, ymin, ymax, ymax]), proj)
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def ground_scale(kind, lat_deg, lat_ts):
    """(kx, ky): ground metres per projected metre at latitude lat_deg."""
    if kind == 'sinu':
        return 1.0, 1.0
    if kind == 'stere':
        g = 0.5 * (1.0 + math.sin(math.radians(abs(lat_deg))))
        return g, g
    return math.cos(math.radians(lat_deg)) / math.cos(math.radians(lat_ts)), 1.0


def pixel_centres(transform, height, width, rows=None, cols=None):
    """Projected coordinates of pixel centres of a grid (or of given rows/cols)."""
    if rows is None:
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
    return x, y


class MolaSampler:
    """Bilinear sampling of the 463 m MOLA gridded DEM at arbitrary
    longitudes and latitudes. The mosaic is global in longitude, so samples
    wrap across its east and west edges."""

    def __init__(self, path):
        with rasterio.open(path) as ds:
            self.arr = ds.read(1)
            self.transform = ds.transform
            self.nodata = ds.nodata
            self.proj = parse_crs(ds.crs)
        self.H, self.W = self.arr.shape
        kind, R, _, lat_ts = self.proj
        span = abs(self.transform.a) * self.W
        self.periodic = (kind != 'sinu'
                         and abs(span - 2.0 * math.pi * R * math.cos(math.radians(lat_ts)))
                         <= abs(self.transform.a) * 1.01)

    def sample(self, lon, lat):
        x, y = lonlat_to_xy(lon, lat, self.proj)
        inv = ~self.transform
        col = inv.a * x + inv.b * y + inv.c
        row = inv.d * x + inv.e * y + inv.f
        if self.periodic:
            col = np.mod(col, self.W)
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        fc, fr = col - c0, row - r0
        if self.periodic:
            c0, c1 = np.mod(c0, self.W), np.mod(c0 + 1, self.W)
        else:
            c0 = np.clip(c0, 0, self.W - 2)
            c1 = c0 + 1
        r0 = np.clip(r0, 0, self.H - 2)
        a = self.arr
        top = a[r0, c0].astype(np.float64) * (1 - fc) + a[r0, c1].astype(np.float64) * fc
        bot = a[r0 + 1, c0].astype(np.float64) * (1 - fc) + a[r0 + 1, c1].astype(np.float64) * fc
        out = top * (1 - fr) + bot * fr
        outside = (row < 0) | (row > self.H - 1)
        if not self.periodic:
            outside |= (col < 0) | (col > self.W - 1)
        if outside.any():
            raise ValueError(f'{int(outside.sum())} MOLA samples fall outside the mosaic')
        if self.nodata is not None and np.any(out == self.nodata):
            raise ValueError('MOLA sample hit no-data')
        return out

    def on_grid(self, transform, height, width, proj):
        """MOLA at the pixel centres of a grid in projection proj (float32)."""
        x, y = pixel_centres(transform, height, width)
        lon, lat = xy_to_lonlat(x, y, proj)
        return self.sample(lon.ravel(), lat.ravel()).reshape(height, width).astype(np.float32)
