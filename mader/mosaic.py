"""Global mosaic: cross-tile levelling, rendering on the global grid, COG.

The 648 de-seamed product tiles (10 x 10 deg, each on its own grid) are joined
on one global equirectangular grid of 1,185,494 x 592,747 pixels (lon -180 to
180, 18.0 m).

coarse   each tile averaged onto a global lattice of 5/1029 deg (about 288 m),
         in an equirectangular projection centred on the tile, so that no tile
         is ever seen across the antimeridian.
level    a smooth additive correction per tile that brings it onto the median
         of all tiles where they overlap, within 4 km of its cell edge:
             c = argmin sum LAM a (c - D)^2 + |grad c|^2 + EPS a c^2
         on nodes of 7 x 7 lattice cells (about 2 km), a = cos(lat), where D is
         the node median of (median of all tiles - this tile), smoothed over
         6 km; c follows D at the edges and relaxes to 0 inside the cell.
render   each 10 x 10 deg cell of the global grid: every tile reaching the cell
         is warped onto it (bilinear), its correction added, and the tiles
         blended linearly over +-2 km of the cell edges. Gaps left between
         tiles are bridged by interpolating the difference from MOLA.
cog      the rendered cells assembled into one Cloud Optimized GeoTIFF.

    python -m mader.mosaic coarse --tiles-root tiles --work global E-030_N-10 ...
    python -m mader.mosaic level  --tiles-root tiles --work global E-030_N-10 ...
    python -m mader.mosaic render --tiles-root tiles --work global --mola MOLA.tif E-030_N-10 ...
    python -m mader.mosaic cog    --work global --out mader_dem.tif

The tile layout is <tiles-root>/<tile>/deseam.tif and <tiles-root>/<tile>/infill.prov.tif
(--dem / --prov change the file names).
"""
import argparse
import json
import math
import os
import re
import subprocess
import time
import warnings

import numpy as np
import rasterio
import scipy.sparse as sp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import spsolve

R = 3396190.0
DEG = R * math.pi / 180.0
ND = -9999.0
W_GLOBAL, H_GLOBAL = 1185494, 592747
PDEG = 360.0 / W_GLOBAL                  # fine pixel in degrees
PM = PDEG * DEG                          # fine pixel in metres (17.99999925 m)
CS_DEG = 5.0 / 1029                      # coarse lattice step: every 5 deg is a whole number of cells
NC, NRC = 72 * 1029, 36 * 1029
NODE = 7
NODE_DEG = CS_DEG * NODE
NODE_KM = NODE_DEG * DEG / 1000.0
FEATHER_KM = 4.0
BAND_KM = 4.0
LAM, EPS_KM, SMOOTH_KM, DMAX = 4.0, 150.0, 6.0, 15.0
EQC0 = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs'


def local_proj4(lonc):
    return f'+proj=eqc +lat_ts=0 +lat_0=0 +lon_0={lonc:g} +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs'


TILE_RE = re.compile(r'^E([p-])(\d{3})_N([p-])(\d{2})$')


def cell(t):
    """(lon_min, lon_max, lat_min, lat_max) of a tile name such as E-030_N-10."""
    m = TILE_RE.match(t)
    lo = (1 if m.group(1) == 'p' else -1) * int(m.group(2))
    la = (1 if m.group(3) == 'p' else -1) * int(m.group(4))
    return lo, lo + 10, la, la + 10


def tile_name(lo, la):
    return f"E{'p' if lo >= 0 else '-'}{abs(lo):03d}_N{'p' if la >= 0 else '-'}{abs(la):02d}"


def centre_lon(t):
    lo0, lo1, _, _ = cell(t)
    return 0.5 * (lo0 + lo1)


def wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def neighbours(t, tiles):
    """Tiles that can reach t's cell. Polar sinusoidal tiles overhang far in
    longitude near the poles, so two columns are searched there."""
    lo0, _, la0, la1 = cell(t)
    reach = lambda a, b: 2 if max(abs(a), abs(b)) >= 80 else 1
    out = []
    for dla in (-10, 0, 10):
        la = la0 + dla
        if la < -90 or la > 80:
            continue
        k = max(reach(la0, la1), reach(la, la + 10))
        for dlo in range(-10 * k, 10 * k + 1, 10):
            n = tile_name((lo0 + dlo + 180) % 360 - 180, la)
            if n != t and n in tiles:
                out.append(n)
    return out


class Layout:
    """Where the inputs and the work products of the mosaic live."""

    def __init__(self, tiles_root, work, dem='deseam.tif', prov='infill.prov.tif'):
        self.root, self.work, self.dem_name, self.prov_name = tiles_root, work, dem, prov
        for d in ('vrt', 'coarse', 'corr', 'tiles'):
            os.makedirs(os.path.join(work, d), exist_ok=True)

    def dem(self, t):
        return os.path.join(self.root, t, self.dem_name.format(tile=t))

    def prov(self, t):
        return os.path.join(self.root, t, self.prov_name.format(tile=t))

    def tiles(self):
        return sorted(t for t in os.listdir(self.root) if TILE_RE.match(t) and os.path.isfile(self.dem(t)))

    def local_vrt(self, t, kind='dem'):
        """VRT that presents a tile's raster in an equirectangular projection
        centred on the tile (an exact relabelling: eqc is linear in longitude).
        Polar tiles are already sinusoidal about their centre and keep their
        projection."""
        path = self.dem(t) if kind == 'dem' else self.prov(t)
        with rasterio.open(path) as s:
            p4, tr, w, h, nd, dt = s.crs.to_proj4(), s.transform, s.width, s.height, s.nodata, s.dtypes[0]
            bx, by = s.block_shapes[0][1], s.block_shapes[0][0]
        proj = re.search(r'\+proj=(\w+)', p4).group(1)
        l0 = float(re.search(r'\+lon_0=(-?[0-9.e]+)', p4).group(1)) if '+lon_0=' in p4 else 0.0
        lc = centre_lon(t)
        if proj == 'eqc':
            assert abs(l0) < 1e-9, (t, p4)
            srs, x0 = local_proj4(lc), tr.c - lc * DEG
        elif proj == 'sinu':
            assert abs(l0 - lc) < 1e-6, (t, p4)
            srs, x0 = p4, tr.c
        else:
            raise ValueError(f'{t}: unexpected projection {p4}')
        gdt = {'float32': 'Float32', 'uint8': 'Byte'}[dt]
        ndx = f'<NoDataValue>{nd:g}</NoDataValue>' if nd is not None else ''
        xml = (f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}"><SRS>{srs}</SRS>'
               f'<GeoTransform>{x0!r}, {tr.a!r}, 0, {tr.f!r}, 0, {tr.e!r}</GeoTransform>'
               f'<VRTRasterBand dataType="{gdt}" band="1">{ndx}<SimpleSource>'
               f'<SourceFilename relativeToVRT="0">{os.path.abspath(path)}</SourceFilename><SourceBand>1</SourceBand>'
               f'<SourceProperties RasterXSize="{w}" RasterYSize="{h}" DataType="{gdt}" BlockXSize="{bx}" '
               f'BlockYSize="{by}"/><SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
               f'<DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/></SimpleSource></VRTRasterBand></VRTDataset>')
        out = os.path.join(self.work, 'vrt', f'{t}_{kind}.vrt')
        tmp = f'{out}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            f.write(xml)
        os.replace(tmp, out)
        return out


# ---------------------------------------------------------------------------
# coarse
# ---------------------------------------------------------------------------

def coarse_window(t):
    """Node-aligned window of the coarse lattice covering a tile's cell with a
    margin of 0.1 deg (in longitude scaled by 1/cos lat): global (col0, col1,
    row0, row1); columns are not wrapped and may lie outside 0..NC."""
    lo0, lo1, la0, la1 = cell(t)
    mlat = 0.1
    php = min(max(abs(la0), abs(la1)), 89.5)
    mlon = min(0.1 / math.cos(math.radians(php)), 15.0)
    j0 = math.floor((lo0 - mlon + 180.0) / NODE_DEG) * NODE
    j1 = math.ceil((lo1 + mlon + 180.0) / NODE_DEG) * NODE
    i0 = max(0, math.floor((90.0 - min(90.0, la1 + mlat)) / NODE_DEG) * NODE)
    i1 = min(NRC, math.ceil((90.0 - max(-90.0, la0 - mlat)) / NODE_DEG) * NODE)
    return j0, j1, i0, i1


def coarse(t, layout, threads=4):
    j0, j1, i0, i1 = coarse_window(t)
    lc = centre_lon(t)
    crs = CRS.from_proj4(local_proj4(lc))
    tr = from_origin((-180.0 + j0 * CS_DEG - lc) * DEG, (90.0 - i0 * CS_DEG) * DEG, CS_DEG * DEG, CS_DEG * DEG)
    z = np.full((i1 - i0, j1 - j0), np.nan, np.float32)
    with rasterio.open(layout.local_vrt(t)) as S:
        reproject(rasterio.band(S, 1), z, dst_transform=tr, dst_crs=crs, src_nodata=S.nodata, dst_nodata=np.nan,
                  resampling=Resampling.average, num_threads=threads, warp_mem_limit=2048)
    z[(z == ND) | (np.abs(z) > 3e4)] = np.nan
    profile = dict(driver='GTiff', dtype='float32', nodata=np.nan, width=z.shape[1], height=z.shape[0], count=1,
                   crs=crs, transform=tr, compress='deflate', predictor=3, tiled=True, blockxsize=256, blockysize=256)
    out = os.path.join(layout.work, 'coarse', f'{t}.tif')
    with rasterio.open(out + '.partial.tif', 'w', **profile) as d:
        d.write(z, 1)
        d.update_tags(tile=t, j0=j0, i0=i0, lonc=lc)
    os.replace(out + '.partial.tif', out)


# ---------------------------------------------------------------------------
# level
# ---------------------------------------------------------------------------

def _load_coarse(t, coarse_dir):
    with rasterio.open(os.path.join(coarse_dir, f'{t}.tif')) as s:
        tags = s.tags()
        return s.read(1), int(tags['j0']), int(tags['i0'])


def _smooth_nodes(Dn, lat_n):
    """Masked box mean over about SMOOTH_KM (ground) along both axes."""
    m = np.isfinite(Dn)
    v = np.where(m, Dn, 0.0)
    f = m.astype(np.float64)
    hy = max(1, int(round(SMOOTH_KM / 2 / NODE_KM)))
    cr = np.maximum(np.cos(np.radians(lat_n)), 0.02)
    hx = np.clip(np.round(SMOOTH_KM / 2 / (NODE_KM * cr)).astype(int), 1, 400)

    def xsum(a):
        cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
        out = np.empty_like(a)
        n = a.shape[1]
        for r in range(a.shape[0]):
            lo = np.clip(np.arange(n) - hx[r], 0, n)
            hi = np.clip(np.arange(n) + hx[r] + 1, 0, n)
            out[r] = cs[r, hi] - cs[r, lo]
        return out

    def ysum(a):
        cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)], axis=0)
        n = a.shape[0]
        lo = np.clip(np.arange(n) - hy, 0, n)
        hi = np.clip(np.arange(n) + hy + 1, 0, n)
        return cs[hi] - cs[lo]

    num, den = ysum(xsum(v)), ysum(xsum(f))
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    out[~m] = np.nan
    return out


def _solve_field(Dsm, lat_n):
    """Sparse solve of the correction field on the node grid (ground metric:
    x edges weighted 1/cos lat, y edges cos lat)."""
    hn, wn = Dsm.shape
    n = hn * wn
    idx = np.arange(n).reshape(hn, wn)
    cr = np.maximum(np.cos(np.radians(lat_n)), 0.02)
    cy = np.maximum(np.cos(np.radians(0.5 * (lat_n[:-1] + lat_n[1:]))), 0.02)
    kx = np.repeat((1.0 / cr)[:, None], wn - 1, axis=1)
    ky = np.repeat(cy[:, None], wn, axis=1)
    ia = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    ib = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    kw = np.r_[kx.ravel(), ky.ravel()]
    A = sp.coo_matrix((np.r_[-kw, -kw], (np.r_[ia, ib], np.r_[ib, ia])), shape=(n, n)).tocsr()
    deg = np.bincount(ia, kw, n) + np.bincount(ib, kw, n)
    m = np.isfinite(Dsm)
    area = np.repeat(cr[:, None], wn, axis=1)
    eps = (NODE_KM / EPS_KM) ** 2
    wd = np.where(m, LAM * area, 0.0)
    A = A + sp.diags(deg + (wd + eps * area).ravel())
    rhs = (wd * np.where(m, Dsm, 0.0)).ravel()
    return spsolve(A.tocsc(), rhs).reshape(hn, wn).astype(np.float32)


def level(t, layout, tiles, coarse_dir=None):
    """Correction field of tile t -> corr/<t>.tif (band 1 correction, band 2
    node differences, band 3 smoothed differences) and corr/<t>.json."""
    coarse_dir = coarse_dir or os.path.join(layout.work, 'coarse')
    zb, j0, i0 = _load_coarse(t, coarse_dir)
    h, w = zb.shape
    jj, ii = j0 + np.arange(w), i0 + np.arange(h)
    layers, used = [zb], []
    for k in neighbours(t, tiles):
        if not os.path.exists(os.path.join(coarse_dir, f'{k}.tif')):
            continue
        zk, j0k, i0k = _load_coarse(k, coarse_dir)
        hk, wk = zk.shape
        ck = (jj - j0k) % NC
        okc = ck < wk
        rk = ii - i0k
        okr = (rk >= 0) & (rk < hk)
        if not okc.any() or not okr.any():
            continue
        L = np.full((h, w), np.nan, np.float32)
        L[np.ix_(okr, okc)] = zk[np.ix_(rk[okr], ck[okc])]
        if np.isfinite(L).any():
            layers.append(L)
            used.append(k)
    Z = np.stack(layers)
    warnings.filterwarnings('ignore', category=RuntimeWarning)    # all-NaN medians are expected

    lo0, lo1, la0, la1 = cell(t)
    lc = centre_lon(t)
    lat = 90.0 - (i0 + np.arange(h) + 0.5) * CS_DEG
    lon_rel = (-180.0 + (j0 + np.arange(w) + 0.5) * CS_DEG) - lc
    cnt = np.isfinite(Z).sum(0)
    M = np.nanmedian(Z, axis=0) if Z.shape[0] > 1 else np.full((h, w), np.nan, np.float32)
    D = np.where((cnt >= 2) & np.isfinite(Z[0]), M - Z[0], np.nan)
    dx = (5.0 - np.abs(lon_rel))[None, :] * DEG / 1000.0 * np.cos(np.radians(lat))[:, None]
    dy = np.minimum(lat - la0, la1 - lat)[:, None] * DEG / 1000.0
    D[~(np.abs(np.minimum(dx, dy)) <= BAND_KM)] = np.nan
    hn, wn = h // NODE, w // NODE
    blk = D[:hn * NODE, :wn * NODE].reshape(hn, NODE, wn, NODE).transpose(0, 2, 1, 3).reshape(hn, wn, NODE * NODE)
    nv = np.isfinite(blk).sum(2)
    Dn = np.where(nv >= 8, np.nanmedian(blk, axis=2), np.nan).astype(np.float32)
    n_gross = int((np.abs(Dn) > DMAX).sum())
    Dn[np.abs(Dn) > DMAX] = np.nan       # a content disagreement, not a datum offset
    lat_n = 90.0 - (i0 / NODE + np.arange(hn) + 0.5) * NODE_DEG
    Dsm = _smooth_nodes(Dn, lat_n)
    c = _solve_field(Dsm, lat_n) if np.isfinite(Dsm).any() else np.zeros((hn, wn), np.float32)

    crs = CRS.from_proj4(local_proj4(lc))
    tr = from_origin((-180.0 + j0 * CS_DEG - lc) * DEG, (90.0 - i0 * CS_DEG) * DEG, NODE_DEG * DEG, NODE_DEG * DEG)
    out = os.path.join(layout.work, 'corr', f'{t}.tif')
    with rasterio.open(out + '.partial.tif', 'w', driver='GTiff', dtype='float32', count=3, width=wn, height=hn,
                       crs=crs, transform=tr, nodata=np.nan, compress='deflate') as d:
        d.write(np.stack([c, Dn, Dsm.astype(np.float32)]))
        d.update_tags(tile=t, j0n=j0 // NODE, i0n=i0 // NODE, lonc=lc, bands='corr,D_node,D_smooth')
    os.replace(out + '.partial.tif', out)
    m = np.isfinite(Dsm)
    stats = dict(tile=t, neighbours_used=used, data_nodes=int(m.sum()), gross_nodes_rejected=n_gross,
                 D_abs_med=float(np.nanmedian(np.abs(Dn))) if m.any() else None,
                 c_min=float(c.min()), c_max=float(c.max()))
    json.dump(stats, open(os.path.join(layout.work, 'corr', f'{t}.json'), 'w'), indent=1)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

TAPER_PX = int(round(1000.0 / PM))      # a tile's weight falls to 0 over 1 km toward its own data edge
SEAM_PX = int(round(1000.0 / PM))       # half-width of the band bridged across a hard seam
GAP_EW_KM, GAP_NS_KM = 16.0, 5.0         # longest east-west and north-south gaps bridged
APRON = 320                              # rows/columns rendered around each block
APRON_ANTIMERIDIAN = 1000
SMOOTH_BRIDGE_KM = 2.0
BLOCK_ROWS = 1024


def feather_weight(lon, lat, t, F=FEATHER_KM):
    """Blend weight of tile t: 0 at F/2 outside its cell, 1 at F/2 inside
    (ground distance), on the 1-D longitudes (columns) and latitudes (rows)."""
    lo0, lo1, la0, la1 = cell(t)
    rel = wrap(np.asarray(lon) - 0.5 * (lo0 + lo1))
    dx = (5.0 - np.abs(rel))[None, :] * DEG / 1000.0 * np.cos(np.radians(np.asarray(lat)))[:, None]
    dy = np.minimum(np.asarray(lat) - la0, la1 - np.asarray(lat))[:, None] * DEG / 1000.0
    return np.clip((np.minimum(dx, dy) + F / 2.0) / F, 0.0, 1.0).astype(np.float32)


class MolaBridge:
    """Bilinear MOLA for the gap bridges, periodic over exactly 360 deg of
    longitude, reading only the rows a block needs."""

    def __init__(self, path):
        self.S = rasterio.open(path)
        t = self.S.transform
        self.left, self.top, self.res = t.c, t.f, t.a
        self.w, self.h, self.nd = self.S.width, self.S.height, self.S.nodata

    def sample(self, lon, lat):
        sx = 2 * math.pi * R / self.w
        fx = np.mod(np.asarray(lon) * DEG - self.left, 2 * math.pi * R) / sx
        fy = np.clip((self.top - np.asarray(lat) * DEG) / self.res, 0, self.h - 1.001)
        y0 = np.floor(fy).astype(int)
        ay = (fy - y0).astype(np.float32)
        r0, r1 = y0.min(), min(y0.max() + 2, self.h)
        a = self.S.read(1, window=Window(0, r0, self.w, r1 - r0)).astype(np.float32)
        if a.shape[0] < y0.max() + 2 - r0:
            a = np.vstack([a, a[-1:]])
        a[a == self.nd] = np.nan
        x0 = np.floor(fx).astype(int) % self.w
        x1 = (x0 + 1) % self.w
        ax = (fx - np.floor(fx)).astype(np.float32)
        yy0 = y0 - r0
        yy1 = yy0 + 1
        rows = np.unique(np.r_[yy0, yy1])
        Cx = np.empty((a.shape[0], len(fx)), np.float32)
        Cx[rows] = a[rows][:, x0] * (1 - ax) + a[rows][:, x1] * ax
        return Cx[yy0] * (1 - ay)[:, None] + Cx[yy1] * ay[:, None]


class Contributor:
    """One tile reaching the cell being rendered."""

    def __init__(self, k, layout):
        self.k, self.cell, self.lc = k, cell(k), centre_lon(k)
        self.S = rasterio.open(layout.local_vrt(k, 'dem'))
        self.P = rasterio.open(layout.local_vrt(k, 'prov'))
        with rasterio.open(os.path.join(layout.work, 'corr', f'{k}.tif')) as s:
            tags = s.tags()
            self.c = s.read(1).astype(np.float32)
            self.j0n, self.i0n = int(tags['j0n']), int(tags['i0n'])
        self.c[~np.isfinite(self.c)] = 0.0
        with rasterio.open(os.path.join(layout.work, 'coarse', f'{k}.tif')) as s:
            tags = s.tags()
            self.zc = np.isfinite(s.read(1))
            self.j0c, self.i0c = int(tags['j0']), int(tags['i0'])

    def close(self):
        self.S.close()
        self.P.close()

    def correction(self, lon, lat):
        """Bilinear interpolation of the correction field."""
        hn, wn = self.c.shape
        fx = np.clip((self.lc + wrap(lon - self.lc) + 180.0) / NODE_DEG - 0.5 - self.j0n, 0, wn - 1)
        fy = np.clip((90.0 - lat) / NODE_DEG - 0.5 - self.i0n, 0, hn - 1)
        x0 = np.minimum(np.floor(fx).astype(int), wn - 2) if wn > 1 else np.zeros(len(fx), int)
        y0 = np.minimum(np.floor(fy).astype(int), hn - 2) if hn > 1 else np.zeros(len(fy), int)
        ax, ay = (fx - x0).astype(np.float32), (fy - y0).astype(np.float32)
        x1, y1 = np.minimum(x0 + 1, wn - 1), np.minimum(y0 + 1, hn - 1)
        rows = np.unique(np.r_[y0, y1])
        Cx = np.empty((hn, len(fx)), np.float32)
        Cx[rows] = self.c[rows][:, x0] * (1 - ax) + self.c[rows][:, x1] * ax
        return Cx[y0] * (1 - ay)[:, None] + Cx[y1] * ay[:, None]

    def may_have(self, lon, lat):
        """True if the tile has coarse data anywhere in the lon/lat box."""
        h, w = self.zc.shape
        cols = (np.floor((lon + 180.0) / CS_DEG).astype(int) - self.j0c) % NC
        rows = np.floor((90.0 - lat) / CS_DEG).astype(int) - self.i0c
        cols = np.unique(cols[cols < w])
        rows = np.unique(rows[(rows >= 0) & (rows < h)])
        return len(cols) > 0 and len(rows) > 0 and bool(self.zc[np.ix_(rows, cols)].any())


def _warp(src, shape, lc, x_left, y_top, resampling, nodata, threads):
    a = np.full(shape, nodata, np.float32 if nodata != 255 else np.uint8)
    reproject(rasterio.band(src, 1), a, dst_transform=from_origin(x_left, y_top, PM, PM),
              dst_crs=CRS.from_proj4(local_proj4(lc)), src_nodata=src.nodata, dst_nodata=nodata,
              resampling=resampling, num_threads=threads, warp_mem_limit=512)
    return a


def _bridge_rows(dem, got, limit, tpos=None):
    """Linear interpolation along each row across gaps of at most limit[row]
    pixels with data on both sides. Returns the bridged pixels."""
    filled = np.zeros_like(got)
    for r in np.flatnonzero((~got).any(1) & got.any(1)):
        idx = np.flatnonzero(got[r])
        if len(idx) < 2:
            continue
        for q in np.flatnonzero(np.diff(idx) > 1):
            a, z = idx[q], idx[q + 1]
            if z - a - 1 <= limit[r]:
                t = (np.arange(a + 1, z) - a) / (z - a)
                dem[r, a + 1:z] = dem[r, a] * (1 - t) + dem[r, z] * t
                filled[r, a + 1:z] = True
                if tpos is not None:
                    tpos[r, a + 1:z] = t
    return filled


def render(t, layout, mola, tiles, threads=8):
    """Render the cell of tile t -> tiles/<t>.tif (float32, -9999 no-data) and
    tiles/<t>_mask.tif (0 predicted from CTX, 1 infilled or bridged, 255 no-data)."""
    out = os.path.join(layout.work, 'tiles', f'{t}.tif')
    outm = os.path.join(layout.work, 'tiles', f'{t}_mask.tif')
    lo0, lo1, la0, la1 = cell(t)
    lc = centre_lon(t)
    c0, c1 = int(round((lo0 + 180.0) / PDEG)), int(round((lo1 + 180.0) / PDEG))
    r0, r1 = int(round((90.0 - la1) / PDEG)), int(round((90.0 - la0) / PDEG))
    Tw, Th = c1 - c0, r1 - r0
    antimeridian = -30 <= la0 <= 20 and lo0 in (-180, 170)
    apr = APRON_ANTIMERIDIAN if antimeridian else APRON
    ca, Ta = c0 - apr, Tw + 2 * apr
    lon = -180.0 + (ca + np.arange(Ta) + 0.5) * PDEG
    xl = (-180.0 + ca * PDEG - lc) * DEG
    contribs = [Contributor(k, layout) for k in [t] + neighbours(t, tiles)]
    gtr = from_origin((-180.0 + c0 * PDEG) * DEG, (90.0 - r0 * PDEG) * DEG, PM, PM)
    fprof = dict(driver='GTiff', dtype='float32', nodata=ND, compress='deflate', predictor=3, zlevel=4, tiled=True,
                 blockxsize=512, blockysize=512, bigtiff='if_safer', count=1, width=Tw, height=Th,
                 crs=CRS.from_proj4(EQC0), transform=gtr)
    mprof = dict(fprof, dtype='uint8', nodata=255, predictor=2)
    with rasterio.open(out + '.partial.tif', 'w', **fprof) as D, \
            rasterio.open(outm + '.partial.tif', 'w', **mprof) as DM:
        for rb in range(0, Th, BLOCK_ROWS):
            h = min(BLOCK_ROWS, Th - rb)
            ga, gb = max(0, r0 + rb - APRON), min(H_GLOBAL, r0 + rb + h + APRON)
            ha = gb - ga
            core = slice(r0 + rb - ga, r0 + rb - ga + h)
            lat = 90.0 - (ga + np.arange(ha) + 0.5) * PDEG
            ytop = (90.0 - ga * PDEG) * DEG
            acc = np.zeros((ha, Ta), np.float64)
            wsum = np.zeros((ha, Ta), np.float32)
            bestw = np.zeros((ha, Ta), np.float32)
            msk = np.full((ha, Ta), 255, np.uint8)
            cset = np.zeros((ha, Ta), np.uint16)            # which tiles contribute to each pixel (bit mask)

            def load(C, ya, yb, xa, xb):
                z = _warp(C.S, (yb - ya, xb - xa), lc, xl + xa * PM, ytop - ya * PM, Resampling.bilinear, np.nan,
                          threads)
                return z, np.isfinite(z) & (z != ND)

            for ci, C in enumerate(contribs):
                wf = feather_weight(lon, lat, C.k)
                rr, cc = np.flatnonzero(wf.any(1)), np.flatnonzero(wf.any(0))
                if len(rr) == 0:
                    continue
                ya, yb, xa, xb = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
                z, ok = load(C, ya, yb, xa, xb)
                if not ok.any():
                    continue
                ww = np.where(ok, wf[ya:yb, xa:xb], 0.0).astype(np.float32)
                if not ok.all():
                    # fade a tile out over 1 km toward its own data edge, so the blend stays
                    # continuous where one tile ends inside the feather band
                    ww *= np.clip(distance_transform_edt(ok) / TAPER_PX, 0.0, 1.0).astype(np.float32)
                cset[ya:yb, xa:xb] |= np.where(ww > 0, np.uint16(1 << ci), np.uint16(0))
                cor = C.correction(lon[xa:xb], lat[ya:yb])
                acc[ya:yb, xa:xb] += ww * (np.where(ok, z, 0.0) + cor)
                wsum[ya:yb, xa:xb] += ww
                pm = _warp(C.P, ww.shape, lc, xl + xa * PM, ytop - ya * PM, Resampling.nearest, 255, threads)
                sub = bestw[ya:yb, xa:xb]
                better = ww > sub
                msk[ya:yb, xa:xb][better] = np.where(pm[better] == 1, 1, 0)
                sub[better] = ww[better]
            got = wsum > 0
            dem = np.full((ha, Ta), np.nan, np.float32)
            dem[got] = (acc[got] / wsum[got]).astype(np.float32)

            # pixels no weighted tile covers: any tile with data there
            for ci, C in enumerate(contribs):
                miss = ~got
                if not miss.any():
                    break
                rr, cc = np.flatnonzero(miss.any(1)), np.flatnonzero(miss.any(0))
                ya, yb, xa, xb = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
                if not C.may_have(lon[xa:xb], lat[ya:yb]):
                    continue
                z, ok = load(C, ya, yb, xa, xb)
                take = ok & miss[ya:yb, xa:xb]
                if not take.any():
                    continue
                cor = C.correction(lon[xa:xb], lat[ya:yb])
                cor = np.broadcast_to(cor, z.shape) if np.ndim(cor) else np.full(z.shape, cor, np.float32)
                pm = _warp(C.P, z.shape, lc, xl + xa * PM, ytop - ya * PM, Resampling.nearest, 255, threads)
                dem[ya:yb, xa:xb][take] = z[take] + cor[take]
                msk[ya:yb, xa:xb][take] = np.where(pm[take] == 1, 1, 0)
                got[ya:yb, xa:xb] |= take
                cset[ya:yb, xa:xb][take] = np.uint16(1 << ci)

            # hard seams (neighbouring pixels with no tile in common) cannot be feathered:
            # blank +-1 km around them and bridge below
            band = None
            hs = (cset[:, 1:] != 0) & (cset[:, :-1] != 0) & ((cset[:, 1:] & cset[:, :-1]) == 0)
            vs = (cset[1:, :] != 0) & (cset[:-1, :] != 0) & ((cset[1:, :] & cset[:-1, :]) == 0)
            if hs.any() or vs.any():
                seam = np.zeros((ha, Ta), bool)
                seam[:, 1:] |= hs
                seam[:, :-1] |= hs
                seam[1:, :] |= vs
                seam[:-1, :] |= vs
                sr, sc = np.nonzero(seam)
                y0_, y1_ = max(0, sr.min() - SEAM_PX - 1), min(ha, sr.max() + SEAM_PX + 2)
                x0_, x1_ = max(0, sc.min() - SEAM_PX - 1), min(Ta, sc.max() + SEAM_PX + 2)
                band = np.zeros((ha, Ta), bool)
                band[y0_:y1_, x0_:x1_] = distance_transform_edt(~seam[y0_:y1_, x0_:x1_]) <= SEAM_PX
                band &= got
                band_dem, band_msk = dem[band], msk[band]
                got &= ~band

            # bridge gaps: interpolate (DEM - MOLA) along rows, smooth it north-south, then along columns
            if (~got).any() and got.any():
                mo = mola.sample(lon, lat)
                res = np.where(got, dem - mo, np.nan).astype(np.float32)
                gm = got & np.isfinite(mo)
                cosl = np.maximum(np.cos(np.radians(lat)), 1e-3)
                lim = np.minimum(GAP_EW_KM * 1000.0 / (PM * cosl), apr).astype(int)
                tp = np.zeros(res.shape, np.float32)
                fr = _bridge_rows(res, gm, lim, tp)
                gm |= fr
                if fr.any():
                    k = int(round(1000.0 * SMOOTH_BRIDGE_KM / 2 / PM))
                    v = np.where(fr, res, 0.0)
                    f = fr.astype(np.float32)
                    cv = np.vstack([np.zeros((1, Ta)), np.cumsum(v, 0)])
                    cf = np.vstack([np.zeros((1, Ta)), np.cumsum(f, 0)])
                    lo = np.clip(np.arange(ha) - k, 0, ha)
                    hi = np.clip(np.arange(ha) + k + 1, 0, ha)
                    num, den = cv[hi] - cv[lo], cf[hi] - cf[lo]
                    sm = np.where(den > 0, num / np.maximum(den, 1), res)
                    res = np.where(fr, res + np.sin(np.pi * tp) * (sm - res), res).astype(np.float32)
                fc = _bridge_rows(res.T, gm.T, np.full(Ta, min(int(GAP_NS_KM * 1000.0 / PM), APRON))).T
                new = (fr | fc) & ~got
                dem[new] = mo[new] + res[new]
                got |= new
                msk[new] = 1
            if band is not None:
                lost = band & ~got                      # never lose coverage to the seam band
                if lost.any():
                    dem[lost] = band_dem[lost[band]]
                    msk[lost] = band_msk[lost[band]]
                    got |= lost
            msk[~got] = 255
            D.write(np.where(got, dem, ND)[core, apr:apr + Tw].astype(np.float32), 1, window=Window(0, rb, Tw, h))
            DM.write(msk[core, apr:apr + Tw], 1, window=Window(0, rb, Tw, h))
    for C in contribs:
        C.close()
    os.replace(outm + '.partial.tif', outm)
    os.replace(out + '.partial.tif', out)


# ---------------------------------------------------------------------------
# COG
# ---------------------------------------------------------------------------

def cog(work, out_dem, out_mask, gdal_bin=''):
    """Assemble the rendered cells into Cloud Optimized GeoTIFFs. The
    projection is written as IAU_2015:49910 (Mars sphere, equirectangular,
    lon_0 = 0), identical to the cells' +proj=eqc +R=3396190."""
    tiles_dir = os.path.join(work, 'tiles')
    for kind, out in (('dem', out_dem), ('mask', out_mask)):
        sfx = '' if kind == 'dem' else '_mask'
        files = sorted(os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir)
                       if re.match(rf'^E[p-]\d{{3}}_N[p-]\d{{2}}{sfx}\.tif$', f))
        lst = os.path.join(work, 'vrt', f'inputs_{kind}.txt')
        open(lst, 'w').write('\n'.join(files) + '\n')
        vrt = os.path.join(work, 'vrt', f'mosaic_{kind}.vrt')
        nd = '-9999' if kind == 'dem' else '255'
        subprocess.run([os.path.join(gdal_bin, 'gdalbuildvrt'), '-overwrite', '-srcnodata', nd, '-vrtnodata', nd,
                        '-input_file_list', lst, vrt], check=True)
        if kind == 'dem':
            co = ['-co', 'PREDICTOR=YES', '-co', 'OVERVIEW_RESAMPLING=AVERAGE',
                  '-mo', 'TITLE=Mars global CTX-derived DEM, 18 m, levelled across 10x10 deg boxes',
                  '-mo', 'UNITS=metre', '-mo', 'VERTICAL_DATUM=MOLA areoid (heights as PEDR ELEV)',
                  '-mo', 'HORIZONTAL=simple cylindrical, Mars sphere R=3396190 m, lon -180..180, pixel 360/1185494 deg']
        else:
            co = ['-co', 'PREDICTOR=NO', '-co', 'OVERVIEW_RESAMPLING=NEAREST',
                  '-mo', 'TITLE=Provenance of the Mars global CTX-derived DEM: 0 = CTX-derived, '
                         '1 = infilled (inpainting net / gap bridge), 255 = nodata']
        subprocess.run([os.path.join(gdal_bin, 'gdal_translate'), '-of', 'COG', '-a_srs', 'IAU_2015:49910',
                        '-co', 'COMPRESS=DEFLATE', '-co', 'LEVEL=6', *co, '-co', 'BLOCKSIZE=512',
                        '-co', 'BIGTIFF=YES', '-co', 'NUM_THREADS=ALL_CPUS',
                        '-co', 'OVERVIEWS=IGNORE_EXISTING', '--config', 'GDAL_TIFF_OVR_BLOCKSIZE', '512',
                        vrt, out], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['coarse', 'level', 'render', 'cog'])
    ap.add_argument('tiles', nargs='*', help='tile names (default: every tile)')
    ap.add_argument('--tiles-root', default=None)
    ap.add_argument('--work', required=True)
    ap.add_argument('--dem', default='deseam.tif', help='file name of a tile DEM inside <tiles-root>/<tile>/')
    ap.add_argument('--prov', default='infill.prov.tif')
    ap.add_argument('--mola', default=None, help='MOLA gridded DEM (render)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out-dem', default=None)
    ap.add_argument('--out-mask', default=None)
    ap.add_argument('--gdal-bin', default='')
    args = ap.parse_args()
    if args.stage == 'cog':
        cog(args.work, args.out_dem, args.out_mask, args.gdal_bin)
        return
    layout = Layout(args.tiles_root, args.work, args.dem, args.prov)
    all_tiles = set(layout.tiles())
    mola = MolaBridge(args.mola) if args.stage == 'render' else None
    for t in args.tiles or sorted(all_tiles):
        t0 = time.time()
        if args.stage == 'coarse':
            coarse(t, layout, args.threads)
        elif args.stage == 'level':
            level(t, layout, all_tiles)
        else:
            render(t, layout, mola, all_tiles, args.threads)
        print(f'{args.stage} {t}: {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
