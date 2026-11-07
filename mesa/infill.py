"""Filling the gaps of a levelled tile (Section 2.6 of the paper).

Two U-Nets without illumination conditioning:

* the coarse network sees a 3,072 px (55 km) window reduced to 512 px at
  108 m: the elevation around the gap, its validity mask, the CTX mosaic, its
  validity mask and MOLA. It predicts a correction to MOLA (its output is added
  to the MOLA input).
* the fine network works on 512 px windows at 18 m. It sees the same layers at
  full resolution, with the coarse prediction (upsampled 6 times, central 512
  px) in place of MOLA, and adds an 18 m residual to that prediction.

All elevations of a window share one normalisation: the median and the robust
standard deviation (1.4826 MAD, clipped to 40-300 m) of the valid coarse
elevation around the gap. The fill is composited only into gaps; the join to
the surrounding DEM is made by seams.seamline_merge.

    python -m mesa.infill --dem tiles/E-030_N-10/leveled.tif --ctx tiles/E-030_N-10/ctx_18m.tif \
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --coarse infill_coarse.pt \
        --fine infill_fine.pt --out tiles/E-030_N-10/infill
"""
import argparse
import math
import re

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_dilation, binary_erosion

CFAC = 6                 # 108 m / 18 m
TILE = 512               # network input size at either level
NWIN = TILE * CFAC       # 3072: coarse window at 18 m
FOFF = (NWIN - TILE) // 2
MOAT_PX = 20             # valid pixels within 20 px (360 m) of a gap are hidden from the network
STD_FLOOR_M = 5.0
EMIN, EMAX = -9000.0, 22000.0
CTX_NODATA = -32768
NODATA = -9999.0


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        g = min(8, cout) if cout % min(8, cout) == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the bottleneck positions."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class InfillUNet(nn.Module):
    """U-Net with four stages (64-512 channels) and bottleneck attention.
    forward(x, base) returns the network output plus base: the network
    predicts a residual on MOLA (coarse) or on the coarse fill (fine)."""

    def __init__(self, in_channels, depths=(64, 128, 256, 512), dropout=0.1):
        super().__init__()
        d = list(depths)
        self.in_conv = DoubleConv(in_channels, d[0], dropout)
        self.downs, self.down_pool = nn.ModuleList(), nn.ModuleList()
        for i in range(len(d) - 1):
            self.down_pool.append(nn.MaxPool2d(2))
            self.downs.append(DoubleConv(d[i], d[i + 1], dropout))
        self.attn = SelfAttention(d[-1])
        self.ups, self.up_conv = nn.ModuleList(), nn.ModuleList()
        for i in range(len(d) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(d[i], d[i - 1], 2, stride=2))
            self.up_conv.append(DoubleConv(2 * d[i - 1], d[i - 1], dropout))
        self.out = nn.Conv2d(d[0], 1, 1)

    def forward(self, x, base):
        skips = [self.in_conv(x)]
        h = skips[0]
        for pool, block in zip(self.down_pool, self.downs):
            h = block(pool(h))
            skips.append(h)
        h = self.attn(h)
        for up, block, skip in zip(self.ups, self.up_conv, reversed(skips[:-1])):
            h = block(torch.cat([up(h), skip], dim=1))
        return self.out(h) + base


def coarse_to_fine(coarse):
    """Coarse prediction [B, 1, 512, 512] at 108 m -> the central 512 px at 18 m."""
    up = F.interpolate(coarse, scale_factor=CFAC, mode='bilinear', align_corners=False)
    return up[:, :, FOFF:FOFF + TILE, FOFF:FOFF + TILE]


def load_infill(path, in_channels, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    state = {k.removeprefix('_forward_module.').removeprefix('module.'): v for k, v in state.items()}
    net = InfillUNet(in_channels, dropout=0.0)
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Window inputs
# ---------------------------------------------------------------------------

def decimate(a, valid, f=CFAC):
    """Mean of the valid pixels of each f x f block; a block is valid when at
    least half of its pixels are."""
    H, W = a.shape
    V = valid.reshape(H // f, f, W // f, f).astype(np.float32)
    den = V.sum(axis=(1, 3))
    num = (a.reshape(H // f, f, W // f, f) * V).sum(axis=(1, 3))
    return np.where(den > 0, num / np.maximum(den, 1.0), 0.0).astype(np.float32), den / (f * f) >= 0.5


def robust_frame(values):
    med = float(np.median(values))
    return med, max(float(np.median(np.abs(values - med))) * 1.4826, STD_FLOOR_M)


def normalise(a, med, scale):
    return ((a - med) / scale).astype(np.float32)


def normalise_ctx(ctx, valid):
    v = ctx[valid]
    if v.size < 16:
        return np.zeros_like(ctx, np.float32)
    lo, hi = np.percentile(v, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((ctx - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def plausible_range(c_dem, c_valid, c_mola):
    """(lo, hi) for the elevation of a window: median +- 12 robust standard
    deviations, and never lower than 1.5 km below the window's MOLA minimum.
    None when fewer than 64 coarse pixels are valid."""
    ref = c_dem[c_valid]
    if ref.size < 64:
        return None
    med = float(np.median(ref))
    sd = max(float(np.median(np.abs(ref - med))) * 1.4826, STD_FLOOR_M)
    mv = c_mola[np.isfinite(c_mola)]
    floor = float(mv.min()) - 1500.0 if mv.size else -9000.0
    return max(med - 12.0 * sd, floor), med + 12.0 * sd


def window_frame(c_dem, c_known, c_valid, c_mola):
    """Normalisation of a window from its known coarse elevation (MOLA where
    the window has none), or None when even MOLA is missing."""
    ring = c_dem[c_known]
    if ring.size < 64:
        ring = c_dem[c_valid]
    if ring.size == 0:
        ring = c_mola[np.isfinite(c_mola)]
        if ring.size == 0:
            return None
    med, scale = robust_frame(ring)
    return med, float(np.clip(scale, 40.0, 300.0))


# ---------------------------------------------------------------------------
# MOLA on a tile grid
# ---------------------------------------------------------------------------

def mola_on_grid(src, crs, transform, width, height):
    """MOLA (open dataset) bilinearly sampled at the pixel centres of a grid.
    An equirectangular grid reaching past +-180 deg is first relabelled to an
    equirectangular projection centred on the grid, so that the warp wraps in
    longitude instead of running off the edge of the mosaic."""
    crs = CRS.from_user_input(crs)
    p4 = crs.to_proj4()
    if '+proj=eqc' in p4:
        get = lambda k, d: float(re.search(rf'\+{k}=(-?[0-9.eE+-]+)', p4).group(1)) if f'+{k}=' in p4 else d
        r = get('R', None) or get('a', None)
        lon0, lat_ts = get('lon_0', 0.0), get('lat_ts', 0.0)
        k = r * math.cos(math.radians(lat_ts)) * math.pi / 180.0
        x0, x1 = transform.c, transform.c + transform.a * width
        if r is not None and get('x_0', 0.0) == 0.0 and max(abs(lon0 + x0 / k), abs(lon0 + x1 / k)) > 180.0:
            dlon = (transform.c + transform.a * width / 2.0) / k
            lonc = (lon0 + dlon + 180.0) % 360.0 - 180.0
            p4 = (re.sub(r'\+lon_0=-?[0-9.eE+-]+', f'+lon_0={lonc!r}', p4) if '+lon_0=' in p4
                  else p4 + f' +lon_0={lonc!r}')
            crs = CRS.from_proj4(p4)
            transform = Affine.translation(-dlon * k, 0.0) * transform
    with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                   resampling=Resampling.bilinear) as v:
        return v.read(1)


def _inside(win, ds, tol=1e-6):
    return (win.col_off >= -tol and win.row_off >= -tol and win.col_off + win.width <= ds.width + tol
            and win.row_off + win.height <= ds.height + tol)


# ---------------------------------------------------------------------------
# Coarse fill of a whole tile
# ---------------------------------------------------------------------------

def bartlett(n):
    w = np.clip(1.0 - np.abs(np.linspace(-1, 1, n, dtype=np.float32)), 0.02, None)
    return np.outer(w, w).astype(np.float32)


def coarse_inputs(a_dem, a_ctx, a_mola, nodata):
    """Coarse network inputs for one 3,072 px window."""
    dem_ok = np.isfinite(a_dem) & (a_dem > EMIN) & (a_dem < EMAX)
    if nodata is not None and np.isfinite(nodata):
        dem_ok &= a_dem != nodata
    ctx_ok = np.isfinite(a_ctx) & (a_ctx != CTX_NODATA)
    c_dem, c_dv = decimate(a_dem, dem_ok)
    c_ctx, c_cv = decimate(a_ctx, ctx_ok)
    c_mola, _ = decimate(a_mola, np.ones_like(dem_ok))
    c_cv &= c_ctx != CTX_NODATA
    rng = plausible_range(c_dem, c_dv, c_mola)
    if rng is not None:
        c_dv &= (c_dem > rng[0]) & (c_dem < rng[1])
    c_known = binary_erosion(c_dv, iterations=max(1, MOAT_PX // CFAC))
    frame = window_frame(c_dem, c_known, c_dv, c_mola)
    if frame is None:
        return None
    med, scale = frame
    c_dem_n = np.where(c_dv, normalise(c_dem, med, scale), 0.0).astype(np.float32)
    c_mola_n = normalise(c_mola, med, scale)
    x = np.stack([np.where(c_known, c_dem_n, 0.0).astype(np.float32), c_known.astype(np.float32),
                  normalise_ctx(c_ctx, c_cv), c_cv.astype(np.float32), c_mola_n], 0)
    return x, c_mola_n, med, scale


def window_centres(n, stride, lo):
    hi = n - 1 - lo
    g = list(range(lo, min(n, hi + 1), stride))
    if not g:
        g = [min(max(n // 2, lo), hi)]
    last = min(max(n - 1, lo), hi)
    if g[-1] != last:
        g.append(last)
    return sorted(set(g))


@torch.no_grad()
def coarse_fill(dem_path, ctx_path, mola_path, net, device, stride=1536, batch=8):
    """Coarse prediction over a whole tile at 18 m: windows every `stride`
    pixels, the 108 m prediction upsampled bilinearly and blended with Bartlett
    weights. Returns (model surface, covered) with NaN where uncovered."""
    dem = rasterio.open(dem_path)
    ctx = rasterio.open(ctx_path)
    mola_src = rasterio.open(mola_path)
    mola = WarpedVRT(mola_src, crs=dem.crs, resampling=Resampling.bilinear)
    H, W, nd, lo = dem.height, dem.width, dem.nodata, NWIN // 2
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    fw = bartlett(NWIN)

    def run(buf):
        x = torch.from_numpy(np.stack([b[0] for b in buf])).to(device)
        m = torch.from_numpy(np.stack([b[1] for b in buf])).unsqueeze(1).to(device)
        up = F.interpolate(net(x, m), scale_factor=CFAC, mode='bilinear', align_corners=False)
        up = up.squeeze(1).cpu().numpy()
        for k, (_, _, med, scale, r0, c0) in enumerate(buf):
            pred = up[k] * scale + med
            rr0, cc0 = max(r0, 0), max(c0, 0)
            rr1, cc1 = min(r0 + NWIN, H), min(c0 + NWIN, W)
            sl = (slice(rr0 - r0, rr1 - r0), slice(cc0 - c0, cc1 - c0))
            acc[rr0:rr1, cc0:cc1] += pred[sl] * fw[sl]
            wsum[rr0:rr1, cc0:cc1] += fw[sl]

    buf = []
    for cr in window_centres(H, stride, lo):
        for cc in window_centres(W, stride, lo):
            win = Window(cc - lo, cr - lo, NWIN, NWIN)
            a_dem = dem.read(1, window=win).astype(np.float32)
            a_ctx = ctx.read(1, window=win).astype(np.float32)
            wv = from_bounds(*dem.window_bounds(win), transform=mola.transform)
            if _inside(wv, mola):
                a_mola = mola.read(1, window=wv, out_shape=(NWIN, NWIN),
                                   resampling=Resampling.bilinear).astype(np.float32)
            else:
                a_mola = mola_on_grid(mola_src, dem.crs, dem.window_transform(win), NWIN, NWIN).astype(np.float32)
            inputs = coarse_inputs(a_dem, a_ctx, a_mola, nd)
            if inputs is None:
                continue
            buf.append(inputs + (cr - lo, cc - lo))
            if len(buf) >= batch:
                run(buf)
                buf = []
    if buf:
        run(buf)
    covered = wsum > 0
    model = np.where(covered, acc / np.maximum(wsum, 1e-6), np.nan).astype(np.float32)
    return model, covered


# ---------------------------------------------------------------------------
# Fine (18 m) pass
# ---------------------------------------------------------------------------

HALF = NWIN // 2
PAD = HALF + 264         # a multiple of CFAC, room for windows centred past the tile edge
FINE_STRIDE = 258        # about half a window, a multiple of CFAC
NEED_BUFFER = 48         # pixels around the gaps that must get the fine surface


@torch.no_grad()
def fine_fill(dem_path, ctx_path, mola_path, coarse_net, fine_net, coarse_model, device, batch=16):
    """18 m pass over every 512 px window that touches a gap or the 48 px
    around it. The coarse network is run again inside each window's 55 km
    context (from a tile-wide 108 m decimation, which equals the per-window
    one because window centres lie on the 6 px grid). Elsewhere the coarse
    surface is kept. Returns the model surface (NaN where uncovered)."""
    D = rasterio.open(dem_path)
    nd, H, W = D.nodata, D.height, D.width
    Hp = ((H + 2 * PAD + CFAC - 1) // CFAC) * CFAC
    Wp = ((W + 2 * PAD + CFAC - 1) // CFAC) * CFAC
    ptr = D.transform * Affine.translation(-PAD, -PAD)

    dem = np.full((Hp, Wp), np.nan, np.float32)
    dem[PAD:PAD + H, PAD:PAD + W] = D.read(1).astype(np.float32)
    dv = np.isfinite(dem) & (dem > EMIN) & (dem < EMAX)
    if nd is not None:
        dv &= dem != nd
    with rasterio.open(ctx_path) as C:
        ctx = np.zeros((Hp, Wp), np.float32)
        ctx[PAD:PAD + H, PAD:PAD + W] = C.read(1).astype(np.float32)
        cv = np.zeros((Hp, Wp), bool)
        cv[PAD:PAD + H, PAD:PAD + W] = True
        cv &= np.isfinite(ctx)
        if C.nodata is not None:
            cv &= ctx != C.nodata
    with rasterio.open(mola_path) as M0:
        mola = mola_on_grid(M0, D.crs, ptr, Wp, Hp).astype(np.float32)

    c_dem, c_dv = decimate(np.nan_to_num(dem), dv)
    c_ctx, c_cv = decimate(ctx, cv)
    c_mola, _ = decimate(mola, np.ones_like(dv))

    hole = np.zeros((Hp, Wp), bool)
    hole[PAD:PAD + H, PAD:PAD + W] = ~dv[PAD:PAD + H, PAD:PAD + W]
    need = binary_dilation(hole, iterations=NEED_BUFFER)
    need[:PAD] = False
    need[PAD + H:] = False
    need[:, :PAD] = False
    need[:, PAD + W:] = False

    def centres(n):
        return [c - (c % CFAC) for c in range(PAD + 252, PAD + n + 256, FINE_STRIDE)]

    windows = [(cr, cc) for cr in centres(H) for cc in centres(W)
               if need[cr - 256:cr + 256, cc - 256:cc + 256].any()]
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros_like(acc)
    fw = bartlett(TILE)
    c_moat = max(1, MOAT_PX // CFAC)

    def run(buf):
        xc = torch.from_numpy(np.stack([b['x_c'] for b in buf])).to(device)
        mc = torch.from_numpy(np.stack([b['mola_c'] for b in buf]))[:, None].to(device)
        xf = torch.from_numpy(np.stack([b['x_f'] for b in buf])).to(device)
        fill = coarse_to_fine(coarse_net(xc, mc))
        pred = fine_net(torch.cat([xf, fill], 1), fill)[:, 0].float().cpu().numpy()
        for k, b in enumerate(buf):
            pm = pred[k] * b['scale'] + b['med']
            tr, tc = b['tr'], b['tc']
            y0, x0 = max(tr, PAD), max(tc, PAD)
            y1, x1 = min(tr + TILE, PAD + H), min(tc + TILE, PAD + W)
            if y1 <= y0 or x1 <= x0:
                continue
            src = (slice(y0 - tr, y1 - tr), slice(x0 - tc, x1 - tc))
            dst = (slice(y0 - PAD, y1 - PAD), slice(x0 - PAD, x1 - PAD))
            acc[dst] += pm[src] * fw[src]
            wsum[dst] += fw[src]

    buf = []
    for cr, cc in windows:
        wr, wc = (cr - HALF) // CFAC, (cc - HALF) // CFAC
        win_c = (slice(wr, wr + TILE), slice(wc, wc + TILE))
        cd, cdv, cx, ccv, cm = c_dem[win_c], c_dv[win_c].copy(), c_ctx[win_c], c_cv[win_c], c_mola[win_c]
        tr, tc = cr - 256, cc - 256
        win_f = (slice(tr, tr + TILE), slice(tc, tc + TILE))
        fd, fdv, fx, fcv = np.nan_to_num(dem[win_f]), dv[win_f].copy(), ctx[win_f], cv[win_f]
        rng = plausible_range(cd, cdv, cm)
        if rng is not None:
            cdv &= (cd > rng[0]) & (cd < rng[1])
            fdv &= (fd > rng[0]) & (fd < rng[1])
        c_known = binary_erosion(cdv, iterations=c_moat)
        f_known = binary_erosion(fdv, iterations=MOAT_PX)
        frame = window_frame(cd, c_known, cdv, cm)
        if frame is None:
            continue
        med, scale = frame
        c_dem_n = np.where(cdv, normalise(cd, med, scale), 0.0).astype(np.float32)
        x_c = np.stack([np.where(c_known, c_dem_n, 0.0), c_known, normalise_ctx(cx, ccv), ccv,
                        normalise(cm, med, scale)], 0).astype(np.float32)
        f_dem_n = np.where(fdv, normalise(fd, med, scale), 0.0)
        x_f = np.stack([np.where(f_known, f_dem_n, 0.0), f_known, normalise_ctx(fx, fcv), fcv], 0).astype(np.float32)
        buf.append({'x_c': x_c, 'mola_c': normalise(cm, med, scale), 'x_f': x_f, 'med': med, 'scale': scale,
                    'tr': tr, 'tc': tc})
        if len(buf) >= batch:
            run(buf)
            buf = []
    if buf:
        run(buf)

    covered = wsum > 0
    model = np.full((H, W), np.nan, np.float32)
    model[covered] = acc[covered] / wsum[covered]
    keep = ~covered & np.isfinite(coarse_model)
    model[keep] = coarse_model[keep]
    return model


def fill_tile(dem_path, ctx_path, mola_path, coarse_path, fine_path, out_prefix, device=None):
    """Writes <out_prefix>.model.tif (the infill surface over every covered
    pixel, the input of seams.seamline_merge) and <out_prefix>.prov.tif
    (1 where a gap of the levelled DEM was filled, 0 elsewhere)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    coarse_net = load_infill(coarse_path, 5, device)
    coarse_model, covered = coarse_fill(dem_path, ctx_path, mola_path, coarse_net, device)
    model = coarse_model
    if fine_path:
        model = fine_fill(dem_path, ctx_path, mola_path, coarse_net, load_infill(fine_path, 5, device),
                          coarse_model, device)
    with rasterio.open(dem_path) as d:
        dem = d.read(1)
        nd = d.nodata if d.nodata is not None else NODATA
        profile = dict(driver='GTiff', height=d.height, width=d.width, count=1, crs=d.crs, transform=d.transform,
                       compress='deflate', tiled=True, bigtiff='yes')
    prov = ((dem == nd) & covered).astype(np.uint8)
    with rasterio.open(f'{out_prefix}.model.tif', 'w', dtype='float32', nodata=nd, **profile) as o:
        o.write(np.where(np.isfinite(model), model, nd).astype(np.float32), 1)
    with rasterio.open(f'{out_prefix}.prov.tif', 'w', dtype='uint8', nodata=255, **profile) as o:
        o.write(prov, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dem', required=True, help='levelled DEM of the tile')
    ap.add_argument('--ctx', required=True, help='CTX mosaic on the same grid (byte, no-data 0)')
    ap.add_argument('--mola', required=True)
    ap.add_argument('--coarse', required=True, help='coarse infill network')
    ap.add_argument('--fine', default=None, help='fine infill network (omit for the coarse surface only)')
    ap.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()
    fill_tile(args.dem, args.ctx, args.mola, args.coarse, args.fine, args.out)


if __name__ == '__main__':
    main()
