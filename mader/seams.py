"""Joins and seams of a product tile (Sections 2.5 step 5 and 2.6 of the paper).

seamline_merge   joins the infill surface to the levelled DEM along a
                 minimum-cost boundary (Boykov and Jolly, 2001): around each
                 gap, a band of 32 px of real DEM is cut where the two surfaces
                 differ least in level and slope, and an 8 px feather is applied
                 across the cut. The result is the tile before de-seaming.

deseam           repairs the steps that blending leaves where two predicted
                 DEMs meet with little overlap (butt joints). The seam lines are
                 known from geometry: they are where the dominant DEM (largest
                 blend weight) changes. At each strong seam the surface is
                 re-integrated from the mean gradient of the DEMs that cover it
                 (Perez et al., 2003), tied weakly to the blended surface:

                     (lam I - L) P = lam F - div(G),   lam = 1e-4,

                 and P is spliced into F along a minimum-cost boundary within
                 160 px of the seam. A splice is kept only if it reduces the
                 step measure by at least 20%.

    python -m mader.seams merge --dem leveled.tif --model infill.model.tif --out final.tif
    python -m mader.seams deseam --final final.tif --prov infill.prov.tif --frames 'leveled/*_leveled.tif' \
        --out deseam.tif
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import shutil

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.sparse.linalg import LinearOperator, cg

NODATA = -9999.0


# ---------------------------------------------------------------------------
# Minimum cut between two surfaces
# ---------------------------------------------------------------------------

def _maxflow_side(A, B, band, force_a, force_b, lam, w_off):
    """Pixels of `band` on A's side of the minimum cut. Cutting between
    neighbours p, q costs 1 + 10 (w_off (|D_p| + |D_q|) + lam |D_p - D_q|),
    D = A - B, so the boundary runs where the surfaces agree in level and
    slope. force_a pixels stay with A, force_b pixels with B."""
    idx = -np.ones(band.shape, np.int64)
    ys, xs = np.where(band)
    n = ys.size
    if n == 0:
        return np.zeros(band.shape, bool)
    idx[ys, xs] = np.arange(n)
    S, T = n, n + 1
    delta = A - B
    rows, cols, caps = [], [], []
    for dy in (0, 1):
        if dy:
            pm = band[:-1, :] & band[1:, :]
            py, px = np.where(pm)
            qy, qx = py + 1, px
        else:
            pm = band[:, :-1] & band[:, 1:]
            py, px = np.where(pm)
            qy, qx = py, px + 1
        ip, iq = idx[py, px], idx[qy, qx]
        kink = np.abs(delta[py, px] - delta[qy, qx])
        step = np.abs(delta[py, px]) + np.abs(delta[qy, qx])
        c = (1 + np.rint(10.0 * (w_off * step + lam * kink))).astype(np.int64)
        rows.append(np.concatenate([ip, iq]))
        cols.append(np.concatenate([iq, ip]))
        caps.append(np.concatenate([c, c]))
    big = 1 << 24
    ia, ib = idx[force_a & band], idx[force_b & band]
    rows.append(np.concatenate([np.full(ia.size, S), ia]))
    cols.append(np.concatenate([ia, np.full(ia.size, S)]))
    caps.append(np.concatenate([np.full(ia.size, big, np.int64), np.zeros(ia.size, np.int64)]))
    rows.append(np.concatenate([ib, np.full(ib.size, T)]))
    cols.append(np.concatenate([np.full(ib.size, T), ib]))
    caps.append(np.concatenate([np.full(ib.size, big, np.int64), np.zeros(ib.size, np.int64)]))
    cap = coo_matrix((np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 2, n + 2)).tocsr()
    cap.sum_duplicates()
    flow = maximum_flow(cap, S, T)
    resid = cap - flow.flow
    resid.eliminate_zeros()
    reach = np.zeros(n + 2, bool)
    reach[breadth_first_order(resid, S, directed=True, return_predecessors=False)] = True
    side = np.zeros(band.shape, bool)
    side[ys, xs] = reach[:n]
    return side


def cut_side(A, B, band, force_a, force_b, lam=1.0, w_off=1.0, target=300000):
    """_maxflow_side, solved on a coarser grid when the band has more than
    `target` pixels (a seam line does not need 18 m precision)."""
    f = max(1, int(round((int(band.sum()) / target) ** 0.5)))
    if f == 1:
        return _maxflow_side(A, B, band, force_a, force_b, lam, w_off)
    H, W = band.shape
    Hp, Wp = -(-H // f) * f, -(-W // f) * f

    def pad(a, val, dt):
        b = np.full((Hp, Wp), val, dt)
        b[:H, :W] = a
        return b

    band_p, fa_p, fb_p = pad(band, False, bool), pad(force_a, False, bool), pad(force_b, False, bool)
    d_p = pad((A - B).astype(np.float32), 0.0, np.float32)
    reduce = lambda m: m.reshape(Hp // f, f, Wp // f, f).any(axis=(1, 3))
    bc = reduce(band_p)
    fac = reduce(fa_p) & bc
    fbc = reduce(fb_p) & bc & ~fac
    cnt = band_p.reshape(Hp // f, f, Wp // f, f).sum(axis=(1, 3)).astype(np.float32)
    dsum = np.where(band_p, d_p, 0.0).reshape(Hp // f, f, Wp // f, f).sum(axis=(1, 3))
    dc = np.where(cnt > 0, dsum / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
    side = _maxflow_side(dc, np.zeros_like(dc), bc, fac, fbc, lam, w_off)
    return np.repeat(np.repeat(side, f, 0), f, 1)[:H, :W] & band


def feathered_alpha(hard, feather):
    """Weight of the first surface: 1 inside `hard`, 0 outside, with a linear
    transition of +-feather px across the boundary."""
    sd = ndi.distance_transform_edt(hard) - ndi.distance_transform_edt(~hard)
    return np.clip(0.5 + sd / (2.0 * max(feather, 1)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Join of the infill (seamline merge)
# ---------------------------------------------------------------------------

def _join_gap(dem, model, gap, valid, w, k, lam, w_off):
    band = ndi.binary_dilation(gap, iterations=w) & valid
    if not band.any():
        return None
    outside = valid & ~band & ~gap
    inner = band & ndi.binary_dilation(gap, iterations=1)
    outer = band & ndi.binary_dilation(outside, iterations=1)
    if not inner.any() or not outer.any():
        return None
    use_model = cut_side(model, dem, band, inner, outer, lam, w_off)
    alpha = feathered_alpha(np.maximum(gap.astype(np.float32), (use_model & band).astype(np.float32)) > 0.5,
                            k).astype(np.float32)
    region = ndi.binary_dilation(gap | band, iterations=k + 1)
    alpha = np.where(region, alpha, 0.0).astype(np.float32)
    return np.where(gap, 1.0, alpha), region


def seamline_merge(dem_path, model_path, out_path, buffer=32, feather=8, min_gap_px=30, clamp_margin=500.0,
                   gap_fill_px=64, pad=8, lam=1.0, w_off=1.0):
    with rasterio.open(model_path) as md:
        model = md.read(1).astype(np.float32)
        mnd = md.nodata
    with rasterio.open(dem_path) as dd:
        dem = dd.read(1).astype(np.float32)
        nd, crs, transform = dd.nodata, dd.crs, dd.transform
    gap = dem == nd
    covered = (model != mnd) & np.isfinite(model)
    valid = np.isfinite(dem) & ~gap & covered

    # Clamp the infill to the range of the surrounding DEM (+-500 m beyond its
    # 0.01 and 99.99 percentiles, from every 37th pixel): a network starved of
    # context can extrapolate wildly.
    vv = dem.reshape(-1)[::37]
    vv = vv[np.isfinite(vv) & (vv != nd)]
    lo = float(np.percentile(vv, 0.01)) - clamp_margin
    hi = float(np.percentile(vv, 99.99)) + clamp_margin
    np.clip(model, lo, hi, out=model)

    out = dem.copy()
    out[gap & covered] = model[gap & covered]
    lab, n = ndi.label(gap, structure=np.ones((3, 3)))
    sizes = np.zeros(n + 1, np.int64)
    for i in range(0, lab.shape[0], 4096):
        sizes += np.bincount(lab[i:i + 4096].ravel(), minlength=n + 1)
    margin = buffer + feather + pad
    for ci, sl in enumerate(ndi.find_objects(lab), start=1):
        if sizes[ci] < min_gap_px:
            continue
        y0, y1 = max(0, sl[0].start - margin), min(dem.shape[0], sl[0].stop + margin)
        x0, x1 = max(0, sl[1].start - margin), min(dem.shape[1], sl[1].stop + margin)
        ldem, lmodel = dem[y0:y1, x0:x1], model[y0:y1, x0:x1]
        r = _join_gap(ldem, lmodel, lab[y0:y1, x0:x1] == ci, valid[y0:y1, x0:x1], buffer, feather, lam, w_off)
        if r is None:
            continue
        alpha, region = r
        # never blend against no-data: model missing -> DEM, DEM missing -> model
        lcov = covered[y0:y1, x0:x1]
        dvalid = np.isfinite(ldem) & (ldem != nd)
        a = np.where(lcov, alpha, 0.0)
        a = np.where(~dvalid & lcov, 1.0, a)
        a = np.where(dvalid | lcov, a, np.nan)
        blended = a * np.where(lcov, lmodel, 0.0) + (1.0 - a) * np.where(dvalid, ldem, 0.0)
        ok = region & np.isfinite(a)
        out[y0:y1, x0:x1] = np.where(ok, blended.astype(np.float32), out[y0:y1, x0:x1])

    # Gaps the infill does not reach (tile borders) are filled from the nearest
    # value, but only within gap_fill_px of data.
    rem = ~np.isfinite(out) | (out == nd)
    if rem.any():
        rlab, _ = ndi.label(rem)
        for sl in ndi.find_objects(rlab):
            y0, y1 = max(0, sl[0].start - gap_fill_px - 1), min(out.shape[0], sl[0].stop + gap_fill_px + 1)
            x0, x1 = max(0, sl[1].start - gap_fill_px - 1), min(out.shape[1], sl[1].stop + gap_fill_px + 1)
            sub = out[y0:y1, x0:x1]
            r = ~np.isfinite(sub) | (sub == nd)
            if not r.any() or r.all():
                continue
            dist, ind = ndi.distance_transform_edt(r, return_distances=True, return_indices=True)
            close = r & (dist <= gap_fill_px)
            if close.any():
                sub[close] = sub[tuple(i[close] for i in ind)]
            out[y0:y1, x0:x1] = sub
    profile = dict(driver='GTiff', height=dem.shape[0], width=dem.shape[1], count=1, dtype='float32', crs=crs,
                   transform=transform, nodata=nd, compress='deflate', tiled=True, bigtiff='yes')
    with rasterio.open(out_path, 'w', **profile) as o:
        o.write(out, 1)


# ---------------------------------------------------------------------------
# Seam lines and step strength
# ---------------------------------------------------------------------------

_G = {}


def _open_frames(paths):
    os.environ.setdefault('GDAL_CACHEMAX', '256')
    return [(ds, ds.bounds) for ds in (rasterio.open(p) for p in paths)]


def _overlapping(frames, wb):
    return [(k, ds) for k, (ds, b) in enumerate(frames)
            if not (b.right <= wb[0] or b.left >= wb[2] or b.top <= wb[1] or b.bottom >= wb[3])]


def _init_mask(prov, weights):
    _G['ref'] = rasterio.open(prov)
    _G['frames'] = _open_frames(weights)


def _boundary(lab, valid):
    edge = np.zeros(lab.shape, bool)
    h = valid[:, :-1] & valid[:, 1:] & (lab[:, :-1] != lab[:, 1:])
    edge[:, :-1] |= h
    edge[:, 1:] |= h
    v = valid[:-1, :] & valid[1:, :] & (lab[:-1, :] != lab[1:, :])
    edge[:-1, :] |= v
    edge[1:, :] |= v
    return edge


def _mask_block(job):
    r0, c0, r1, c1, halo, dilate = job
    ref, frames = _G['ref'], _G['frames']
    H, W = ref.height, ref.width
    hr0, hr1, hc0, hc1 = max(0, r0 - halo), min(H, r1 + halo), max(0, c0 - halo), min(W, c1 + halo)
    th, tw = hr1 - hr0, hc1 - hc0
    win = Window(hc0, hr0, tw, th)
    wtr = rasterio.windows.transform(win, ref.transform)
    best = np.zeros((th, tw), np.float32)
    dom = np.full((th, tw), -1, np.int16)
    for k, ds in _overlapping(frames, rasterio.windows.bounds(win, ref.transform)):
        with WarpedVRT(ds, crs=ref.crs, transform=wtr, width=tw, height=th, resampling=Resampling.nearest) as v:
            w = v.read(1).astype(np.float32)
        m = np.isfinite(w) & (w > 0.0)
        if ds.nodata is not None:
            m &= w != ds.nodata
        take = m & (w > best)
        best[take] = w[take]
        dom[take] = k
    ff = _boundary(dom, dom >= 0)                               # DEM <-> DEM
    prov = ref.read(1, window=win)
    pv = np.isin(prov, (0, 1))
    if ref.nodata is not None:
        pv &= prov != ref.nodata
    fi = _boundary(prov, pv)                                    # DEM <-> infill
    if dilate > 0:
        ff = ndi.binary_dilation(ff, iterations=dilate)
        fi = ndi.binary_dilation(fi, iterations=dilate)
    mask = np.zeros((th, tw), np.uint8)
    mask[ff] |= 1
    mask[fi] |= 2
    s = (slice(r0 - hr0, r0 - hr0 + r1 - r0), slice(c0 - hc0, c0 - hc0 + c1 - c0))
    return r0, c0, mask[s], dom[s]


def seam_mask(prov_path, weight_paths, out_prefix, block=2048, dilate=2, jobs=16):
    """<prefix>_mask.tif: bit 1 = seam between two predicted DEMs, bit 2 =
    boundary of the infill. <prefix>_domid.tif: index (into the sorted weight
    files) of the DEM with the largest blend weight, -1 where none."""
    with rasterio.open(prov_path) as ref:
        H, W, crs, tr = ref.height, ref.width, ref.crs, ref.transform
    mask = np.zeros((H, W), np.uint8)
    dom = np.full((H, W), -1, np.int16)
    blocks = [(r, c, min(r + block, H), min(c + block, W), dilate + 1, dilate)
              for r in range(0, H, block) for c in range(0, W, block)]
    with mp.get_context('fork').Pool(jobs, _init_mask, (prov_path, weight_paths), maxtasksperchild=40) as pool:
        for r0, c0, m, d in pool.imap_unordered(_mask_block, blocks, chunksize=1):
            mask[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m
            dom[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = d
    profile = dict(driver='GTiff', height=H, width=W, count=1, crs=crs, transform=tr, compress='deflate',
                   tiled=True, bigtiff='yes')
    with rasterio.open(f'{out_prefix}_mask.tif', 'w', dtype='uint8', nodata=0, **profile) as d:
        d.write(mask, 1)
    with rasterio.open(f'{out_prefix}_domid.tif', 'w', dtype='int16', nodata=-1, **profile) as d:
        d.write(dom, 1)


def step_strength(M, frames):
    """aF = min over DEMs f of |grad(M - A_f)|, using only pixels where A_f and
    its four neighbours are valid. M and every A_f show the same terrain, so the
    terrain cancels and what remains is a step committed by the blend."""
    a = np.full(M.shape, np.inf, np.float32)
    for A, ok in frames:
        mg = ok.copy()
        mg[1:-1, :] &= ok[2:, :] & ok[:-2, :]
        mg[:, 1:-1] &= ok[:, 2:] & ok[:, :-2]
        mg[0, :] = mg[-1, :] = mg[:, 0] = mg[:, -1] = False
        gy, gx = np.gradient(M - A)
        np.minimum(a, np.where(mg, np.hypot(gx, gy).astype(np.float32), np.inf), out=a)
    return np.where(np.isfinite(a), a, 0.0)


def _init_census(final, levelled):
    _G['fe'] = rasterio.open(final)
    _G['frames'] = _open_frames(levelled)


def _census_block(job):
    r0, c0, block, decim = job
    fe, frames = _G['fe'], _G['frames']
    H, W = fe.height, fe.width
    r1, c1 = min(r0 + block, H), min(c0 + block, W)
    hr0, hr1, hc0, hc1 = max(0, r0 - 1), min(H, r1 + 1), max(0, c0 - 1), min(W, c1 + 1)
    th, tw = hr1 - hr0, hc1 - hc0
    win = Window(hc0, hr0, tw, th)
    wtr = rasterio.windows.transform(win, fe.transform)
    F = fe.read(1, window=win).astype(np.float64)
    okF = np.isfinite(F) & (F != NODATA)
    cov = np.zeros((th, tw), np.int16)
    aF = np.full((th, tw), np.inf, np.float32)
    for _, ds in _overlapping(frames, rasterio.windows.bounds(win, fe.transform)):
        with WarpedVRT(ds, crs=fe.crs, transform=wtr, width=tw, height=th, resampling=Resampling.nearest) as v:
            A = v.read(1).astype(np.float32)
        m = np.isfinite(A) & (A != (ds.nodata if ds.nodata is not None else NODATA)) & okF
        if m.sum() < 200:
            continue
        cov += m.astype(np.int16)
        mg = m.copy()
        mg[1:-1, :] &= m[2:, :] & m[:-2, :]
        mg[:, 1:-1] &= m[:, 2:] & m[:, :-2]
        mg[0, :] = mg[-1, :] = mg[:, 0] = mg[:, -1] = False
        gy, gx = np.gradient(F - A)
        np.minimum(aF, np.where(mg, np.hypot(gx, gy).astype(np.float32), np.inf), out=aF)
    s = (slice(r0 - hr0, r0 - hr0 + r1 - r0), slice(c0 - hc0, c0 - hc0 + c1 - c0))
    aF, cov = aF[s], cov[s]
    aF = np.where((cov >= 2) & np.isfinite(aF), aF, 0.0).astype(np.float32)
    ph, pw = (-aF.shape[0]) % decim, (-aF.shape[1]) % decim
    if ph or pw:
        aF = np.pad(aF, ((0, ph), (0, pw)))
        cov = np.pad(cov, ((0, ph), (0, pw)))
    dh, dw = aF.shape[0] // decim, aF.shape[1] // decim
    return (r0 // decim, c0 // decim, aF.reshape(dh, decim, dw, decim).max(axis=(1, 3)),
            cov.reshape(dh, decim, dw, decim).max(axis=(1, 3)).astype(np.int16))


def seam_sites(final_path, levelled_paths, block=2048, decim=4, thresh=8.0, min_site_px=8, jobs=16):
    """Strong seams: connected areas where the step strength, max-pooled over
    4 x 4 px, exceeds 8 m/px, in overlaps of at least two DEMs. Returns a list
    of sites with their bounding boxes (full-resolution pixels)."""
    with rasterio.open(final_path) as fe:
        H, W = fe.height, fe.width
    DH, DW = -(-H // decim), -(-W // decim)
    aF = np.zeros((DH, DW), np.float32)
    cov = np.zeros((DH, DW), np.int16)
    blocks = [(r, c, block, decim) for r in range(0, H, block) for c in range(0, W, block)]
    with mp.get_context('fork').Pool(jobs, _init_census, (final_path, levelled_paths), maxtasksperchild=40) as pool:
        for dr, dc, a, c in pool.imap_unordered(_census_block, blocks, chunksize=1):
            aF[dr:dr + a.shape[0], dc:dc + a.shape[1]] = np.maximum(aF[dr:dr + a.shape[0], dc:dc + a.shape[1]], a)
            cov[dr:dr + c.shape[0], dc:dc + c.shape[1]] = np.maximum(cov[dr:dr + c.shape[0], dc:dc + c.shape[1]], c)
    lab, _ = ndi.label(aF > thresh)
    sites = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        comp = lab[sl] == i
        if int(comp.sum()) < min_site_px:
            continue
        sites.append(dict(peak=float(np.where(comp, aF[sl], 0.0).max()),
                          rows=(sl[0].start * decim, sl[0].stop * decim),
                          cols=(sl[1].start * decim, sl[1].stop * decim)))
    sites.sort(key=lambda s: s['peak'], reverse=True)
    return sites


def _init_gate(final, levelled, mask):
    _G['fin'] = rasterio.open(final)
    _G['srcs'] = [rasterio.open(p) for p in levelled]
    _G['sm'] = rasterio.open(mask)


def _read_frames_bilinear(srcs, crs, win_tr, w, h, wb):
    out = []
    for s in srcs:
        b = s.bounds
        if b.right <= wb[0] or b.left >= wb[2] or b.top <= wb[1] or b.bottom >= wb[3]:
            continue
        with WarpedVRT(s, crs=crs, transform=win_tr, width=w, height=h, resampling=Resampling.bilinear,
                       src_nodata=s.nodata, nodata=NODATA) as v:
            A = v.read(1).astype(np.float64)
        ok = np.isfinite(A) & (A != NODATA)
        if ok.any():
            out.append((A, ok))
    return out


def _gate_site(job):
    k, r0, c0, r1, c1 = job
    fin, sm = _G['fin'], _G['sm']
    win = Window(c0, r0, c1 - c0, r1 - r0)
    tr = rasterio.windows.transform(win, fin.transform)
    seam = (sm.read(1, window=win) & 1) > 0
    if not seam.any():
        return k, 0.0
    F = fin.read(1, window=win).astype(np.float64)
    frames = _read_frames_bilinear(_G['srcs'], fin.crs, tr, c1 - c0, r1 - r0,
                                   rasterio.windows.bounds(win, fin.transform))
    if not frames:
        return k, 0.0
    a = step_strength(F, frames)
    band = ndi.binary_dilation(seam, iterations=2) & (a > 0)
    return k, (float(a[band].max()) if band.any() else 0.0)


# ---------------------------------------------------------------------------
# Gradient-domain splice
# ---------------------------------------------------------------------------

def grad(a):
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, :-1] = a[:, 1:] - a[:, :-1]
    gy[:-1, :] = a[1:, :] - a[:-1, :]
    return gx, gy


def div(gx, gy):
    """Negative adjoint of grad, zero flux at the border."""
    d = np.zeros_like(gx)
    d[:, 1:] += gx[:, 1:] - gx[:, :-1]
    d[:, 0] += gx[:, 0]
    d[1:, :] += gy[1:, :] - gy[:-1, :]
    d[0, :] += gy[0, :]
    return d


def poisson(Gx, Gy, prior, lam, iters=600, tol=1e-6):
    """argmin ||grad P - G||^2 + lam ||P - prior||^2 by conjugate gradients."""
    H, W = prior.shape

    def matvec(v):
        m = v.reshape(H, W)
        gx, gy = grad(m)
        return (lam * m - div(gx, gy)).ravel()

    op = LinearOperator((H * W, H * W), matvec=matvec, dtype=np.float64)
    rhs = (lam * prior - div(Gx, Gy)).ravel()
    try:
        P, _ = cg(op, rhs, x0=prior.ravel().copy(), rtol=tol, maxiter=iters)
    except TypeError:                                     # SciPy < 1.12
        P, _ = cg(op, rhs, x0=prior.ravel().copy(), tol=tol, maxiter=iters)
    return P.reshape(H, W)


def splice_window(F, srcs, crs, win_tr, wb, seam, lam=1e-4, band_px=160, feather=24):
    """Returns (spliced window, step strength before, after)."""
    th, tw = F.shape
    Fv = np.isfinite(F) & (F != NODATA)
    Gx, Gy = np.zeros((th, tw)), np.zeros((th, tw))
    Nx, Ny = np.zeros((th, tw)), np.zeros((th, tw))
    frames = _read_frames_bilinear(srcs, crs, win_tr, tw, th, wb)
    for A, ok in frames:
        gx, gy = grad(np.where(ok, A, 0.0))
        vx = ok.copy()
        vx[:, :-1] &= ok[:, 1:]
        vy = ok.copy()
        vy[:-1, :] &= ok[1:, :]
        Gx += np.where(vx, gx, 0.0)
        Nx += vx
        Gy += np.where(vy, gy, 0.0)
        Ny += vy
    aF_before = step_strength(F, frames)
    Gx = np.where(Nx > 0, Gx / np.maximum(Nx, 1), 0.0)
    Gy = np.where(Ny > 0, Gy / np.maximum(Ny, 1), 0.0)
    P = poisson(Gx, Gy, np.where(Fv, F, 0.0), lam)

    valid = Fv & ((Nx > 0) | (Ny > 0))
    band = ndi.binary_dilation(seam, iterations=band_px) & valid
    inner = band & ndi.binary_dilation(seam, iterations=1)
    outer = band & ndi.binary_dilation(valid & ~band, iterations=1)
    if not (inner.any() and outer.any()):
        return F.copy(), aF_before, aF_before
    use_p = cut_side(P, F, band, inner, outer, 1.0, 1.0)
    alpha = feathered_alpha(seam | (band & use_p), feather).astype(np.float64)
    region = ndi.binary_dilation(band, iterations=feather + 1)
    alpha = np.where(region & valid, alpha, 0.0)
    alpha = np.where(seam, 1.0, alpha)
    out = np.where(valid, alpha * P + (1.0 - alpha) * F, F)
    return out, aF_before, step_strength(out, frames)


def _splice_site(job):
    r0, c0, r1, c1, halo = job
    fin, sm, srcs = _G['fin'], _G['sm'], _G['srcs']
    H, W = fin.height, fin.width
    hr0, hr1, hc0, hc1 = max(0, r0 - halo), min(H, r1 + halo), max(0, c0 - halo), min(W, c1 + halo)
    win = Window(hc0, hr0, hc1 - hc0, hr1 - hr0)
    tr = rasterio.windows.transform(win, fin.transform)
    F = fin.read(1, window=win).astype(np.float64)
    seam = (sm.read(1, window=win) & 1) > 0
    if not seam.any():
        return r0, c0, r1, c1, None, None, 0.0, 0.0
    out, aF_F, aF_out = splice_window(F, srcs, fin.crs, tr, rasterio.windows.bounds(win, fin.transform), seam)
    s = (slice(r0 - hr0, r0 - hr0 + r1 - r0), slice(c0 - hc0, c0 - hc0 + c1 - c0))
    changed = np.abs(out[s] - F[s]) > 1e-4
    b = ndi.binary_dilation(seam[s], iterations=2) & (aF_F[s] > 0)
    mx_before = float(aF_F[s][b].max()) if b.any() else 0.0
    mx_after = float(aF_out[s][b].max()) if b.any() else 0.0
    return r0, c0, r1, c1, out[s].astype(np.float32), changed, mx_before, mx_after


def deseam(final_path, prov_path, levelled_paths, out_path, work_prefix, min_gate=8.0, improve=0.8,
           halo=512, jobs=16):
    """De-seam a tile. Also writes <work_prefix>_mask.tif, _domid.tif and
    _sites.csv. levelled_paths and the matching *_w.tif weights must be in the
    same (sorted) order."""
    weights = [p.replace('_leveled.tif', '_w.tif') for p in levelled_paths]
    seam_mask(prov_path, weights, work_prefix, jobs=jobs)
    sites = seam_sites(final_path, levelled_paths, jobs=jobs)
    mask = f'{work_prefix}_mask.tif'
    with rasterio.open(final_path) as fin:
        H, W = fin.height, fin.width
    # step strength of each site at full resolution: only real steps on a seam line are spliced
    jobs_gate = [(k, s['rows'][0], s['cols'][0], min(H, s['rows'][1]), min(W, s['cols'][1]))
                 for k, s in enumerate(sites)]
    with mp.get_context('fork').Pool(jobs, _init_gate, (final_path, levelled_paths, mask), maxtasksperchild=20) as pool:
        gate = dict(pool.imap_unordered(_gate_site, jobs_gate, chunksize=1))
    with open(f'{work_prefix}_sites.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'peak_mpx', 'af_base', 'row0', 'row1', 'col0', 'col1'])
        for k, s in enumerate(sites):
            w.writerow([k + 1, f"{s['peak']:.2f}", f'{gate[k]:.4f}', *s['rows'], *s['cols']])

    todo = [(s['rows'][0], s['cols'][0], min(H, s['rows'][1]), min(W, s['cols'][1]), halo)
            for k, s in enumerate(sites) if gate[k] >= min_gate]
    tmp = str(out_path) + '.partial.tif'
    shutil.copyfile(final_path, tmp)
    kept = rejected = 0
    pool = mp.get_context('fork').Pool(jobs, _init_gate, (final_path, levelled_paths, mask), maxtasksperchild=20)
    with rasterio.open(tmp, 'r+') as dst, pool:
        for r0, c0, r1, c1, core, changed, before, after in pool.imap(_splice_site, todo, chunksize=1):
            if core is None or not changed.any():
                continue
            if before > 0 and after < improve * before:
                win = Window(c0, r0, c1 - c0, r1 - r0)
                cur = dst.read(1, window=win)
                cur[changed] = core[changed]
                dst.write(cur, 1, window=win)
                kept += 1
            else:
                rejected += 1
    os.replace(tmp, out_path)
    print(f'{len(sites)} seam sites, {len(todo)} above {min_gate} m/px, {kept} spliced, {rejected} rejected')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('merge', help='join the infill to the levelled DEM')
    m.add_argument('--dem', required=True)
    m.add_argument('--model', required=True, help='infill surface (<prefix>.model.tif)')
    m.add_argument('--out', required=True)
    d = sub.add_parser('deseam', help='repair butt-joint steps')
    d.add_argument('--final', required=True, help='output of merge')
    d.add_argument('--prov', required=True, help='infill provenance (<prefix>.prov.tif)')
    d.add_argument('--frames', required=True, help="glob of the levelled DEMs, e.g. 'leveled/*_leveled.tif'")
    d.add_argument('--out', required=True)
    d.add_argument('--jobs', type=int, default=16)
    args = ap.parse_args()
    if args.cmd == 'merge':
        seamline_merge(args.dem, args.model, args.out)
    else:
        deseam(args.final, args.prov, sorted(glob.glob(args.frames)), args.out,
               str(args.out).removesuffix('.tif') + '_seams', jobs=args.jobs)


if __name__ == '__main__':
    main()
