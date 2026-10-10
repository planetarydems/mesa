"""Registration of a map-projected CTX image to a reference image (Section 2.1
and Section 2.5 step 1 of the paper; Text S2).

Stage 1, similarity. Both images are read at about 2,500 px across (the
reference warped onto the image's grid), stretched between their 2nd and 98th
percentiles and matched with SIFT (Lowe, 2004), a ratio test of 0.8 and a
symmetric cross-check. Gross mismatches are removed by voting on the dominant
translation, and a similarity transform (translation, rotation, scale) is
fitted to the remaining matches with RANSAC. A fit needs 30 inliers, a rotation
below 5 deg and a scale within 5% of unity.

Stage 2, refinement. The image is warped with the stage-1 transform onto the
reference's grid (24 m), both are high-passed, and 256 px tiles at a stride of
128 px are phase-correlated (whitened, band-limited, upsampled for sub-pixel
peaks; Guizar-Sicairos et al., 2008). Translation, similarity and affine models
are fitted to this displacement field by Huber-weighted least squares, and the
model is chosen by two-fold cross-validation on blocks along the track
(Roberts et al., 2017); a richer model must lower the held-out residual by
10%. The refinement is applied only if it lowers the held-out residual below
the raw field's RMS, its linear part is below 1% and its translation below
12 px.

The result is written as a new geotransform: a VRT pointing at the unchanged
pixels of the input image.

    python -m mader.register --image mapped/B01_009956_2169_mapped.tif \
        --reference reference_24m.tif --out aligned/B01_009956_2169_aligned.vrt
"""
import argparse
import math
import os
import re

import cv2
import numpy as np
import rasterio
import scipy.fft as sfft
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

cv2.setNumThreads(1)


# ---------------------------------------------------------------------------
# Stage 1: SIFT similarity
# ---------------------------------------------------------------------------

def to_uint8(a, pct=(2, 98)):
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a) & (a > 0)
    if valid.sum() < 100:
        return None
    lo, hi = np.percentile(a[valid], pct)
    if hi <= lo:
        return None
    s = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    out = np.where(np.isfinite(s), s, 0).astype(np.uint8)
    out[~valid] = 0
    return out


def valid_mask(*arrays, erode_px=8):
    """Mutually valid area, eroded so that footprint edges cannot match."""
    m = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(a) & (a > 0)
    k = 2 * erode_px + 1
    return cv2.erode(m.astype(np.uint8) * 255, np.ones((k, k), np.uint8), iterations=1)


def shrink(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def sift_matches(img1, img2, mask1, mask2, ratio=0.80, max_features=20000, contrast=0.01):
    # The randomised k-d trees of FLANN draw from OpenCV's global generator.
    # Resetting it makes the matches of an image independent of the images
    # matched before it in the same process.
    cv2.setRNGSeed(0)
    det = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    kp1, d1 = det.detectAndCompute(img1, mask1)
    kp2, d2 = det.detectAndCompute(img2, mask2)
    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return kp1 or [], kp2 or [], []
    d1, d2 = d1.astype(np.float32, copy=False), d2.astype(np.float32, copy=False)
    good = [m[0] for m in matcher.knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if good:
        used = sorted({m.trainIdx for m in good})
        back = {used[i]: r.trainIdx for i, r in enumerate(matcher.match(np.stack([d2[i] for i in used]), d1))}
        good = [m for m in good if back.get(m.trainIdx) == m.queryIdx]
    return kp1, kp2, good


def translation_inliers(p1, p2, ransac_px=10.0, bin_px=3.0, min_inliers=8):
    """Matches near the dominant translation (2-D histogram vote)."""
    if len(p1) < min_inliers:
        return None
    off = (p2 - p1).astype(np.float64)
    nx = max(8, int(np.ceil(max(np.ptp(off[:, 0]), 1.0) / bin_px)))
    ny = max(8, int(np.ceil(max(np.ptp(off[:, 1]), 1.0) / bin_px)))
    hist, xe, ye = np.histogram2d(off[:, 0], off[:, 1], bins=[nx, ny])
    i, j = np.unravel_index(np.argmax(hist), hist.shape)
    peak = np.array([0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])])
    m = np.linalg.norm(off - peak, axis=1) < ransac_px
    if m.sum() < min_inliers:
        return None
    final = np.linalg.norm(off - np.median(off[m], axis=0), axis=1) < ransac_px
    return final if final.sum() >= min_inliers else None


def fit_similarity(p1, p2, ransac_px=10.0, max_rot_deg=5.0, max_scale_dev=0.05, min_inliers=30):
    """2x3 similarity mapping p1 -> p2, or ValueError."""
    t = translation_inliers(p1, p2, ransac_px, min_inliers=min(8, min_inliers))
    if t is None:
        raise ValueError('no dominant translation')
    p1, p2 = p1[t], p2[t]
    if len(p1) < min_inliers:
        raise ValueError(f'{len(p1)} translation inliers')
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
                                         maxIters=5000, confidence=0.99, refineIters=20)
    if M is None or inl is None:
        raise ValueError('no similarity')
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if abs(rot) > max_rot_deg or abs(scale - 1.0) > max_scale_dev:
        raise ValueError(f'rotation {rot:.2f} deg, scale {scale:.4f}')
    if int(inl.sum()) < min_inliers:
        raise ValueError(f'{int(inl.sum())} inliers')
    return M.astype(np.float64)


def ncc_translation(mov8, ref8, mask, max_shift_px, min_peak=0.20, max_plateau_r=8.0, min_verify_r=0.22):
    """Translation-only fallback for feature-poor images (used against the Global
    CTX Mosaic poleward of 30 deg): masked zero-normalised cross-correlation of
    the stretched images. Rejected unless the peak is at least 0.2, the
    near-peak plateau has a radius of at most 8 px and covers at most a quarter
    of the search, and the shifted images correlate with r >= 0.22. Returns the
    2x3 translation on the matching grid, or raises ValueError."""
    valid = (mask > 0) & (mov8 > 0)
    if valid.sum() < 2000:
        raise ValueError('fallback: too few valid pixels')
    ys, xs = np.where(valid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(max(8, min(max_shift_px, 256)))
    ry0, ry1 = max(0, y0 - pad), min(ref8.shape[0], y1 + pad)
    rx0, rx1 = max(0, x0 - pad), min(ref8.shape[1], x1 + pad)
    min_search = max(24, int(math.ceil(3.0 * max_plateau_r)))
    target = max(min_search, int(min(pad, 64)))

    def trim(t0, t1, r0, r1):      # shrink the template, never by more than 40%, to leave room to search
        short = 2 * target - ((r1 - r0) - (t1 - t0))
        if short <= 0:
            return t0, t1
        short = min(short, int((t1 - t0) * 0.40))
        return t0 + short // 2, t1 - (short - short // 2)

    ty0, ty1 = trim(y0, y1, ry0, ry1)
    tx0, tx1 = trim(x0, x1, rx0, rx1)
    if min((ry1 - ry0) - (ty1 - ty0), (rx1 - rx0) - (tx1 - tx0)) < 2 * min_search:
        raise ValueError('fallback: no room to search')
    if (ty1 - ty0) < 64 or (tx1 - tx0) < 64:
        raise ValueError('fallback: template too small')
    tmpl = mov8[ty0:ty1, tx0:tx1].astype(np.float32)
    tmask = valid[ty0:ty1, tx0:tx1]
    ref = ref8[ry0:ry1, rx0:rx1].astype(np.float32)
    if tmask.sum() < 2000:
        raise ValueError('fallback: too few valid template pixels')
    tm = tmpl[tmask]
    tmpl_n = np.where(tmask, (tmpl - tm.mean()) / max(tm.std(), 1e-6), 0.0).astype(np.float32)
    rvalid = ref > 0
    rv = ref[rvalid]
    if rv.size < 2000:
        raise ValueError('fallback: reference nearly empty')
    ref_n = np.where(rvalid, (ref - rv.mean()) / max(rv.std(), 1e-6), 0.0).astype(np.float32)
    res = cv2.matchTemplate(ref_n, tmpl_n, cv2.TM_CCORR_NORMED, mask=tmask.astype(np.uint8))
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, peak, _, (px, py) = cv2.minMaxLoc(res)
    dx, dy = float(rx0 + px - tx0), float(ry0 + py - ty0)
    plateau = int((res >= peak - 0.02).sum())
    if peak < min_peak or math.sqrt(plateau / math.pi) > max_plateau_r or plateau / max(res.size, 1) > 0.25:
        raise ValueError(f'fallback: peak {peak:.3f}, plateau {plateau} px')
    H, W = mov8.shape
    sdx, sdy = int(round(dx)), int(round(dy))
    my0, my1 = max(0, -sdy), min(H, ref8.shape[0] - sdy)
    mx0, mx1 = max(0, -sdx), min(W, ref8.shape[1] - sdx)
    if my1 - my0 < 50 or mx1 - mx0 < 50:
        raise ValueError('fallback: shifted overlap vanishes')
    a = mov8[my0:my1, mx0:mx1].astype(np.float32)
    b = ref8[my0 + sdy:my1 + sdy, mx0 + sdx:mx1 + sdx].astype(np.float32)
    m = valid[my0:my1, mx0:mx1] & (b > 0)
    if m.sum() < 2000:
        raise ValueError('fallback: too little shifted overlap')
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    if not np.isfinite(r) or r < min_verify_r:
        raise ValueError(f'fallback: shifted correlation {r:.3f}')
    try:                                   # sub-pixel step by phase correlation, at most 1.5 px
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (rdx, rdy), _ = cv2.phaseCorrelate(np.where(m, a, float(a[m].mean())).astype(np.float32),
                                           np.where(m, b, float(b[m].mean())).astype(np.float32), win)
        if abs(rdx) <= 1.5 and abs(rdy) <= 1.5:
            dx, dy = dx + rdx, dy + rdy
    except cv2.error:
        pass
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def coarse_fit(image, reference, max_dim=2500, ref_nodata=None, ref_offset=0.0, translation_fallback=False,
               fallback_max_shift_m=5000.0):
    """Stage 1: geotransform of the image after the similarity fit."""
    with rasterio.open(image) as src:
        crs, t, W, H = src.crs, src.transform, src.width, src.height
        dx_m = abs(t.a)
        dec = max(1, int(math.ceil(max(W, H) / max_dim)))
        dW, dH = int(math.ceil(W / dec)), int(math.ceil(H / dec))
        mov = src.read(1, out_shape=(dH, dW), resampling=Resampling.average).astype(np.float32)
    sx, sy = W / dW, H / dH
    with rasterio.open(reference) as rsrc:
        kw = dict(crs=crs, transform=t * Affine.scale(sx, sy), width=dW, height=dH, resampling=Resampling.cubic)
        if ref_nodata is not None:
            kw.update(src_nodata=ref_nodata, nodata=ref_nodata)
        with WarpedVRT(rsrc, **kw) as vrt:
            ref = vrt.read(1, masked=True)
    ref = np.where(ref.mask, np.nan, ref.data.astype(np.float32) + ref_offset).astype(np.float32)
    if float(((mov > 0) & np.isfinite(ref) & (ref > 0)).mean()) < 0.02:
        raise ValueError('image and reference hardly overlap')
    m8, r8 = to_uint8(mov), to_uint8(ref)
    if m8 is None or r8 is None:
        raise ValueError('too few valid pixels')
    mask = valid_mask(mov, ref)
    m_ds, scale = shrink(m8, max_dim)
    r_ds, _ = shrink(r8, max_dim)
    th, tw = m_ds.shape[:2]
    mask_ds = cv2.erode(cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    kp1, kp2, matches = sift_matches(m_ds, r_ds, mask_ds, mask_ds)
    try:
        if len(matches) < 8:
            raise ValueError(f'{len(matches)} matches')
        M = fit_similarity(np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32),
                           np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32))
    except ValueError:
        if not translation_fallback:
            raise
        M = ncc_translation(m_ds, r_ds, mask_ds, fallback_max_shift_m / max(dx_m * (sx / scale), 1e-6))
    # similarity on the matching grid -> full-resolution pixels of the image
    D = np.diag([sx / scale, sy / scale])
    S = np.eye(3)
    S[:2, :2] = D @ M[:, :2] @ np.linalg.inv(D)
    S[:2, 2] = D @ M[:, 2]
    T = np.array([[t.a, t.b, t.c], [t.d, t.e, t.f], [0, 0, 1]]) @ S
    return Affine(T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2])


# ---------------------------------------------------------------------------
# Stage 2: phase-correlation refinement
# ---------------------------------------------------------------------------

def _masked_gauss(z, m, sigma):
    num = ndimage.gaussian_filter(np.where(m, z, 0.0).astype(np.float32), sigma, mode='nearest')
    den = ndimage.gaussian_filter(m.astype(np.float32), sigma, mode='nearest')
    out = np.zeros_like(num)
    ok = den > 1e-3
    out[ok] = num[ok] / den[ok]
    return out


def highpass(z, m, sigma=4.0):
    f = np.where(m, z - _masked_gauss(z, m, sigma), 0.0).astype(np.float32)
    s = float(f[m].std()) if m.any() else 0.0
    if s > 0:
        np.clip(f, -5 * s, 5 * s, out=f)
        f /= s
    return f


def _lowpass_weight(shape, fc=0.25, roll=0.3):
    ny, nx = shape
    f = np.hypot(np.fft.fftfreq(ny)[:, None], np.fft.fftfreq(nx)[None, :])
    f0 = fc * (1 - roll)
    w = 0.5 * (1 + np.cos(np.pi * np.clip((f - f0) / max(fc - f0, 1e-9), 0, 1)))
    w[f >= fc] = 0
    return w.astype(np.float32)


def _upsampled_peak(Rf, coarse, upsample=20, region=1.5):
    ny, nx = Rf.shape
    n = int(np.ceil(region * upsample)) * 2 + 1
    sy = (np.arange(n) - n // 2) / upsample + coarse[0]
    sx = (np.arange(n) - n // 2) / upsample + coarse[1]
    Ey = np.exp(2j * np.pi * np.outer(sy, np.fft.fftfreq(ny)))
    Ex = np.exp(2j * np.pi * np.outer(np.fft.fftfreq(nx), sx))
    out = np.real(Ey @ Rf.astype(np.complex128) @ Ex) / (ny * nx)
    j = np.unravel_index(np.argmax(out), out.shape)
    return float(sy[j[0]]), float(sx[j[1]])


def phase_correlate(a, b, max_shift, fc=0.25):
    """(dx, dy, peak-to-sidelobe ratio): content of a at q belongs at q + (dx, dy) in b."""
    ny, nx = a.shape
    Rf = sfft.fft2(a) * np.conj(sfft.fft2(b))
    mag = np.abs(Rf)
    Rf = Rf / np.where(mag > 1e-12, mag, 1.0) * _lowpass_weight((ny, nx), fc)
    r = np.fft.fftshift(np.real(sfft.ifft2(Rf)))
    cy, cx = ny // 2, nx // 2
    s = int(max(2, min(max_shift, min(ny, nx) // 2 - 2)))
    sub = r[cy - s:cy + s + 1, cx - s:cx + s + 1]
    j = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[j])
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    side = sub[np.hypot(yy - j[0], xx - j[1]) > 3.0]
    psr = (peak - side.mean()) / max(side.std(), 1e-12) if side.size > 16 else 0.0
    dy, dx = float(j[0] + cy - s - cy), float(j[1] + cx - s - cx)
    try:
        dyu, dxu = _upsampled_peak(Rf, (dy, dx))
        if abs(dyu - dy) <= 1 and abs(dxu - dx) <= 1:
            dy, dx = dyu, dxu
    except Exception:
        pass
    return -dx, -dy, float(psr)


def displacement_field(mov, ref, tile=256, stride=128, sigma=4.0, max_shift=12, min_valid=0.85, erode_px=4):
    """Tile phase correlations -> rows of (x, y, dx, dy, psr) in pixels."""
    m = np.isfinite(mov) & np.isfinite(ref)
    k = 2 * erode_px + 1
    m = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    fa, fb = highpass(mov, m, sigma), highpass(ref, m, sigma)
    H, W = fa.shape
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    out = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            mm = m[y:y + tile, x:x + tile]
            if mm.mean() < min_valid:
                continue
            A, B = fa[y:y + tile, x:x + tile], fb[y:y + tile, x:x + tile]
            if A[mm].std() < 1e-6 or B[mm].std() < 1e-6:
                continue
            a = ((A - A[mm].mean()) * mm * win).astype(np.float32)
            b = ((B - B[mm].mean()) * mm * win).astype(np.float32)
            dx, dy, psr = phase_correlate(a, b, max_shift)
            out.append((x + tile / 2, y + tile / 2, dx, dy, psr))
    return np.array(out, float).reshape(-1, 5)


def _irls(A, b, w, iters=4, huber=1.0):
    p = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
    for _ in range(iters):
        r = b - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        ww = w * np.minimum(1.0, huber * s / np.maximum(np.abs(r), 1e-9))
        p = np.linalg.lstsq(A * ww[:, None], b * ww, rcond=None)[0]
    return p


MIN_TILES = {'trans': 3, 'sim': 6, 'affine': 12}


def fit_field(xy, d, w, kind):
    x, y = xy[:, 0], xy[:, 1]
    o, z = np.ones_like(x), np.zeros_like(x)
    if kind == 'trans':
        return np.array([_irls(o[:, None], d[:, 0], w)[0], _irls(o[:, None], d[:, 1], w)[0]])
    if kind == 'sim':          # dx = a + c x - s y, dy = b + s x + c y
        A = np.vstack([np.stack([o, z, x, -y], 1), np.stack([z, o, y, x], 1)])
        return _irls(A, np.concatenate([d[:, 0], d[:, 1]]), np.concatenate([w, w]))
    A = np.stack([o, x, y], 1)
    return np.concatenate([_irls(A, d[:, 0], w), _irls(A, d[:, 1], w)])


def predict_field(xy, p, kind):
    x, y = xy[:, 0], xy[:, 1]
    if kind == 'trans':
        return np.stack([np.full_like(x, p[0]), np.full_like(y, p[1])], 1)
    if kind == 'sim':
        a, b, c, s = p
        return np.stack([a + c * x - s * y, b + s * x + c * y], 1)
    return np.stack([p[0] + p[1] * x + p[2] * y, p[3] + p[4] * x + p[5] * y], 1)


def field_matrix(p, kind, centre):
    """Fitted parameters (coordinates centred and divided by 1000) -> [A | b]
    with d(q) = A q + b in grid pixels."""
    if kind == 'trans':
        return np.array([[0, 0, p[0]], [0, 0, p[1]]], float)
    if kind == 'sim':
        a, b, c, s = p
        A, bb = np.array([[c, -s], [s, c]]) / 1000.0, np.array([a, b])
    else:
        A, bb = np.array([[p[1], p[2]], [p[4], p[5]]]) / 1000.0, np.array([p[0], p[3]])
    return np.hstack([A, (bb - A @ centre)[:, None]])


def select_model(tiles, min_psr=8.0, block_tiles=4, stride=128):
    """Cross-validated choice among translation, similarity and affine."""
    good = tiles[tiles[:, 4] >= min_psr] if len(tiles) else tiles
    if len(good) < MIN_TILES['trans']:
        return None
    xy, d, w = good[:, :2], good[:, 2:4], np.ones(len(good))
    c = xy.mean(0)
    u = (xy - c) / 1000.0
    raw_rms = float(np.sqrt((np.hypot(d[:, 0], d[:, 1]) ** 2).mean()))
    _, _, vt = np.linalg.svd(u - u.mean(0), full_matrices=False)
    along = u @ vt[0]
    fold = np.floor((along - along.min()) / (block_tiles * stride * 1e-3 + 1e-12)).astype(int) % 2
    cv = {}
    for k in ('trans', 'sim', 'affine'):
        if len(good) < MIN_TILES[k]:
            continue
        errs, ok = [], True
        for f in (0, 1):
            tr, te = fold != f, fold == f
            if tr.sum() < MIN_TILES[k] or te.sum() < 1:
                ok = False
                break
            e = d[te] - predict_field(u[te], fit_field(u[tr], d[tr], w[tr], k), k)
            errs.append(np.hypot(e[:, 0], e[:, 1]))
        if ok:
            cv[k] = float(np.sqrt((np.concatenate(errs) ** 2).mean()))
    if not cv:
        return dict(kind='trans', M=field_matrix(fit_field(u, d, w, 'trans'), 'trans', c), cv=None, raw_rms=raw_rms)
    order = [k for k in ('trans', 'sim', 'affine') if k in cv]
    chosen = order[0]
    for k in order[1:]:
        if cv[k] < 0.9 * cv[chosen]:
            chosen = k
    return dict(kind=chosen, M=field_matrix(fit_field(u, d, w, chosen), chosen, c), cv=cv[chosen], raw_rms=raw_rms)


def refine(image, reference, T_coarse, ref_nodata=None, ref_offset=0.0, grid_px=None, max_shift=12, max_lin=1e-2):
    """Stage 2: refined geotransform, or None when the refinement is rejected."""
    with rasterio.open(reference) as rs:
        ref_px, rcrs, rt, rb = float(rs.transform.a), rs.crs, rs.transform, rs.bounds
        if grid_px:
            ref_px = float(grid_px)
            rt = Affine(ref_px, 0.0, rt.c, 0.0, -ref_px, rt.f)
        rnod = ref_nodata if ref_nodata is not None else rs.nodata
        with rasterio.open(image) as ms:
            crs, W, H, nod = ms.crs, ms.width, ms.height, ms.nodata
            dec = max(1, int(round(ref_px / math.hypot(ms.transform.a, ms.transform.b))))
            dW, dH = int(math.ceil(W / dec)), int(math.ceil(H / dec))
            mov = ms.read(1, out_shape=(dH, dW), resampling=Resampling.average).astype(np.float32)
        try:
            from pyproj import CRS as PCRS
            same = PCRS.from_wkt(rcrs.to_wkt()).equals(PCRS.from_wkt(crs.to_wkt()), ignore_axis_order=True)
        except Exception:
            same = rcrs == crs
        if not same:
            # different projections: build the analysis grid in the image's projection
            rb = rasterio.coords.BoundingBox(*transform_bounds(rcrs, crs, *rb, densify_pts=21))
            rt = Affine(ref_px, 0.0, math.floor(rb.left / ref_px) * ref_px,
                        0.0, -ref_px, math.ceil(rb.top / ref_px) * ref_px)
        ok = np.isfinite(mov) & (mov > 0)
        if nod is not None and np.isfinite(nod):
            ok &= mov != nod
        mov[~ok] = np.nan
        T2 = T_coarse * Affine.scale(W / dW, H / dH)
        cols, rows = np.array([0, dW, 0, dW], float), np.array([0, 0, dH, dH], float)
        X, Y = T2.a * cols + T2.b * rows + T2.c, T2.d * cols + T2.e * rows + T2.f
        xmin, xmax = max(X.min(), rb.left), min(X.max(), rb.right)
        ymin, ymax = max(Y.min(), rb.bottom), min(Y.max(), rb.top)
        if xmax - xmin < 256 * ref_px or ymax - ymin < 256 * ref_px:
            return None
        c0, c1 = math.floor((xmin - rt.c) / rt.a), math.ceil((xmax - rt.c) / rt.a)
        r0, r1 = math.floor((ymax - rt.f) / rt.e), math.ceil((ymin - rt.f) / rt.e)
        gw, gh = c1 - c0, r1 - r0
        G = Affine(rt.a, 0.0, rt.c + c0 * rt.a, 0.0, rt.e, rt.f + r0 * rt.e)
        moved = np.full((gh, gw), np.nan, np.float32)
        reproject(source=np.nan_to_num(mov, nan=0.0), destination=moved, src_transform=T2, src_crs=crs,
                  src_nodata=0.0, dst_transform=G, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, num_threads=1)
        moved[~(moved > 0)] = np.nan
        kw = dict(crs=crs, transform=G, width=gw, height=gh,
                  resampling=Resampling.average if grid_px else (Resampling.nearest if same else Resampling.bilinear))
        if rnod is not None:
            kw.update(src_nodata=rnod, nodata=rnod)
        with WarpedVRT(rs, **kw) as v:
            r = v.read(1, masked=True)
        refa = np.where(r.mask, np.nan, r.data.astype(np.float32) + ref_offset).astype(np.float32)
        refa[~(refa > 0)] = np.nan
    sel = select_model(displacement_field(moved, refa, max_shift=max_shift), stride=128)
    if sel is None:
        return None
    A, b = sel['M'][:, :2], sel['M'][:, 2]
    shift = np.hypot(*(b + A @ np.array([gw / 2, gh / 2])))
    if np.abs(A).max() > max_lin or shift > max_shift or (sel['cv'] is not None and sel['cv'] >= sel['raw_rms']):
        return None
    Rm = Affine(1 + A[0, 0], A[0, 1], b[0], A[1, 0], 1 + A[1, 1], b[1])
    return G * Rm * ~G * T_coarse


def write_vrt(src_path, dst_path, transform):
    """VRT carrying a new geotransform over the unchanged pixels of src_path."""
    with rasterio.open(src_path) as src:
        W, H, wkt, nd, dt = src.width, src.height, src.crs.to_wkt(), src.nodata, src.dtypes[0]
    gdal_type = {'float32': 'Float32', 'float64': 'Float64', 'uint8': 'Byte', 'int16': 'Int16',
                 'uint16': 'UInt16', 'int32': 'Int32'}[dt]
    t = transform
    gt = ','.join(repr(float(v)) for v in (t.c, t.a, t.b, t.f, t.d, t.e))
    ndx = '' if nd is None else f'<NoDataValue>{float(nd)!r}</NoDataValue>'
    xml = (f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}"><SRS>'
           f'{wkt.replace("&", "&amp;").replace("<", "&lt;")}</SRS><GeoTransform>{gt}</GeoTransform>'
           f'<VRTRasterBand dataType="{gdal_type}" band="1">{ndx}<SimpleSource><SourceFilename relativeToVRT="0">'
           f'{os.path.abspath(src_path)}</SourceFilename><SourceBand>1</SourceBand>'
           f'<SrcRect xOff="0" yOff="0" xSize="{W}" ySize="{H}"/><DstRect xOff="0" yOff="0" xSize="{W}" '
           f'ySize="{H}"/></SimpleSource></VRTRasterBand></VRTDataset>\n')
    with open(str(dst_path) + '.part', 'w') as f:
        f.write(xml)
    os.replace(str(dst_path) + '.part', dst_path)


def register(image, reference, out, ref_nodata=None, ref_offset=0.0, translation_fallback=False, refine_grid_px=None):
    T = coarse_fit(image, reference, ref_nodata=ref_nodata, ref_offset=ref_offset,
                   translation_fallback=translation_fallback)
    T_refined = refine(image, reference, T, ref_nodata=ref_nodata, ref_offset=ref_offset, grid_px=refine_grid_px)
    write_vrt(image, out, T_refined if T_refined is not None else T)
    return T, T_refined


# ---------------------------------------------------------------------------
# Correction of the Global CTX Mosaic onto the controlled mosaic (Text S8)
# ---------------------------------------------------------------------------

TREND_E, TREND_N = -1.3, 1.2      # drift of the offset with latitude, m per degree


def mosaic_corrections(fits, tiles):
    """Offsets (east, north, metres) that move the Global CTX Mosaic onto the
    controlled mosaic in tiles poleward of 30 deg, from the offsets measured in
    the equatorial tiles (`fits`: rows with tile, lon_sw, lat_sw, lon_c, lat_c,
    dxc_m_east, dyc_m_north; the offset at the tile centre).

    The first band beyond the equatorial tiles (30-40 deg N, 40-30 deg S) takes
    the offset of the adjacent equatorial tile in the same column plus the
    latitude drift; other tiles, and first-band tiles without a measured
    neighbour, take a fitted model a + b lon + Fourier(2) in lon + c lat, with
    the latitude clamped to +-35 deg beyond the first band."""
    lon_c = np.array([float(r['lon_c']) for r in fits])
    lat_c = np.array([float(r['lat_c']) for r in fits])
    dx = np.array([float(r['dxc_m_east']) for r in fits])
    dy = np.array([float(r['dyc_m_north']) for r in fits])

    def design(lon, lat):
        L = np.radians(lon)
        return np.column_stack([np.ones_like(lon), lon, np.sin(L), np.cos(L), np.sin(2 * L), np.cos(2 * L), lat])

    A = design(lon_c, lat_c)
    cx = np.linalg.lstsq(A, dx, rcond=None)[0]
    cy = np.linalg.lstsq(A, dy, rcond=None)[0]
    edge = {(int(r['lon_sw']), 1 if int(r['lat_sw']) == 20 else -1): (float(r['dxc_m_east']), float(r['dyc_m_north']))
            for r in fits if int(r['lat_sw']) in (20, -30)}
    out = {}
    for t in tiles:
        m = re.match(r'E([p+-])(\d{3})_N([p+-])(\d{2})$', t)
        lon_sw = (-1 if m.group(1) == '-' else 1) * int(m.group(2))
        lat_sw = (-1 if m.group(3) == '-' else 1) * int(m.group(4))
        if -30 <= lat_sw <= 20:
            continue
        lc, hemi = lat_sw + 5.0, (1 if lat_sw > 0 else -1)
        first = lat_sw == 30 if hemi > 0 else lat_sw == -40
        if first and (lon_sw, hemi) in edge:
            ex, ey = edge[(lon_sw, hemi)]
            out[t] = (ex + TREND_E * (lc - 25.0 * hemi), ey + TREND_N * (lc - 25.0 * hemi))
        else:
            q = design(np.array([((lon_sw + 5.0 + 180) % 360) - 180]), np.array([max(-35.0, min(35.0, lc))]))
            out[t] = (float(q @ cx), float(q @ cy))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='map-projected CTX image')
    ap.add_argument('--reference', required=True, help='reference image (orthoimage or CTX mosaic crop)')
    ap.add_argument('--out', required=True, help='output VRT')
    ap.add_argument('--ref-nodata', type=float, default=None)
    ap.add_argument('--ref-offset', type=float, default=0.0,
                    help='added to valid reference pixels, e.g. 32768 for the signed 16-bit controlled mosaic')
    ap.add_argument('--translation-fallback', action='store_true',
                    help='if the similarity fails, try a cross-correlation translation (polar tiles)')
    args = ap.parse_args()
    T, Tr = register(args.image, args.reference, args.out, args.ref_nodata, args.ref_offset,
                     args.translation_fallback)
    print(f'similarity: {tuple(T)[:6]}\nrefined: {tuple(Tr)[:6] if Tr is not None else "rejected"}')


if __name__ == '__main__':
    main()
