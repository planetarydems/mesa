"""combfix.py -- estimate and subtract the MOLA-input comb that v3 copies at the poles (sandbox; products only read).

Model: inside an oriented band (wave-vector across the local meridian, i.e. bars along meridians; periods PMIN..PMAX px),
    band(v3) = a(x) * band(M) + b(x) * quad(M) + rest,
with M = the MOLA input exactly as the network saw it (the frame's mola_on_grid, or the mosaic sampled on the box grid),
quad() = the Hilbert transform of band() along the across-meridian direction (lets the fit absorb sub-pixel shifts of the copy),
and (a, b) fitted by Gaussian-weighted local least squares (sigma SIG px). The correction is
    C = ramp(|lat|) * w(R2) * clip(a band(M) + b quad(M)),
w(R2) = 0 below R2_LO, 1 above R2_HI (R2 = local fraction of band(v3) explained), |(a, b)| capped at KMAX; optional amplitude gate:
w *= 0 below A_LO, 1 above A_HI m of local RMS of band(M) (applies the correction only where the MOLA input's comb band is strong).
Ramp 0 at |lat| <= LAT0 and 1 at |lat| >= LAT1. Works tile-wise (TILE px, overlap OV) so the meridian direction is local.
Optional infill notch (NOTCH_FRAC > 0, needs notch_mask in [0, 1], e.g. the feathered infill mask): after the linear step, also
remove NOTCH_FRAC of what is left in the oriented band, gated by the local RMS of band(M) (0 below NOTCH_A_LO, 1 above NOTCH_A_HI m):
    C += NOTCH_FRAC * ramp * gate * notch_mask * (band(v3) - C_linear)        (NOTCH_FRAC = 0: identical to v2; NSUB must be 1)
Rule v3 (2026-10-30, session polar-filter):
  - k gate: w *= clip((KG_HI - |(a, b)|) / (KG_HI - KG_LO), 0, 1) (KG_HI = 0: off). A comb leak has a small |k|; where v3 and MOLA share real relief |k| is near 1 and the fit must not remove it (HiRISE: frames with
    |C| > 10 m, k < 0.45 the misfit falls 10-12 m, k >= 0.6 it rises 8.5 m). NOTCH_KG = 1 applies the same factor to the notch term.
  - limit(): box_correct.py bounds the total correction to LIM_T m after tile blending. limit(X, L) = X * min(1, L / envelope(X)), envelope = max |X| over 21 px (one comb period) smoothed with a 6 px Gaussian.
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

DEF = dict(PMIN=9.0, PMAX=45.0, HALF_ANG=20.0, SIG=48.0, R2_LO=0.25, R2_HI=0.5, KMAX=0.6, LAT0=74.0, LAT1=76.0,
           TILE=1024, OV=192, EDGE_DEG=5.0, EDGE_FRAC=0.15, A_LO=0.0, A_HI=0.0, PAD=128, QUAD=1, NSUB=1,
           NOTCH_FRAC=0.0, NOTCH_A_LO=2.0, NOTCH_A_HI=5.0, KG_LO=0.0, KG_HI=0.0, NOTCH_KG=0, LIM_T=0.0)


def limit(X, L):
    """Scale X down smoothly where its local envelope exceeds L (metres; float32 in, float32 out)."""
    env = gaussian_filter(maximum_filter(np.abs(X), size=21), 6.0)
    return X * np.minimum(1.0, L / np.maximum(env, 1e-6)).astype(np.float32)


def fill(z, m):
    if m.all(): return z.astype(np.float64)
    zf = np.where(m, z, np.nan).astype(np.float64)
    # fill gaps with a heavily smoothed version (keeps the band free of step artefacts at the edges)
    w = gaussian_filter(m.astype(np.float64), 16); v = gaussian_filter(np.where(m, z, 0).astype(np.float64), 16)
    sm = np.where(w > 1e-3, v / np.maximum(w, 1e-3), np.nanmedian(zf) if m.any() else 0.0)
    return np.where(m, z, sm)


def sector_filters(shape, phi, p):
    """Real band filter and its quadrature partner (as complex multipliers) for an n x m FFT.
    phi = direction (radians, in (col,row) pixel axes) of the across-meridian unit vector."""
    n, m = shape; ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(m), indexing="ij"); k = np.hypot(kx, ky)
    ang = np.arctan2(ky, kx); da = np.abs((ang - phi + np.pi / 2) % np.pi - np.pi / 2)         # angle to +-u
    k1, k2 = 1.0 / p["PMAX"], 1.0 / p["PMIN"]; e = 0.25                                            # raised-cosine edges (25 % in log k)
    lk = np.log(np.maximum(k, 1e-9)); a1, a2 = math.log(k1), math.log(k2); de = e * (a2 - a1)
    rad = np.clip(np.minimum((lk - (a1 - de / 2)) / de, ((a2 + de / 2) - lk) / de), 0, 1); rad = 0.5 - 0.5 * np.cos(np.pi * rad)
    h = math.radians(p["HALF_ANG"]); he = math.radians(p["EDGE_DEG"])
    angw = np.clip(((h + he / 2) - da) / he, 0, 1); angw = 0.5 - 0.5 * np.cos(np.pi * angw)
    Bf = rad * angw; Bf[k == 0] = 0
    sgn = np.sign(kx * math.cos(phi) + ky * math.sin(phi))
    Qf = -1j * sgn * Bf
    return Bf, Qf


def apply(z, filt):
    return np.real(np.fft.ifft2(np.fft.fft2(z) * filt))


def local_fit(BF, BM, QM, sig, valid, quad=True):
    g = lambda a: gaussian_filter(np.where(valid, a, 0.0), sig)
    Smm, Sqq, Smq = g(BM * BM), g(QM * QM), g(BM * QM)
    Sfm, Sfq, Sff = g(BF * BM), g(BF * QM), g(BF * BF)
    if quad:
        det = Smm * Sqq - Smq ** 2 + 1e-12
        a = (Sfm * Sqq - Sfq * Smq) / det; b = (Sfq * Smm - Sfm * Smq) / det
    else:                                   # in-phase only (ablation)
        a = Sfm / (Smm + 1e-12); b = np.zeros_like(a)
    R2 = np.clip((a * Sfm + b * Sfq) / (Sff + 1e-12), 0, 1)
    return a, b, R2, Smm


def meridian_phi(lat, lon):
    """direction of grad(lon) in (col,row) axes at the tile centre (across-meridian)."""
    n, m = lon.shape; i, j = n // 2, m // 2; d = min(32, n // 4, m // 4)
    dl_c = (((lon[i, j + d] - lon[i, j - d]) + 180) % 360 - 180) / (2 * d)
    dl_r = (((lon[i + d, j] - lon[i - d, j]) + 180) % 360 - 180) / (2 * d)
    return math.atan2(dl_r, dl_c)


def correct(Z, M, lat, lon, valid=None, p=None, return_parts=False, notch_mask=None):
    """Z = v3 (frame or box), M = MOLA input on the same grid, lat/lon pixel centres. Returns correction C (Z_fixed = Z - C)."""
    p = dict(DEF, **(p or {})); H, W = Z.shape
    vz = np.isfinite(Z) if valid is None else valid
    vm = np.isfinite(M); v = vz & vm
    C = np.zeros((H, W), np.float64); Wsum = np.zeros((H, W), np.float64)
    parts = {k: np.zeros((H, W), np.float64) for k in ("a", "b", "R2", "w", "N")} if return_parts else None
    T, OV = p["TILE"], p["OV"]; step = T - 2 * OV
    ramp = np.clip((np.abs(lat) - p["LAT0"]) / (p["LAT1"] - p["LAT0"]), 0, 1)
    rows = list(range(0, max(1, H - 2 * OV), step)) if H > T else [0]
    cols = list(range(0, max(1, W - 2 * OV), step)) if W > T else [0]
    for r0 in rows:
        for c0 in cols:
            r1, c1 = min(H, r0 + T), min(W, c0 + T); r0_, c0_ = max(0, r1 - T), max(0, c1 - T)
            sl = (slice(r0_, r1), slice(c0_, c1)); vv = v[sl]
            if vv.mean() < 0.02 or ramp[sl].max() <= 0: continue
            phi = meridian_phi(lat[sl], lon[sl])
            zt, mt = fill(Z[sl], vz[sl] & np.isfinite(Z[sl])), fill(M[sl], vm[sl]); P_ = int(p["PAD"]); vp = vv
            if P_:                                                                          # mirror halo: no FFT wrap-around at the tile edge
                zt, mt = np.pad(zt, P_, mode="reflect"), np.pad(mt, P_, mode="reflect"); vp = np.pad(vv, P_, mode="reflect")
            zt = zt - gaussian_filter(zt, 64); mt = mt - gaussian_filter(mt, 64)          # remove long waves before the FFT
            nsub = int(p["NSUB"]); edges = np.geomspace(p["PMIN"], p["PMAX"], nsub + 1); Ct = 0.0; Nt = 0.0; wk = 1.0
            for isub in range(nsub):                                                    # optional sub-bands: frequency-dependent leak
                ps = dict(p, PMIN=edges[isub], PMAX=edges[isub + 1]) if nsub > 1 else p
                Bf, Qf = sector_filters(zt.shape, phi, ps)
                BF, BM, QM = apply(zt, Bf), apply(mt, Bf), apply(mt, Qf)
                a, b, R2, Smm = local_fit(BF, BM, QM, p["SIG"], vp, quad=bool(p["QUAD"]))
                if P_:
                    cr = (slice(P_, P_ + vv.shape[0]), slice(P_, P_ + vv.shape[1])); BF, BM, QM, a, b, R2, Smm = (x[cr] for x in (BF, BM, QM, a, b, R2, Smm))
                kk = np.hypot(a, b); cap = np.where(kk > p["KMAX"], p["KMAX"] / np.maximum(kk, 1e-9), 1.0)
                w = np.clip((R2 - p["R2_LO"]) / (p["R2_HI"] - p["R2_LO"]), 0, 1)
                if p["KG_HI"] > 0: wk = np.clip((p["KG_HI"] - kk) / (p["KG_HI"] - p["KG_LO"]), 0, 1); w = w * wk
                if p["A_HI"] > 0:
                    amp = np.sqrt(np.maximum(Smm, 0) / np.maximum(gaussian_filter(vv.astype(np.float64), p["SIG"]), 1e-6))
                    w = w * np.clip((amp - p["A_LO"]) / (p["A_HI"] - p["A_LO"]), 0, 1)
                Ct = Ct + ramp[sl] * w * cap * (a * BM + b * QM)
            if p["NOTCH_FRAC"] > 0 and notch_mask is not None and nsub == 1 and float(notch_mask[sl].max()) > 0:
                amp_n = np.sqrt(np.maximum(Smm, 0) / np.maximum(gaussian_filter(vv.astype(np.float64), p["SIG"]), 1e-6))
                gate = np.clip((amp_n - p["NOTCH_A_LO"]) / (p["NOTCH_A_HI"] - p["NOTCH_A_LO"]), 0, 1)
                Nt = p["NOTCH_FRAC"] * ramp[sl] * gate * notch_mask[sl] * (BF - Ct)
                if p["NOTCH_KG"]: Nt = Nt * wk
                Ct = Ct + Nt
            # tile blending weight: cosine taper over the overlap
            tw = np.ones(vv.shape)
            for ax, (lo, hi, n_) in enumerate(((r0_, r1, H), (c0_, c1, W))):
                L = hi - lo; t = np.ones(L)
                if lo > 0: t[:OV] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(OV) + 0.5) / OV)
                if hi < n_: t[-OV:] = np.minimum(t[-OV:], 0.5 + 0.5 * np.cos(np.pi * (np.arange(OV) + 0.5) / OV))
                tw = tw * (t[:, None] if ax == 0 else t[None, :])
            C[sl] += tw * Ct; Wsum[sl] += tw
            if return_parts:
                for k_, a_ in (("a", a), ("b", b), ("R2", R2), ("w", w * cap * ramp[sl]), ("N", Nt)): parts[k_][sl] += tw * a_
    C = np.where(Wsum > 0, C / np.maximum(Wsum, 1e-9), 0.0); C[~vz] = 0.0
    if return_parts:
        for k_ in parts: parts[k_] = np.where(Wsum > 0, parts[k_] / np.maximum(Wsum, 1e-9), 0.0)
        return C.astype(np.float32), parts
    return C.astype(np.float32)
