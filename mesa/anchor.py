"""Anchoring a predicted DEM to PEDR shots (Section 2.5 of the paper).

A prediction rescaled with MOLA's statistics is compared with the shots through
a Gaussian footprint (FOOTPRINT_M, 168 m full width at half maximum): the
prediction is averaged through the footprint at each shot, and a gain a and
offset b are fitted between these averages and the shot elevations by
Huber-weighted iteratively reweighted least squares. Because the footprint
kernel sums to one, averaging a*z + b equals a*average(z) + b, so the
prediction is convolved once and the fit is 1-D.

Anchoring, evaluation and leveling use the same nominal 168 m FWHM.
PRODUCT_FOOTPRINT_M is retained as an alias for callers of earlier versions.
Historical production comparisons are recorded in verification/README.md;
changing the configured footprint does not regenerate stored products.

Gates applied by the callers:
  * at least 10 shots, otherwise only the MOLA offset is applied;
  * the gain is fitted only if the shots span more than 20 m (2nd to 98th
    percentile); otherwise a = 1 and b is the median shot residual;
  * the fit is rejected in favour of the MOLA offset if its residual RMS
    exceeds 2 sigma, or if its offset differs from the MOLA offset by more
    than 3 sigma, where sigma is MOLA's standard deviation floored at 10 m.
"""
import numpy as np
from scipy.ndimage import correlate

NODATA = -9999.0
FOOTPRINT_M = 168.0             # full width at half maximum of the footprint (Section 2.5)
PRODUCT_FOOTPRINT_M = FOOTPRINT_M  # compatibility alias; use the common 168 m footprint
MIN_SHOTS = 10
MIN_SPREAD_M = 20.0
RESID_REL = 2.0
OFFSET_REL = 3.0
SIGMA_FLOOR_M = 10.0


def footprint_kernel(fwhm_m, pixel_size_m=18.0):
    """Normalised Gaussian footprint, truncated at 3 sigma."""
    sigma = (fwhm_m / 2.3548) / pixel_size_m
    r = max(1, int(np.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def footprint_average(dem, kernel, nodata=NODATA, min_valid_frac=0.5):
    """Footprint average of a DEM, ignoring no-data. Returns (average, support),
    where support is the kernel weight on valid pixels; the average is NaN where
    support < min_valid_frac."""
    valid = np.isfinite(dem) & (dem != nodata)
    num = correlate(np.where(valid, dem, 0.0).astype(np.float64), kernel, mode='constant', cval=0.0)
    den = correlate(valid.astype(np.float64), kernel, mode='constant', cval=0.0)
    avg = np.full_like(num, np.nan)
    ok = den >= min_valid_frac
    avg[ok] = num[ok] / den[ok]
    return avg, den


def huber_line_fit(x, y, n_iter=10, eps=1e-9):
    """Robust fit y ~ a x + b. The Huber threshold is 1.345 times the robust
    standard deviation (1.4826 MAD) of the least-squares residuals, at least 1 m."""
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (a * x + b)
    delta = max(1.345 * 1.4826 * np.median(np.abs(resid - np.median(resid))), 1.0)
    w = np.ones(x.size)
    for _ in range(n_iter):
        sw = np.sqrt(w)
        a, b = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        resid = y - (a * x + b)
        ar = np.abs(resid)
        w = np.where(ar <= delta, 1.0, delta / np.maximum(ar, eps))
    return a, b, resid


def fit_gain_offset(pred, rows, cols, elev, pixel_size_m=18.0, footprint_m=FOOTPRINT_M,
                    nodata=NODATA, min_support=0.5, min_shots=MIN_SHOTS):
    """Gain and offset of a prediction (metres) against shots at fractional
    pixel positions (rows, cols). Raises ValueError when fewer than min_shots
    shots have a well-supported footprint average or when the shots are flat.

    Returns a, b, the raw residual RMS, the shots' elevation spread (2nd to
    98th percentile) and the offset-only alternative (median of shot minus
    average)."""
    avg, support = footprint_average(np.asarray(pred, np.float64), footprint_kernel(footprint_m, pixel_size_m),
                                     nodata, min_support)
    H, W = avg.shape
    rr = np.rint(rows).astype(int)
    cc = np.rint(cols).astype(int)
    inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr, cc, e = rr[inside], cc[inside], np.asarray(elev, np.float64)[inside]
    p, s = avg[rr, cc], support[rr, cc]
    good = np.isfinite(p) & np.isfinite(e) & (s >= min_support)
    p, e = p[good], e[good]
    if p.size < min_shots:
        raise ValueError(f'{p.size} usable shots')
    spread = float(np.percentile(e, 98) - np.percentile(e, 2))
    if spread < 1.0:
        raise ValueError('flat shots')
    a, b, resid = huber_line_fit(p, e)
    return {'a': float(a), 'b': float(b), 'resid_rms': float(np.sqrt(np.mean(resid ** 2))),
            'spread': spread, 'offset_only': float(np.median(e - p)), 'n': int(p.size)}


def anchor_window(pred, valid, mola_mean, mola_std, rows, cols, elev, pixel_size_m=18.0,
                  footprint_m=FOOTPRINT_M):
    """Delivered surface of one evaluation window, anchored as in the product.

    pred is the prediction rescaled with MOLA's mean and standard deviation over
    the window, valid the pixels that are scored, and (rows, cols, elev) the
    shots inside the window. Without a usable fit the window is shifted so that
    its mean over the valid pixels equals MOLA's mean. Returns the surface and
    how it was anchored: 'affine', 'offset' (gain gate), 'rejected', 'fit_failed'
    or 'few_shots'.
    """
    pm = float(np.mean(pred[valid]))
    b_mola = mola_mean - pm
    if len(elev) < MIN_SHOTS:
        return pred + b_mola, 'few_shots'
    try:
        fit = fit_gain_offset(pred, rows, cols, elev, pixel_size_m, footprint_m)
    except ValueError:
        return pred + b_mola, 'fit_failed'
    a, b, how = fit['a'], fit['b'], 'affine'
    if fit['spread'] < MIN_SPREAD_M:
        a, b, how = 1.0, fit['offset_only'], 'offset'
    sigma = max(mola_std, SIGMA_FLOOR_M)
    if abs(a * pm + b - pm - b_mola) > OFFSET_REL * sigma or fit['resid_rms'] > RESID_REL * sigma:
        return pred + b_mola, 'rejected'
    return a * pred + b, how
