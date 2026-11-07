#!/usr/bin/env python3
"""make_pass2.py STAGE_COPY OUT : builds seam_pass2.py from a copy of r1-seam-install's seam_stage_v3.py by text patches (2026-10-30, seam-remaining), so the second pass follows the installed stage
exactly except for: header, defaults (lam 1e-3, band 64, feather 8, authority 0, min-off 20), --trim (lip trim, harmonic fill, prior tie released), --taper (rim taper), --skip-hard (default on: narrow pairs only,
r1-seam-install's 16:20 decision; --solve-hard turns it off). Every patch asserts on the exact text it replaces, so a stage copy that changed those lines stops the build instead of silently diverging."""
import sys
src, out = sys.argv[1:3]; s = open(src).read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"patch anchor not found ({s.count(old)} of {n}): {old[:90]!r}"
    s = s.replace(old, new)
rep('#!/usr/bin/env python3\n"""seam_stage_v3.py BOX SURFACE OUT.tif [options] :', '''#!/usr/bin/env python3
"""seam_pass2.py BOX SURFACE OUT.tif [options] : SECOND PASS of the v3 seam stage, drafted 2026-10-30 by seam-remaining (research; not installed). Built by make_pass2.py from r1-seam-install's seam_stage_v3.py
with these additions, run after the first pass (S32R / FR) on the narrow lines whose census offset is still >= --min-off (default 20 m):
  --trim N     : lip trim. Each of the line's two frames has its gradient distrusted within N px of its own data edge where it meets the other frame (the network's edge lip or droop, mechanism A of the
                 10-27 solutions report); G is filled harmonically there (sparse Laplace solve) where no other frame covers, and the prior tie is released there (lam 1e-4 inside the trim zone).
  --taper N    : rim taper (default 0: measured as identical to no taper at all 9 lab sites on 10-30; kept as an option only).
  --force-px N : force P out to N px from the seam before the min-cut starts (default 0 = the stage's 1.5 px). The cliff clause refuses big steps because the P -> F transition starts 3-4 px from the seam
                 while |P - F| is still 18-24 m; forcing P further out moves that transition to where |P - F| has decayed. Under test.
  --solve-hard : also solve hard (0-px) joints. Off by default: hard joints stay with the relaxation (r1-seam-install, 2026-10-30 16:20: every Poisson variant integrates a pixel-staircase joint into a chain).
  defaults     : lam 1e-3, band 64, feather 8, trim 32, taper 0, force-px 0, authority 0 (no PEDR in the solve; the PEDR guard clause stays), min-off 20. Everything else is the stage's. Evidence: seam_remaining_2026-10-30/REPORT.md.
Original header follows.
seam_stage_v3.py BOX SURFACE OUT.tif [options] :''')
HAS_SH = '"--solve-hard"' in s   # the stage copy of 16:30 (cf033b08) already has --solve-hard and leaves hard pairs to the relaxation itself
rep('ap.add_argument("--band-px", type=int, default=32); ap.add_argument("--feather", type=int, default=4); ap.add_argument("--lam", type=float, default=1e-2)',
    'ap.add_argument("--band-px", type=int, default=64); ap.add_argument("--feather", type=int, default=8); ap.add_argument("--lam", type=float, default=1e-3); ap.add_argument("--trim", type=int, default=32); ap.add_argument("--trim-soft", type=int, default=0, help="soften the trim boundary: frame weight ramps 0 -> 1 over N px beyond the trim zone and the prior tie ramps back over 2N px (0 = hard boundary)"); ap.add_argument("--taper", type=int, default=0); ap.add_argument("--reach-guard", action="store_true", help="keep the stage\'s reach clause (99th pct of |change| beyond 48 px) although the band is 64: off by default, the band bounds the reach by construction (replay_score: 0 px beyond 73 px)"); ap.add_argument("--force-px", type=float, default=0.0, help="force P (not the min-cut) out to this distance from the seam; 0 = the stage\'s 1.5 px")' + ('' if HAS_SH else '; ap.add_argument("--solve-hard", action="store_true", help="also solve hard (0-px) joints; default: narrow lines only, hard joints stay with the relaxation")'))
import re
n_auth = len(re.findall(r'"--authority", type=float, default=[0-9.e-]+', s)); assert n_auth == 1, f"authority anchor {n_auth}"
s = re.sub(r'"--authority", type=float, default=[0-9.e-]+', '"--authority", type=float, default=0.0', s)
rep('ap.add_argument("--min-off", type=float, default=2.0)', 'ap.add_argument("--min-off", type=float, default=20.0)')
rep('if a.band_px > 48:', 'if a.band_px > 48 and a.reach_guard:')   # the reach clause was written for a 32 px band; with band 64 it refuses chunks for changes 49-73 px out that the band allows (E-040_N-20 gate: 8 of 8 chunks on the crater-wall line)
rep('"--halo", type=int, default=64', '"--halo", type=int, default=96')   # the halo must reach band + feather + 1 = 73 px (the stage asserts it); 64 was the band-32 stage's default
rep('\ndef solve_line(li):', '''
def harmonic_fill(Gf, hole):
    """fill Gf inside `hole` with the Laplace solution from the surrounding values (sparse direct solve; Neumann where a hole touches the array edge)"""
    import scipy.sparse as sps, scipy.sparse.linalg as spl
    Hh, Ww = Gf.shape; idx = -np.ones((Hh, Ww), np.int64); ys, xs = np.nonzero(hole); n_ = len(ys)
    if n_ == 0: return Gf
    idx[ys, xs] = np.arange(n_); g = Gf.copy(); g[hole] = 0.0; rows, cols, vals = [], [], []; rhs = np.zeros(n_); diag = np.zeros(n_)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yn, xn = ys + dy, xs + dx; inb = (yn >= 0) & (yn < Hh) & (xn >= 0) & (xn < Ww); diag += inb
        j = np.where(inb, idx[np.clip(yn, 0, Hh - 1), np.clip(xn, 0, Ww - 1)], -1); m = inb & (j >= 0); rows.append(np.flatnonzero(m)); cols.append(j[m]); vals.append(-np.ones(int(m.sum())))
        k = inb & (j < 0); rhs[k] += g[yn[k], xn[k]]
    rows.append(np.arange(n_)); cols.append(np.arange(n_)); vals.append(np.maximum(diag, 1.0))
    Lap = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_, n_)); u = spl.spsolve(Lap.tocsc(), rhs); out = Gf.copy(); out[ys, xs] = u; return out


def solve_varlam(Gx, Gy, prior, lam_map, iters, tol):
    """(lam_map I - L) M = lam_map prior - div(G): the stage's solve with a per-pixel prior tie (released inside the trim zone)"""
    from scipy.sparse.linalg import LinearOperator, cg
    from poisson_mosaic import div, laplacian
    Hh, Ww = prior.shape; rhs = (lam_map * prior - div(Gx, Gy)).ravel()
    def mv(v): m_ = v.reshape(Hh, Ww); return (lam_map * m_ - laplacian(m_)).ravel()
    m, inf = cg(LinearOperator((Hh * Ww, Hh * Ww), matvec=mv, dtype=np.float64), rhs, x0=prior.ravel().copy(), rtol=tol, maxiter=iters); return m.reshape(Hh, Ww), inf


def solve_line(li):''')
rep('''        Gx, Gy, Nx, Ny = np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw))
        for k, A_, ok, wt in parts:
            f = FAC[k]; e = np.ones((th, tw)) if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull / a.demote_ramp, 0.0, 1.0); gx, gy = grad(np.where(ok, A_, 0.0)); vx = ok.copy(); vx[:, :-1] &= ok[:, 1:]; vx[:, -1] = False; vy = ok.copy(); vy[:-1, :] &= ok[1:, :]; vy[-1, :] = False
            Gx += np.where(vx, e * gx, 0.0); Nx += np.where(vx, e, 0.0); Gy += np.where(vy, e * gy, 0.0); Ny += np.where(vy, e, 0.0)''',
'''        Gx, Gy, Nx, Ny = np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw)); hole = np.zeros((th, tw), bool); kpq = (int(Q[0, 3]), int(Q[0, 4]))
        if a.trim > 0:   # distance to this line's hand-over pixels, for 'where the two frames meet'
            sl_ = np.ones((th, tw), bool); jp0, jq0 = Q[:, 5].astype(np.int64), Q[:, 6].astype(np.int64)
            for j in range(0, int((jp0 + jq0).max()) + 1):
                k_ = np.minimum(-jp0 + j, jq0); rr_ = Q[:, 1].astype(np.int64) + k_ * (Q[:, 0] == 0) - hr0; cc_ = Q[:, 2].astype(np.int64) + k_ * (Q[:, 0] == 1) - hc0; ok_ = (rr_ >= 0) & (rr_ < th) & (cc_ >= 0) & (cc_ < tw); sl_[rr_[ok_], cc_[ok_]] = False
            dl_ = ndi.distance_transform_edt(sl_)
        for k, A_, ok, wt in parts:
            f = FAC[k]; e = np.ones((th, tw)) if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull / a.demote_ramp, 0.0, 1.0); gx, gy = grad(np.where(ok, A_, 0.0)); vx = ok.copy(); vx[:, :-1] &= ok[:, 1:]; vx[:, -1] = False; vy = ok.copy(); vy[:-1, :] &= ok[1:, :]; vy[-1, :] = False
            if a.trim > 0 and k in kpq:
                ek_ = ndi.distance_transform_edt(ok); tz = ok & (ek_ <= a.trim) & (dl_ <= a.trim + 24); hole |= tz
                if a.trim_soft > 0: e = e * np.where(dl_ <= a.trim + 24 + a.trim_soft, np.clip((ek_ - a.trim) / float(a.trim_soft), 0.0, 1.0), 1.0)
                else: e = np.where(tz, 0.0, e)
            Gx += np.where(vx, e * gx, 0.0); Nx += np.where(vx, e, 0.0); Gy += np.where(vy, e * gy, 0.0); Ny += np.where(vy, e, 0.0)''')
rep('''        Gxf = np.where(Nx > 0, Mx, np.where(bx_, ex, fx)); Gyf = np.where(Ny > 0, My, np.where(by_, ey, fy)); Pz = None; rec.update(shots_fit=0, cg=-1)''',
'''        Gxf = np.where(Nx > 0, Mx, np.where(bx_, ex, fx)); Gyf = np.where(Ny > 0, My, np.where(by_, ey, fy)); Pz = None; rec.update(shots_fit=0, cg=-1)
        if a.trim > 0 and hole.any():
            hx = hole & (Nx == 0) & bx_; hy = hole & (Ny == 0) & by_
            if hx.any(): Gxf = harmonic_fill(Gxf, hx)
            if hy.any(): Gyf = harmonic_fill(Gyf, hy)
            lam_map = (1e-4 + (a.lam - 1e-4) * np.clip(ndi.distance_transform_edt(~hole) / float(2 * a.trim_soft), 0.0, 1.0)) if a.trim_soft > 0 else np.where(hole, 1e-4, a.lam)
            Pz, inf = solve_varlam(Gxf, Gyf, Fg, lam_map, a.iters, a.tol); rec.update(cg=int(inf), trim_px=int(hole.sum()))''')
rep('inner = band & (dseam <= 1.5)', 'inner = band & (dseam <= max(1.5, a.force_px))')
rep('''        alpha = np.where((dseam <= a.band_px + a.feather + 1) & valid, alpha, 0.0); alpha = np.where(seam & valid, 1.0, alpha); out = np.where(valid, alpha * Pz + (1.0 - alpha) * Fg, F); C = np.nan_to_num(out - F)''',
'''        alpha = np.where((dseam <= a.band_px + a.feather + 1) & valid, alpha, 0.0); alpha = np.where(seam & valid, 1.0, alpha)
        if a.taper > 0: e0_ = a.band_px + a.feather + 1; wg_ = np.clip((e0_ - dseam) / float(a.taper), 0.0, 1.0); alpha = alpha * (0.5 - 0.5 * np.cos(np.pi * wg_))
        out = np.where(valid, alpha * Pz + (1.0 - alpha) * Fg, F); C = np.nan_to_num(out - F)''')
# narrow only unless --solve-hard (the stage may already exclude hard pairs itself; this keeps pass 2 narrow-only whichever copy it is built from)
assert 'a.solve_hard' in s, "stage copy has --solve-hard but never reads it"
# pass 2 gates per 2.3 km chunk, not per line: a 60 km line with 2 km above 20 m has a median of a few metres and would never be solved; here the line is cut into the runs of chunks whose median |offset| is
# >= --min-off (one chunk of margin each side) and each run is solved as its own line, so the first pass's result is untouched elsewhere along the line
rep('gated = [l for l in LINES if l["off"] >= a.min_off]', '''def seg_lines(l):
    jx = l["ix"]; al = PS[jx, 1] if PS[jx[0], 0] == 1 else PS[jx, 2]; o_ = np.argsort(al, kind="stable"); jx = jx[o_]; al = al[o_]; cid = ((al - al.min()) // a.chunk_along).astype(np.int64); n = int(cid.max()) + 1
    med = np.array([np.nanmedian(np.abs(oS[jx[cid == c]])) if np.isfinite(oS[jx[cid == c]]).any() else 0.0 for c in range(n)]); hot = med >= a.min_off
    if not hot.any(): return []
    hot = ndi.binary_dilation(hot, iterations=1); out = []; c = 0
    while c < n:
        if not hot[c]: c += 1; continue
        d = c
        while d < n and hot[d]: d += 1
        kx = jx[(cid >= c) & (cid < d)]
        if len(kx) >= a.min_px: r_, c_ = PS[kx, 1].astype(int), PS[kx, 2].astype(int); mm = np.nanmedian(np.abs(oS[kx])); out.append(dict(ix=kx, r0=int(r_.min()), r1=int(r_.max()) + 1, c0=int(c_.min()), c1=int(c_.max()) + 1, off=float(mm) if np.isfinite(mm) else 0.0))
        c = d
    return out
gated = [sg for l in LINES for sg in seg_lines(l)]''')
open(out, "w").write(s); print("built", out, len(s))
