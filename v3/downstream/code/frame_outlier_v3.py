#!/usr/bin/env python3
"""frame_outlier_v3.py BOX OUT_DIR [--parity all|even|odd] [--no-score] : per-frame gross-error check (2026-10-27, read only).
In the v3 QA it runs with --parity all --no-score: a REPORT of flagged tracks / polar frame edges, never a product change.

Detection (only CTX-derived pixels: prov 0, not replaced by the infill rule):
  track  : PEDR census shots of the box (orbit/segment masks applied) of the chosen orbit parity; the v3 surface (box_dem_<T3>_
           deseam.tif) footprint-smoothed (FWHM 168 m) at each shot; per orbit, shots ordered by latitude; running median of
           (surface - shot) over --win shots against the running median of the shot threshold max(--t-floor, 250 tan MOLA slope of
           the shot's 128 px census cell); a stretch of >= --min-run consecutive shots above it is a segment.
  edge   : every levelled frame (leveled_<T3>, own grid) vs its MOLA on the same grid (preds_<T3>/<stem>/mola_on_grid.tif):
           |Gaussian(8 px) frame - MOLA| > --t-mola within --edge-px px of the frame's own west/east data edge, components
           >= 200 px.
Region: on a box-grid window around each segment, pixels the segment's frame dominates (largest effective blend weight, the
  v3 leveller's formula), CTX-derived, deviating from MOLA by more than max(30, half the segment misfit) (edge: t-mola / 2),
  connected to the segment and within --grow px of it.
Replacement (for scoring only): blend of the other frames there (effective weights without the frame), else MOLA plus the
  surface-minus-MOLA residual of the region's rim interpolated inside (IDW); feathered in over 16 px.
Score: shots of the OTHER parity inside the region (held out), and all shots: |surface - shot| before vs after. A segment whose
  held-out shots get worse is counted as a false positive. Writes OUT_DIR/<BOX>_<parity>_segments.tsv (+ npz windows)."""
import argparse, csv, glob, json, math, os, sys, time, numpy as np, rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, map_coordinates, label, distance_transform_edt, binary_dilation
from pyproj import CRS, Transformer
os.environ.setdefault("PROJ_NETWORK", "OFF")
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; PC = f"{G}/global_run/fix_2026-10/r2_delivery/pedr_census"
V3D = os.environ.get("V3D", f"{G}/global_run/v3_2026-10/downstream")
ap = argparse.ArgumentParser()
ap.add_argument("box"); ap.add_argument("out_dir"); ap.add_argument("--parity", default="all", choices=["all", "even", "odd"])
ap.add_argument("--tag", default=os.environ.get("T3", "ta_implicit_d1_spline100s1_e098v3"))
ap.add_argument("--t-floor", type=float, default=50.0); ap.add_argument("--win", type=int, default=15); ap.add_argument("--min-run", type=int, default=10)
ap.add_argument("--t-mola", type=float, default=150.0); ap.add_argument("--edge-px", type=int, default=64); ap.add_argument("--grow", type=int, default=167)
ap.add_argument("--no-score", action="store_true", help="QA report mode: detect and size the regions only (no replacement, no scoring)")
ap.add_argument("--edge-all", action="store_true", help="MOLA edge check in every box (default: polar boxes only, |lat| >= 60)")
ap.add_argument("--np", type=int, default=int(os.environ.get("NP", "16")))
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
box = a.box; RD = f"{G}/global_run/boxes/{box}/box_run"; T3 = a.tag
SURF = f"{RD}/box_dem_{T3}_deseam.tif"; PROV = f"{RD}/box_dem_{T3}_filled.tif.prov.tif"; MIF = f"{RD}/box_dem_{T3}_deseam_mask.tif"
LL = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs"); SIG = 168.0 / 2.3548 / 18.0
with rasterio.open(SURF) as s: H, W, TR, CRS_ = s.height, s.width, s.transform, s.crs
FR = sorted(glob.glob(f"{RD}/leveled_{T3}/*_leveled.tif")); STEM = [os.path.basename(p).split("_pred")[0] for p in FR]
fac = {}
if os.path.exists(f"{V3D}/weights/{box}.csv"):
    for r in csv.DictReader(open(f"{V3D}/weights/{box}.csv")): fac[r["stem"]] = float(r["factor"])
FAC = [min(fac.get(s, 1.0), 1.0) for s in STEM]
BND = []
for p in FR:
    with rasterio.open(p) as s: BND.append(s.bounds)

# ---------------- shots ----------------
Z = dict(np.load(f"{PC}/shots/{box}.shots.npz")); n = len(Z["elev"]); drop = np.zeros(n, bool)
if os.path.exists(f"{PC}/cells/{box}.drop.npz"):
    D = np.load(f"{PC}/cells/{box}.drop.npz"); drop = (np.unpackbits(D["orbit"])[:n] | np.unpackbits(D["seg"])[:n]).astype(bool)
C = np.load(f"{PC}/cells/{box}.cells.npz"); ms = {(int(r), int(c)): float(m) for r, c, m in zip(C["crow"], C["ccol"], C["mslope"])}
x, y = Transformer.from_crs(LL, CRS_, always_xy=True).transform(Z["lon"].astype(float), Z["lat"].astype(float))
SC = (np.asarray(x) - TR.c) / TR.a - 0.5; SR = (np.asarray(y) - TR.f) / TR.e - 0.5
TSH = np.array([max(a.t_floor, 250.0 * math.tan(math.radians(ms.get((int(r) // 128, int(c) // 128), 0.0)))) for r, c in zip(Z["row"], Z["col"])])
ORB = Z["orbit"].astype(np.int64); EL = Z["elev"].astype(np.float64)


def fp_sample(z, r, c):
    m = np.isfinite(z); zs = gaussian_filter(np.where(m, z, 0.0), SIG); ws = gaussian_filter(m.astype(float), SIG)
    with np.errstate(all="ignore"): f = np.where(ws > 0.9, zs / ws, np.nan)
    ok = (r >= 0) & (r <= z.shape[0] - 2) & (c >= 0) & (c <= z.shape[1] - 2); v = np.full(len(r), np.nan)
    if ok.any():
        v[ok] = map_coordinates(np.where(np.isfinite(f), f, 0), [r[ok], c[ok]], order=1)
        vv = map_coordinates(np.isfinite(f).astype(float), [r[ok], c[ok]], order=1); v[np.flatnonzero(ok)[vv < 0.999]] = np.nan
    return v


def rd(path, win, dem=True):
    with rasterio.open(path) as s:
        z = s.read(1, window=win, boundless=True, fill_value=s.nodata if s.nodata is not None else 0).astype(np.float64); nd = s.nodata
    if not dem: return z
    m = np.isfinite(z) & (z > -1e4) & (z < 3e4)
    if nd is not None: m &= z != nd
    return np.where(m, z, np.nan)


def block(job):
    br, bc, h, w = job; M_ = 24
    sel = np.flatnonzero((SR >= br) & (SR < br + h) & (SC >= bc) & (SC < bc + w))
    if not len(sel): return sel, np.zeros(0), np.zeros(0)
    win = Window(bc - M_, br - M_, w + 2 * M_, h + 2 * M_)
    z = rd(SURF, win); pv = rd(PROV, win, dem=False); mv = rd(MIF, win, dem=False) if os.path.exists(MIF) else np.zeros_like(pv)
    v = fp_sample(z, SR[sel] - (br - M_), SC[sel] - (bc - M_))
    ri, ci = np.round(SR[sel] - (br - M_)).astype(int), np.round(SC[sel] - (bc - M_)).astype(int)
    ctx = (pv[ri, ci] == 0) & (mv[ri, ci] == 0)
    return sel, v, ctx


import multiprocessing as mp
SV = np.full(n, np.nan); CTX = np.zeros(n, bool)
jobs = [(br, bc, min(2048, H - br), min(2048, W - bc)) for br in range(0, H, 2048) for bc in range(0, W, 2048)]
with mp.get_context("fork").Pool(a.np) as pool:
    for sel, v, ctx in pool.imap_unordered(block, jobs): SV[sel] = v; CTX[sel] = ctx
RES = SV - EL
par = {"all": np.ones(n, bool), "even": ORB % 2 == 0, "odd": ORB % 2 == 1}[a.parity]
held = {"all": np.ones(n, bool), "even": ORB % 2 == 1, "odd": ORB % 2 == 0}[a.parity]
use = par & ~drop & CTX & np.isfinite(RES)
print(f"[{box}] shots {n}, detection set {use.sum()} ({a.parity}) ({time.time() - t0:.0f}s)", flush=True)

# ---------------- track segments ----------------
SEG = []
from numpy.lib.stride_tricks import sliding_window_view as swv
for o in np.unique(ORB[use]):
    ix = np.flatnonzero(use & (ORB == o))
    if len(ix) < a.win: continue
    ix = ix[np.argsort(Z["lat"][ix])]; h_ = a.win // 2
    rm = np.median(swv(RES[ix], a.win), 1); rt = np.median(swv(TSH[ix], a.win), 1)
    fl = np.zeros(len(ix), bool); fl[h_:h_ + len(rm)] = np.abs(rm) > rt
    lab_, nl = label(fl)
    for L_ in range(1, nl + 1):
        k = np.flatnonzero(lab_ == L_)
        if len(k) < a.min_run: continue
        s_ = ix[k]; SEG.append(dict(kind="track", orbit=int(o), shots=s_, misfit=float(np.median(RES[s_])), rows=SR[s_], cols=SC[s_]))
print(f"[{box}] track segments {len(SEG)} ({time.time() - t0:.0f}s)", flush=True)


# ---------------- edge segments (frame grid vs its MOLA) ----------------
def edge_frame(k):
    p = FR[k]; mo = f"{RD}/preds_{T3}/{STEM[k]}/mola_on_grid.tif"
    if not os.path.exists(mo): return []
    with rasterio.open(p) as s: z = s.read(1).astype(np.float64); nd = s.nodata; tr = s.transform
    with rasterio.open(mo) as s:
        if (s.height, s.width) != z.shape: return []
        m = s.read(1).astype(np.float64); mnd = s.nodata
    v = np.isfinite(z) & (z > -1e4) & (z < 3e4)
    if nd is not None: v &= z != nd
    vm = np.isfinite(m) & (m > -1e4) & (m < 3e4)
    if mnd is not None: vm &= m != mnd
    zs = gaussian_filter(np.where(v, z, 0), 8); ws = gaussian_filter(v.astype(float), 8)
    with np.errstate(all="ignore"): dev = np.where(v & vm & (ws > 0.5), zs / ws - m, np.nan)
    first = np.where(v.any(1), v.argmax(1), -1); last = np.where(v.any(1), v.shape[1] - 1 - v[:, ::-1].argmax(1), -1)
    cc = np.arange(v.shape[1])[None, :]
    ez = v & (first[:, None] >= 0) & ((cc - first[:, None] < a.edge_px) | (last[:, None] - cc < a.edge_px))
    fl = ez & (np.abs(np.nan_to_num(dev)) > a.t_mola)
    lab_, nl = label(fl, np.ones((3, 3))); out = []
    for L_ in range(1, nl + 1):
        rr, cc_ = np.nonzero(lab_ == L_)
        if len(rr) < 200: continue
        X, Y = tr * (cc_ + 0.5, rr + 0.5); bc = (np.asarray(X) - TR.c) / TR.a - 0.5; br = (np.asarray(Y) - TR.f) / TR.e - 0.5
        out.append(dict(kind="edge", frame=k, misfit=float(np.median(dev[rr, cc_])), rows=br[::7], cols=bc[::7], npx=len(rr)))
    return out


_la0 = int(box.split("_")[1][1:].replace("p", "+"))
POLAR = _la0 <= -70 or _la0 >= 60
if POLAR or a.edge_all:
    with mp.get_context("fork").Pool(a.np) as pool:
        for r in pool.imap_unordered(edge_frame, range(len(FR))): SEG += r
print(f"[{box}] segments total {len(SEG)} (edge {sum(s['kind'] == 'edge' for s in SEG)}) ({time.time() - t0:.0f}s)", flush=True)


# ---------------- region, replacement, score ----------------
def window_frames(r0, c0, S_):
    wtr = rasterio.windows.transform(Window(c0, r0, S_, S_), TR); wb = rasterio.windows.bounds(Window(c0, r0, S_, S_), TR)
    lev, wt, ks = [], [], []
    for k, b in enumerate(BND):
        if b.right <= wb[0] or b.left >= wb[2] or b.top <= wb[1] or b.bottom >= wb[3]: continue
        with rasterio.open(FR[k]) as s, WarpedVRT(s, crs=CRS_, transform=wtr, width=S_, height=S_, resampling=Resampling.bilinear, nodata=np.nan) as v:
            z = v.read(1).astype(np.float64); nd = s.nodata
        with rasterio.open(FR[k].replace("_leveled.tif", "_w.tif")) as s, WarpedVRT(s, crs=CRS_, transform=wtr, width=S_, height=S_, resampling=Resampling.bilinear, nodata=0) as v:
            w = v.read(1).astype(np.float64)
        ok = np.isfinite(z) & (w > 0)
        if nd is not None: ok &= z != nd
        if not ok.any(): continue
        lev.append(np.where(ok, z, np.nan)); wt.append(np.where(ok, w, 0.0)); ks.append(k)
    if not ks: return [], None, None
    wt = np.stack(wt); wfull = sum((wt[i] for i, k in enumerate(ks) if FAC[k] >= 1.0), np.zeros((S_, S_)))
    e = np.stack([wt[i] * (1.0 if FAC[k] >= 1.0 else FAC[k] + (1 - FAC[k]) * np.clip(1 - wfull / 512.0, 0, 1)) for i, k in enumerate(ks)])
    return ks, np.stack(lev), e


def mola_window(k, r0, c0, S_):
    mo = f"{RD}/preds_{T3}/{STEM[k]}/mola_on_grid.tif"; wtr = rasterio.windows.transform(Window(c0, r0, S_, S_), TR)
    with rasterio.open(mo) as s, WarpedVRT(s, crs=CRS_, transform=wtr, width=S_, height=S_, resampling=Resampling.bilinear, nodata=np.nan) as v:
        m = v.read(1).astype(np.float64)
    m[(m < -1e4) | (m > 3e4)] = np.nan; return m


def region(si):
    sg = SEG[si]
    rr, cc = np.asarray(sg["rows"]), np.asarray(sg["cols"])
    ext = int(max(rr.max() - rr.min(), cc.max() - cc.min()) / 2 + a.grow + 32); S_ = int(min(max(2 * ext, 512), 3072))
    r0, c0 = int(rr.mean()) - S_ // 2, int(cc.mean()) - S_ // 2
    ks, LEV, E = window_frames(r0, c0, S_)
    if not ks: return None
    win = Window(c0, r0, S_, S_); S0 = rd(SURF, win); pv = rd(PROV, win, dem=False)
    mv = rd(MIF, win, dem=False) if os.path.exists(MIF) else np.zeros_like(pv)
    tot = E.sum(0); dom = np.where(tot > 0, np.argmax(E, 0), -1)
    sr_, sc_ = np.clip(np.round(rr - r0).astype(int), 0, S_ - 1), np.clip(np.round(cc - c0).astype(int), 0, S_ - 1)
    if sg["kind"] == "track":
        dd = dom[sr_, sc_]; dd = dd[dd >= 0]
        if not len(dd): return None
        fi = int(np.bincount(dd).argmax()); k = ks[fi]
    else:
        k = sg["frame"]
        if k not in ks: return None
        fi = ks.index(k)
    M = mola_window(k, r0, c0, S_)
    ms_ = np.isfinite(S0); zs = gaussian_filter(np.where(ms_, S0, 0), 8); ws = gaussian_filter(ms_.astype(float), 8)
    with np.errstate(all="ignore"): dev = np.where(ms_ & (ws > 0.5), zs / ws - M, np.nan)
    thr = max(30.0, 0.5 * abs(sg["misfit"])) if sg["kind"] == "track" else a.t_mola / 2
    cand = (dom == fi) & (pv == 0) & (mv == 0) & (np.abs(np.nan_to_num(dev)) > thr) & (np.sign(np.nan_to_num(dev)) == np.sign(sg["misfit"]))
    seed = np.zeros((S_, S_), bool); seed[sr_, sc_] = True
    near = distance_transform_edt(~seed) <= a.grow
    lab_, nl = label(cand & near, np.ones((3, 3))); keep = np.unique(lab_[seed & cand]); keep = keep[keep > 0]
    reg = np.isin(lab_, keep)
    if reg.sum() < 50: return dict(seg=si, kind=sg["kind"], frame=STEM[k], factor=FAC[k], misfit=sg["misfit"], region_px=int(reg.sum()), note="no region")
    if a.no_score:
        so_ = E[[i for i in range(len(ks)) if i != fi]].sum(0) if len(ks) > 1 else np.zeros((S_, S_))
        la, lo_ = Transformer.from_crs(CRS_, LL, always_xy=True).transform(*(TR * (cc.mean() + 0.5, rr.mean() + 0.5)))[::-1]
        return dict(seg=si, kind=sg["kind"], orbit=sg.get("orbit", -1), frame=STEM[k], factor=FAC[k], misfit=round(sg["misfit"], 1),
                    lat=round(float(la), 3), lon=round(float(lo_), 3), row=int(rr.mean()), col=int(cc.mean()), region_px=int(reg.sum()),
                    other_frames_share=round(float((so_ > 0)[reg].mean()), 2), n_shots=len(sg["shots"]) if "shots" in sg else 0)
    # replacement
    oth = [i for i in range(len(ks)) if i != fi]
    if oth:
        eo = E[oth]; lo = LEV[oth]; so = eo.sum(0)
        with np.errstate(all="ignore"): blend_o = np.where(so > 0, np.nansum(eo * np.nan_to_num(lo), 0) / np.maximum(so, 1e-12), np.nan)
    else: blend_o = np.full((S_, S_), np.nan)
    rim = binary_dilation(reg, iterations=3) & ~reg & np.isfinite(S0) & np.isfinite(M)
    ry, rx = np.nonzero(rim); rres = (S0 - M)[rim]
    if len(ry) > 800: pick = np.random.default_rng(0).choice(len(ry), 800, replace=False); ry, rx, rres = ry[pick], rx[pick], rres[pick]
    gy, gx = np.nonzero(reg); rep = np.full((S_, S_), np.nan)
    if len(ry):
        out = np.empty(len(gy))
        for i0 in range(0, len(gy), 4096):
            d2 = (gy[i0:i0 + 4096, None] - ry[None]) ** 2 + (gx[i0:i0 + 4096, None] - rx[None]) ** 2; w_ = 1.0 / np.maximum(d2, 1.0)
            out[i0:i0 + 4096] = (w_ @ rres) / w_.sum(1)
        rep[gy, gx] = M[gy, gx] + out
    newv = np.where(np.isfinite(blend_o), blend_o, rep)
    al = np.clip(distance_transform_edt(reg) / 16.0, 0, 1)
    S1 = np.where(reg & np.isfinite(newv), al * newv + (1 - al) * S0, S0)
    # score
    inw = (SR >= r0) & (SR < r0 + S_ - 1) & (SC >= c0) & (SC < c0 + S_ - 1) & ~drop
    ri, ci = np.clip(np.round(SR - r0).astype(int), 0, S_ - 1), np.clip(np.round(SC - c0).astype(int), 0, S_ - 1)
    inr = inw.copy(); inr[inw] = reg[ri[inw], ci[inw]]
    def sc_(mask):
        idx = np.flatnonzero(mask)
        if not len(idx): return 0, np.nan, np.nan
        b0 = fp_sample(S0, SR[idx] - r0, SC[idx] - c0) - EL[idx]; b1 = fp_sample(S1, SR[idx] - r0, SC[idx] - c0) - EL[idx]
        ok = np.isfinite(b0) & np.isfinite(b1)
        return int(ok.sum()), float(np.mean(np.abs(b0[ok]))) if ok.any() else np.nan, float(np.mean(np.abs(b1[ok]))) if ok.any() else np.nan
    nh, h0, h1 = sc_(inr & held); na, a0_, a1_ = sc_(inr)
    verdict = "NA" if nh < 3 else ("TP" if h1 < h0 else "FP")
    la, lo_ = Transformer.from_crs(CRS_, LL, always_xy=True).transform(*(TR * (cc.mean() + 0.5, rr.mean() + 0.5)))[::-1]
    row = (dict(seg=si, kind=sg["kind"], orbit=sg.get("orbit", -1), frame=STEM[k], factor=FAC[k], misfit=round(sg["misfit"], 1),
                     lat=round(float(la), 3), lon=round(float(lo_), 3), row=int(rr.mean()), col=int(cc.mean()), region_px=int(reg.sum()),
                     other_frames_share=round(float(np.isfinite(blend_o)[reg].mean()), 2), held_n=nh, held_mae_before=round(h0, 2), held_mae_after=round(h1, 2),
                     all_n=na, all_mae_before=round(a0_, 2), all_mae_after=round(a1_, 2), verdict=verdict, max_change=round(float(np.nanmax(np.abs(S1 - S0))), 1)))
    if os.environ.get("SAVE"): np.savez_compressed(f"{a.out_dir}/{box}_{a.parity}_seg{si}.npz", r0=r0, c0=c0, S0=S0.astype(np.float32), S1=S1.astype(np.float32), reg=reg, M=M.astype(np.float32))
    return row


with mp.get_context("fork").Pool(a.np) as pool:
    rows = [r for r in pool.imap_unordered(region, range(len(SEG))) if r is not None]
rows.sort(key=lambda r: r["seg"])
keys = ["seg", "kind", "orbit", "frame", "factor", "misfit", "lat", "lon", "row", "col", "region_px", "other_frames_share", "n_shots", "held_n", "held_mae_before",
        "held_mae_after", "all_n", "all_mae_before", "all_mae_after", "verdict", "max_change", "note"]
with open(f"{a.out_dir}/{box}_{a.parity}_segments.tsv", "w") as fh:
    fh.write("\t".join(keys) + "\n")
    for r in rows: fh.write("\t".join(str(r.get(k_, "")) for k_ in keys) + "\n")
if a.no_score:
    v = [r for r in rows if "note" not in r]; v.sort(key=lambda r: -r["region_px"])
    print(f"OUTLIERS {box}: track {sum(r['kind'] == 'track' for r in v)}, polar edge {sum(r['kind'] == 'edge' for r in v)} regions, "
          f"{sum(r['region_px'] for r in v)} px; largest: " + "; ".join(f"{r['frame']} {r['kind']} {r['misfit']:+.0f} m {r['region_px']} px at "
          f"{r['lat']:.2f},{r['lon']:.2f}" for r in v[:3]) + f" (segments {len(SEG)}; report only, no product change)", flush=True)
    sys.exit(0)
v = [r for r in rows if "verdict" in r]
print(f"[{box}] {a.parity}: segments {len(SEG)}, regions {len(v)} ({sum(r['region_px'] for r in v)} px): TP {sum(r['verdict'] == 'TP' for r in v)} "
      f"FP {sum(r['verdict'] == 'FP' for r in v)} NA {sum(r['verdict'] == 'NA' for r in v)} ({time.time() - t0:.0f}s)", flush=True)
