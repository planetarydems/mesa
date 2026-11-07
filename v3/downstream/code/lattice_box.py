#!/usr/bin/env python3
"""lattice_box.py BOX NAME=PATH [NAME=PATH ...] : the project's lattice metric (delattice_2026-10/code/lattice_axis.axis_stats) on box
surfaces, read only. N random 256 px windows (seed 0) inside the box's own cell (+2 km) where every surface is fully valid; per
surface: median axis16 / axis8 (power on the grid axes at the 16 / 8 px period relative to the diagonal ring; 1 = no lattice),
median amp16 (RMS of the excess sinusoids, m), share of windows with axis16 > 3. Use: v3 surface vs R1 deseam (before filter) and
R1 deseam_dlc2 (after filter 2.1) to decide whether the v3 box still needs the lattice filter. Env N (default 300)."""
import os, sys, numpy as np, rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
sys.path.insert(0, "/home/gridsan/groups/planetary_dtm/mars_dem/global_run/delattice_2026-10/code")
from lattice_axis import axis_stats
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; R1 = "ta_implicit_d1_implrel100s1_e088"
box = sys.argv[1]; N = int(os.environ.get("N", "300")); W = 256
KEY = dict(r1=f"{G}/global_run/boxes/{box}/box_run/box_dem_{R1}_deseam.tif", r1dlc2=f"{G}/global_run/boxes/{box}/box_run/box_dem_{R1}_deseam_dlc2.tif")
surf = []
for a in sys.argv[2:]:
    k, p = a.split("=", 1) if "=" in a else (a, KEY[a]); surf.append((k, KEY.get(p, p)))
LL = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
e, n_ = box.split("_"); lo0 = int(e[1:].replace("p", "+")); la0 = int(n_[1:].replace("p", "+"))
srcs = [(k, rasterio.open(p)) for k, p in surf]
ref = srcs[0][1]; tr = ref.transform; H, Wd = ref.shape; tf = Transformer.from_crs(ref.crs, LL, always_xy=True)
rng = np.random.default_rng(0); res = {k: [] for k, _ in srcs}; tried = 0
while len(res[srcs[0][0]]) < N and tried < 40 * N:
    tried += 1
    r0, c0 = int(rng.integers(0, H - W)), int(rng.integers(0, Wd - W))
    x, y = tr * (c0 + W / 2, r0 + W / 2); lon, lat = tf.transform(x, y)
    dl = ((lon - lo0 + 180) % 360) - 180; kx = 59.27 * max(np.cos(np.radians(lat)), 0.02)
    if not (min(dl, 10 - dl) * kx > -2 and min(lat - la0, la0 + 10 - lat) * 59.27 > -2): continue
    zs = []
    for k, s in srcs:
        if s.transform == tr and s.shape == (H, Wd): win = Window(c0, r0, W, W)
        else:                                            # other grid (R1): window around the same point
            cc, rr = ~s.transform * (x, y); win = Window(int(cc) - W // 2, int(rr) - W // 2, W, W)
        z = s.read(1, window=win, boundless=True, fill_value=s.nodata).astype(float)
        if not (np.isfinite(z).all() and (z > -1e4).all() and (s.nodata is None or (z != s.nodata).all())): break
        zs.append(z)
    if len(zs) < len(srcs): continue
    for (k, _), z in zip(srcs, zs): res[k].append(axis_stats(z))
for k, v in res.items():
    if not v: print(f"{box} {k}: no window"); continue
    a16 = np.array([d["axis16"] for d in v]); a8 = np.array([d["axis8"] for d in v]); m16 = np.array([d["amp16"] for d in v])
    print(f"{box} {k:10s} windows {len(v)} | axis16 median {np.nanmedian(a16):.2f} (share > 3: {np.mean(a16 > 3):.3f}) | axis8 median "
          f"{np.nanmedian(a8):.2f} | amp16 median {np.nanmedian(m16) * 100:.1f} cm", flush=True)
