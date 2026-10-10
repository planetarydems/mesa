"""Training of the two infill networks (Section 2.6 of the paper; Text S8).

census   real gap shapes: every gap of at least 30 px that does not touch the
         tile border, from the provenance masks of the levelled tiles, as an
         equivalent diameter, an elongation (ratio of the principal axes) and
         an orientation.
store    training windows cut from complete regions of the levelled tiles: for
         each window centre, the 3,072 px (55 km) window reduced to 512 px at
         108 m and, for the fine network, its central 512 px at 18 m (elevation,
         CTX mosaic and MOLA, with validity masks).
train    synthetic gaps are drawn from the census (an oriented capsule with a
         few bumps, up to 20 or 60 km long), cut into a window and restored.
         Loss: Huber (delta 1) on standardised elevation inside the gap; the
         fine network also on the visible pixels with weight 0.25.

    python -m mader.infill_train census --tiles-root tiles --out holes.npz
    python -m mader.infill_train store --dem tiles/X/leveled.tif --ctx tiles/X/ctx_18m.tif --mola MOLA.tif \
        --out store --tile X --split train --stride 1024 --coarse-only --min-valid 0.4
    python -m mader.infill_train train --stage coarse --store store --holes holes.npz --out runs/coarse ...

Training ran on two nodes with two V100 GPUs each (Lightning Fabric, 16-bit
mixed precision, batch 8 per GPU); see docs/pipeline.md for the exact runs.
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from .infill import (CFAC, CTX_NODATA, EMAX, EMIN, FOFF, MOAT_PX, NWIN, TILE, InfillUNet, coarse_to_fine,
                     decimate, normalise, normalise_ctx, plausible_range, robust_frame)

MINOR_FLOOR_M = 54.0       # a synthetic gap is at least about 3 px wide at 18 m


# ---------------------------------------------------------------------------
# Gap census
# ---------------------------------------------------------------------------

def census_tile(prov_path, min_px=30):
    """Shape of every interior gap of one tile's provenance mask (1 = infilled)."""
    with rasterio.open(prov_path) as ds:
        prov, res = ds.read(1), abs(ds.transform.a)
    lab, n = ndi.label(prov == 1, structure=np.ones((3, 3), int))
    out = {'diam_m': [], 'elong': [], 'theta_deg': [], 'area_px': []}
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    H, W = lab.shape
    for k, sl in enumerate(ndi.find_objects(lab)):
        if sizes[k] < min_px or sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue
        rr, cc = np.nonzero(lab[sl] == k + 1)
        cov = np.cov(np.stack([rr, cc]).astype(np.float64)) if rr.size > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        l2, l1 = max(float(w[0]), 1e-6), max(float(w[1]), 1e-6)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))            # 0 = along the rows (north-south)
        out['diam_m'].append(2.0 * np.sqrt(rr.size / np.pi) * res)
        out['elong'].append(np.sqrt(l1 / l2))
        out['theta_deg'].append(((ang + 90.0) % 180.0) - 90.0)
        out['area_px'].append(float(rr.size))
    return out


# ---------------------------------------------------------------------------
# Synthetic gaps
# ---------------------------------------------------------------------------

_GRID = {}


def _grid(res, size):
    key = (round(float(res), 6), int(size))
    if key not in _GRID:
        ax = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * float(res)
        _GRID[key] = np.meshgrid(ax, ax)
    return _GRID[key]


def _segment_distance(X, Y, A, B):
    abx, aby = B[0] - A[0], B[1] - A[1]
    denom = abx * abx + aby * aby
    if denom < 1e-9:
        return np.hypot(X - A[0], Y - A[1])
    t = np.clip(((X - A[0]) * abx + (Y - A[1]) * aby) / denom, 0.0, 1.0)
    return np.hypot(X - (A[0] + t * abx), Y - (A[1] + t * aby))


def load_hole_samples(path):
    z = np.load(path)
    d = np.asarray(z['diam_m'], float)
    return {'diam_m': d, 'elong': np.asarray(z['elong'], float) if 'elong' in z else np.ones(d.size),
            'theta_deg': np.asarray(z['theta_deg'], float) if 'theta_deg' in z else np.zeros(d.size)}


@dataclass
class Gap:
    """A capsule of length L and half-width r at angle theta from the rows,
    with a few disks along it, defined in metres so that the same gap can be
    drawn at 18 m and at 108 m."""
    L: float
    r: float
    theta: float
    bumps: np.ndarray

    @classmethod
    def sample(cls, rng, samples, max_diam_m=20000.0):
        k = int(rng.integers(0, samples['diam_m'].size))
        d = float(samples['diam_m'][k])
        e = float(max(samples['elong'][k], 1.0))
        major = d * math.sqrt(e)
        if major > max_diam_m:
            d, major = max_diam_m / math.sqrt(e), max_diam_m
        minor = max(d / math.sqrt(e), MINOR_FLOOR_M)
        L, r = max(major - minor, 0.0), minor / 2.0
        theta = math.radians(float(samples['theta_deg'][k]))
        theta = (1.0 if rng.random() < 0.5 else -1.0) * theta + math.radians(rng.uniform(-5.0, 5.0))
        K = int(rng.integers(3, 8))
        s = rng.uniform(-0.6 * L, 0.6 * L, size=K) if L > 0 else np.zeros(K)
        p = rng.uniform(-0.5 * r, 0.5 * r, size=K)
        rad = rng.uniform(0.4 * r, 1.0 * r, size=K)
        return cls(L=L, r=r, theta=theta, bumps=np.stack([s, p, rad], 1))

    def rasterize(self, res, size, shift=(0.0, 0.0)):
        X, Y = _grid(res, size)
        X, Y = X - float(shift[0]), Y - float(shift[1])
        ux, uy = math.sin(self.theta), math.cos(self.theta)
        px, py = -uy, ux
        A, B = (-0.5 * self.L * ux, -0.5 * self.L * uy), (0.5 * self.L * ux, 0.5 * self.L * uy)
        mask = _segment_distance(X, Y, A, B) <= self.r
        for s, p, rad in self.bumps:
            cx, cy = s * ux + p * px, s * uy + p * py
            mask |= ((X - cx) ** 2 + (Y - cy) ** 2) <= rad * rad
        return mask


# ---------------------------------------------------------------------------
# Tile store
# ---------------------------------------------------------------------------

_S, _P = {}, {}


def _store_window(arg):
    i, cr, cc = arg
    if not _S:
        _S['dem'] = rasterio.open(_P['dem'])
        _S['ctx'] = rasterio.open(_P['ctx'])
        _S['mola'] = WarpedVRT(rasterio.open(_P['mola']), crs=_S['dem'].crs, resampling=Resampling.bilinear)
    dem, ctx, mola = _S['dem'], _S['ctx'], _S['mola']
    win = Window(cc - NWIN // 2, cr - NWIN // 2, NWIN, NWIN)
    a_dem = dem.read(1, window=win).astype(np.float32)
    a_ctx = ctx.read(1, window=win).astype(np.float32)
    a_mola = mola.read(1, window=from_bounds(*dem.window_bounds(win), transform=mola.transform),
                       out_shape=(NWIN, NWIN), resampling=Resampling.bilinear).astype(np.float32)
    dem_ok = np.isfinite(a_dem) & (a_dem > EMIN) & (a_dem < EMAX)
    if dem.nodata is not None and np.isfinite(dem.nodata):
        dem_ok &= a_dem != dem.nodata
    ctx_ok = np.isfinite(a_ctx)
    if ctx.nodata is not None and np.isfinite(ctx.nodata):
        ctx_ok &= a_ctx != ctx.nodata
    sl = slice(FOFF, FOFF + TILE)
    coarse_only = _P['coarse_only']
    valid_frac = float(dem_ok.mean()) if coarse_only else float(dem_ok[sl, sl].mean())
    if valid_frac < _P['min_valid']:
        return i, None
    gap_frac = float((~dem_ok).mean()) if coarse_only else float((~dem_ok[sl, sl]).mean())
    c_dem, c_dv = decimate(a_dem, dem_ok)
    c_ctx, c_cv = decimate(a_ctx, ctx_ok)
    c_mola, _ = decimate(a_mola, np.ones_like(dem_ok))
    arrays = dict(c_dem=c_dem, c_dem_valid=c_dv.astype(np.uint8), c_ctx=c_ctx, c_ctx_valid=c_cv.astype(np.uint8),
                  c_mola=c_mola, center=np.array([cr, cc], np.int32),
                  fine_transform=np.array(window_transform(Window(cc - TILE // 2, cr - TILE // 2, TILE, TILE),
                                                           dem.transform)[:6], np.float64))
    if not coarse_only:
        arrays.update(f_dem=a_dem[sl, sl], f_dem_valid=dem_ok[sl, sl].astype(np.uint8), f_ctx=a_ctx[sl, sl],
                      f_ctx_valid=ctx_ok[sl, sl].astype(np.uint8), f_mola=a_mola[sl, sl])
    tid = f"{_P['prefix']}_{i:06d}"
    np.savez_compressed(_P['tiles'] / f'{tid}.tmp.npz', **arrays)
    os.replace(_P['tiles'] / f'{tid}.tmp.npz', _P['tiles'] / f'{tid}.npz')
    return i, (tid, cr, cc, valid_frac, gap_frac)


def build_store(dem, ctx, mola, out_dir, tile, split, stride, coarse_only=False, min_valid=0.5, jobs=None):
    """Cut training windows from one levelled tile; writes tiles/<tile>_NNNNNN.npz
    and manifest_<tile>.csv (tile_id, center_row, center_col, valid_frac,
    real_hole_frac, box, split)."""
    out = Path(out_dir)
    (out / 'tiles').mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem) as d:
        H, W = d.height, d.width
    centres = [(r, c) for r in range(NWIN // 2, H - NWIN // 2, stride) for c in range(NWIN // 2, W - NWIN // 2, stride)]
    _P.update(dem=dem, ctx=ctx, mola=mola, tiles=out / 'tiles', min_valid=min_valid, coarse_only=coarse_only,
              prefix=tile)
    kept = []
    with mp.get_context('fork').Pool(jobs or len(os.sched_getaffinity(0)), maxtasksperchild=1000) as pool:
        for _, res in pool.imap_unordered(_store_window, [(i, r, c) for i, (r, c) in enumerate(centres)], chunksize=8):
            if res is not None:
                kept.append(res)
    kept.sort()
    with open(out / f'manifest_{tile}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tile_id', 'center_row', 'center_col', 'valid_frac', 'real_hole_frac', 'box', 'split'])
        w.writerows([k + (tile, split) for k in kept])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GapDataset(Dataset):
    """Windows of one split (column `split` of the store's manifest.csv) with
    one or more synthetic gaps. Training draws fresh gaps every time; the
    validation gaps are fixed per window."""

    def __init__(self, store, hole_samples, split='train', max_holes=2, max_hole_m=20000.0, jitter_m=1500.0, seed=0):
        rows = [r for m in sorted(Path(store).glob('manifest_*.csv')) for r in csv.DictReader(open(m))]
        rows = sorted((r for r in rows if r['split'] == split
                       and (Path(store) / 'tiles' / f"{r['tile_id']}.npz").exists()), key=lambda r: r['tile_id'])
        self.paths = [str(Path(store) / 'tiles' / f"{r['tile_id']}.npz") for r in rows]
        self.samples = load_hole_samples(hole_samples)
        self.split, self.seed = split, seed
        self.max_holes, self.max_hole_m, self.jitter_m = max_holes, max_hole_m, jitter_m

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.split == 'train' else (self.seed * 1_000_003 + i) & 0x7fffffff)
        z = np.load(self.paths[i])
        fine = 'f_dem' in z.files
        c_dem, c_dv = z['c_dem'].astype(np.float32), z['c_dem_valid'].astype(bool)
        c_ctx, c_cv = z['c_ctx'].astype(np.float32), z['c_ctx_valid'].astype(bool)
        c_mola = z['c_mola'].astype(np.float32)
        c_cv &= c_ctx != CTX_NODATA
        if fine:
            f_dem, f_dv = z['f_dem'].astype(np.float32), z['f_dem_valid'].astype(bool)
            f_ctx, f_cv = z['f_ctx'].astype(np.float32), z['f_ctx_valid'].astype(bool)
            f_mola = z['f_mola'].astype(np.float32)
            f_cv &= f_ctx != CTX_NODATA
        rng_e = plausible_range(c_dem, c_dv, c_mola)
        if rng_e is not None:
            c_dv &= (c_dem > rng_e[0]) & (c_dem < rng_e[1])
            if fine:
                f_dv &= (f_dem > rng_e[0]) & (f_dem < rng_e[1])
        c_dve = ndi.binary_erosion(c_dv, iterations=max(1, MOAT_PX // CFAC))
        if fine:
            f_dve = ndi.binary_erosion(f_dv, iterations=MOAT_PX)

        # gaps at both resolutions from one metric shape; retry until a gap lies on real data
        hole_f = hole_c = None
        for _ in range(8):
            hf = np.zeros((TILE, TILE), bool)
            hc = np.zeros((TILE, TILE), bool)
            for _ in range(int(rng.integers(1, self.max_holes + 1))):
                gap = Gap.sample(rng, self.samples, self.max_hole_m)
                shift = rng.uniform(-self.jitter_m, self.jitter_m, size=2)
                if fine:
                    hf |= gap.rasterize(18.0, TILE, shift)
                hc |= gap.rasterize(108.0, TILE, shift)
            hole_f, hole_c = hf, hc
            if ((hf & f_dve) if fine else (hc & c_dve)).any():
                break
        c_in = c_dve & ~hole_c
        ring = c_dem[c_in]
        if ring.size < 64:
            ring = c_dem[c_dv]
        med, scale = robust_frame(ring)
        scale = float(np.clip(scale, 40.0, 300.0))
        c_dem_n = np.where(c_dv, normalise(c_dem, med, scale), 0.0).astype(np.float32)
        c_mola_n = normalise(c_mola, med, scale)
        x_c = np.stack([np.where(c_in, c_dem_n, 0.0).astype(np.float32), c_in.astype(np.float32),
                        normalise_ctx(c_ctx, c_cv), c_cv.astype(np.float32), c_mola_n], 0)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        out = {'x_c': t(x_c), 'mola_c_n': t(c_mola_n[None]), 'c_dem_gt_n': t(c_dem_n[None]),
               'hole_c': t((hole_c & c_dve)[None].astype(np.float32)),
               'med': torch.tensor(med, dtype=torch.float32), 'scale': torch.tensor(scale, dtype=torch.float32)}
        if fine:
            f_in = f_dve & ~hole_f
            f_dem_n = np.where(f_dv, normalise(f_dem, med, scale), 0.0).astype(np.float32)
            x_f = np.stack([np.where(f_in, f_dem_n, 0.0).astype(np.float32), f_in.astype(np.float32),
                            normalise_ctx(f_ctx, f_cv), f_cv.astype(np.float32), normalise(f_mola, med, scale)], 0)
            out.update({'x_f': t(x_f), 'f_dem_gt_n': t(f_dem_n[None]),
                        'hole_f': t((hole_f & f_dve)[None].astype(np.float32)),
                        'known_f': t(f_in[None].astype(np.float32))})
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def masked_huber(pred, gt, mask, delta=1.0):
    return (F.huber_loss(pred, gt, delta=delta, reduction='none') * mask).sum() / mask.sum().clamp_min(1.0)


def forward(stage, net, coarse_net, b):
    """(prediction, target, gap mask) of a batch. The fine network sees the
    image-side inputs without MOLA plus the frozen coarse network's prediction,
    which is also the base it adds its residual to."""
    if stage == 'coarse':
        return net(b['x_c'], b['mola_c_n']), b['c_dem_gt_n'], b['hole_c']
    with torch.no_grad():
        fill = coarse_to_fine(coarse_net(b['x_c'], b['mola_c_n'])).detach()
    x_f = torch.cat([b['x_f'][:, :4], fill], dim=1)
    return net(x_f, fill), b['f_dem_gt_n'], b['hole_f']


def train(args):
    from lightning.fabric import Fabric
    fabric = Fabric(accelerator='cuda', devices=args.devices, num_nodes=args.num_nodes,
                    strategy='ddp' if args.devices * args.num_nodes > 1 else 'auto', precision=args.precision)
    fabric.launch()
    out = Path(args.out)
    if fabric.global_rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    net = InfillUNet(5, dropout=0.1)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    net, opt = fabric.setup(net, opt)
    if args.init:
        fabric.load(args.init, {'model': net})
    coarse_net = None
    if args.stage == 'fine':
        coarse_net = InfillUNet(5, dropout=0.0)
        fabric.load(args.coarse, {'model': coarse_net})
        coarse_net = fabric.to_device(coarse_net).eval()
        for p in coarse_net.parameters():
            p.requires_grad_(False)

    def loader(split):
        ds = GapDataset(args.store, args.holes, split, args.max_holes, args.max_hole_m)
        return fabric.setup_dataloaders(DataLoader(ds, batch_size=args.batch, shuffle=split == 'train',
                                                   num_workers=args.workers, pin_memory=True, drop_last=True,
                                                   persistent_workers=args.workers > 0))
    train_dl, val_dl = loader('train'), loader('val')
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_dl)), eta_min=1e-6)
    best, stale = float('inf'), 0
    for epoch in range(args.epochs):
        net.train()
        for b in train_dl:
            opt.zero_grad()
            pred, gt, mask = forward(args.stage, net, coarse_net, b)
            loss = masked_huber(pred, gt, mask)
            if args.stage == 'fine' and args.known_weight > 0:
                loss = loss + args.known_weight * masked_huber(pred, gt, b['known_f'])
            fabric.backward(loss)
            opt.step()
            sched.step()
        # validation: MAE in metres over the gap pixels, averaged over windows with a gap
        net.eval()
        tot, cnt = torch.zeros((), device=fabric.device), torch.zeros((), device=fabric.device)
        with torch.no_grad():
            for b in val_dl:
                pred, gt, mask = forward(args.stage, net, coarse_net, b)
                n = mask.sum(dim=[1, 2, 3])
                mae = ((pred - gt).abs() * mask).sum(dim=[1, 2, 3]) / n.clamp_min(1.0) * b['scale']
                tot += mae[n > 0].sum()
                cnt += (n > 0).sum().float()
        val = (fabric.all_reduce(tot, reduce_op='sum') / fabric.all_reduce(cnt, reduce_op='sum').clamp_min(1.0)).item()
        improved = val < best - 0.02
        best = min(best, val) if improved else best
        stale = 0 if improved else stale + 1
        state = {'model': net, 'opt': opt, 'epoch': epoch, 'best': best}
        fabric.save(out / 'last.pt', state)
        if improved:
            fabric.save(out / 'best.pt', state)
        if fabric.global_rank == 0:
            print(f'epoch {epoch}: validation MAE {val:.3f} m{"  (best)" if improved else ""}', flush=True)
        if args.patience and stale >= args.patience:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('census')
    c.add_argument('--provs', nargs='+', required=True, help='provenance masks of the levelled tiles')
    c.add_argument('--out', required=True)
    s = sub.add_parser('store')
    s.add_argument('--dem', required=True)
    s.add_argument('--ctx', required=True)
    s.add_argument('--mola', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--tile', required=True)
    s.add_argument('--split', required=True)
    s.add_argument('--stride', type=int, required=True)
    s.add_argument('--coarse-only', action='store_true')
    s.add_argument('--min-valid', type=float, default=0.5)
    t = sub.add_parser('train')
    t.add_argument('--stage', choices=['coarse', 'fine'], required=True)
    t.add_argument('--store', required=True)
    t.add_argument('--holes', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--init', default=None, help='initial weights (fine-tuning)')
    t.add_argument('--coarse', default=None, help='frozen coarse network (fine stage)')
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--lr', type=float, default=5e-5)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--max-holes', type=int, default=2)
    t.add_argument('--max-hole-m', type=float, default=20000.0)
    t.add_argument('--known-weight', type=float, default=0.0)
    t.add_argument('--patience', type=int, default=10)
    t.add_argument('--devices', type=int, default=2)
    t.add_argument('--num-nodes', type=int, default=2)
    t.add_argument('--precision', default='16-mixed')
    t.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    if args.cmd == 'census':
        merged = {k: [] for k in ('diam_m', 'elong', 'theta_deg', 'area_px')}
        for p in sorted(args.provs):
            for k, v in census_tile(p).items():
                merged[k] += v
        np.savez(args.out, **{k: np.array(v) for k, v in merged.items()})
    elif args.cmd == 'store':
        build_store(args.dem, args.ctx, args.mola, args.out, args.tile, args.split, args.stride, args.coarse_only,
                    args.min_valid)
    else:
        train(args)


if __name__ == '__main__':
    main()
