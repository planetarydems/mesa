"""Image footprints and per-pixel image provenance of a product tile (the
footprint product of the release, Section 4.4 of the paper).

The surface of a tile is a feather blend of the levelled DEMs of the CTX images
that cover it (level.py). Where images overlap, a pixel is a weighted mean of
two or more predictions from images of different dates. This module records,
from the files the tile chain leaves behind, which images formed each pixel:

  <tile>_image_index.tif    int16, 18 m, Cloud Optimized GeoTIFF, no-data -1.
                            The index of the image whose levelled DEM carries
                            the largest blend weight at the pixel (the dominant
                            image; where two or more weights are equal, deep
                            inside an overlap, the first of them in index
                            order). -1 where no image has a weight: infilled
                            pixels and no-data. The index is `frame_index` of
                            the CSV. It is the `_domid.tif` raster of seams.py.
  <tile>_overlap_depth.tif  int16, 18 m, COG, no-data 0. The number of levelled
                            DEMs with valid data at the pixel, i.e. how many
                            images were averaged there (1 = one image).
  <tile>_frames.csv         one row per image index: the CTX product id, the
                            acquisition time (IMAGE_TIME of the CTX cumulative
                            index), the solar incidence and azimuth the network
                            was given, the index's emission, phase, solar
                            longitude, north and sub-solar azimuth and orbit
                            number, and the image's share of the tile: pixels it
                            covers, covers alone, is dominant at, and its share
                            of the blend weight summed over the predicted pixels.
  <tile>_footprints.gpkg    one polygon per image: the valid-data extent of its
                            levelled DEM, traced at 4 x 4 pixels and simplified
                            to 100 m, in planetocentric longitude and latitude on
                            the Mars sphere (IAU_2015:49900), with the columns
                            of the CSV.
  <tile>_frames_summary.json  totals and consistency checks (the dominant and
                            weight shares of the images each sum to one over the
                            predicted pixels that have a dominant image).

A pixel's dominant image is read from the index raster and its row of the CSV;
its number of contributing images from the depth raster. Infilled pixels
(provenance mask 1) come from no image.

    python -m mesa.footprints tile --tile E-030_N-10 --tile-dir tiles/E-030_N-10 \\
        --suns data/images.csv --cumindex cumindex.tab --out footprints/E-030_N-10
    python -m mesa.footprints merge --tiles 'footprints/*/*_footprints.gpkg' --out mesa_image_footprints.gpkg

The tile directory is the output of scripts/run_tile.sh: `deseam_seams_domid.tif`,
`infill.prov.tif` and `leveled/<image>_w.tif`; the index order is the sorted order
of the weight files (seams.py). A work directory with other names is given
explicitly (--domid, --prov, --frames-list, --weights-dir, --weights-suffix).
The cumulative index is the PDS table `cumindex.tab` with its label beside it.
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

PIXEL = 18.0
ROWS = 2048                 # rows per strip of the census
DECIM = 4                   # the footprint is traced at DECIM x DECIM pixels (72 m)
SIMPLIFY_M = 100.0
LAYER = 'image_footprints'
MARS_GEOGRAPHIC = 'IAU_2015:49900'      # Mars (2015) sphere, planetocentric, east longitude
INDEX_COLUMNS = ('PRODUCT_ID', 'IMAGE_TIME', 'INCIDENCE_ANGLE', 'EMISSION_ANGLE', 'PHASE_ANGLE', 'SOLAR_LONGITUDE',
                 'NORTH_AZIMUTH', 'SUB_SOLAR_AZIMUTH', 'ORBIT_NUMBER')
CSV_COLUMNS = ('tile', 'frame_index', 'image', 'product_id', 'image_time_utc', 'acquisition_date', 'incidence_deg',
               'sun_azimuth_deg', 'index_incidence_deg', 'emission_deg', 'phase_deg', 'solar_longitude_deg',
               'north_azimuth_deg', 'sub_solar_azimuth_deg', 'orbit_number', 'n_px_valid', 'n_px_covered',
               'n_sole_px', 'n_dominant_px', 'dominant_share', 'weight_share', 'in_mosaic', 'area_km2',
               'projection', 'note', 'weights_file')
SCHEMA = {'geometry': 'MultiPolygon',
          'properties': {'tile': 'str', 'frame_index': 'int', 'image': 'str', 'product_id': 'str',
                         'image_time_utc': 'str', 'acquisition_date': 'str', 'incidence_deg': 'float',
                         'sun_azimuth_deg': 'float', 'index_incidence_deg': 'float', 'emission_deg': 'float',
                         'phase_deg': 'float', 'solar_longitude_deg': 'float', 'north_azimuth_deg': 'float',
                         'sub_solar_azimuth_deg': 'float', 'orbit_number': 'int', 'n_px_valid': 'int64',
                         'n_px_covered': 'int64', 'n_sole_px': 'int64', 'n_dominant_px': 'int64',
                         'dominant_share': 'float', 'weight_share': 'float', 'in_mosaic': 'int',
                         'area_km2': 'float', 'projection': 'str', 'note': 'str'}}


# ---------------------------------------------------------------------------
# CTX cumulative index
# ---------------------------------------------------------------------------

def read_cumindex(tab_path, columns=INDEX_COLUMNS):
    """{image id (first 15 characters of PRODUCT_ID): {column: value}} from a
    PDS3 fixed-width table and its label (same name, .lbl). The first row of an
    image id is kept."""
    tab_path = Path(tab_path)
    lbl = tab_path.with_suffix('.lbl')
    if not lbl.exists():
        lbl = tab_path.with_suffix('.LBL')
    cols, cur = [], None
    for line in open(lbl, errors='replace'):
        s = line.strip()
        if s.startswith('OBJECT') and 'COLUMN' in s:
            cur = {}
        elif s.startswith('END_OBJECT') and cur is not None:
            cols.append(cur)
            cur = None
        elif cur is not None and '=' in s:
            k, v = [t.strip() for t in s.split('=', 1)]
            cur[k] = v.strip('"')
    spec = {c['NAME']: (int(c['START_BYTE']) - 1, int(c['BYTES'])) for c in cols if c.get('NAME') in columns}
    missing = [c for c in columns if c not in spec]
    if missing:
        raise ValueError(f'{lbl}: columns {missing} not found')
    index = {}
    with open(tab_path, 'rb') as f:
        for raw in f:
            line = raw.decode('ascii', 'replace')
            a, b = spec['PRODUCT_ID']
            pid = line[a:a + b].strip().strip('"')
            if len(pid) >= 15:
                index.setdefault(pid[:15], {k: line[a:a + b].strip().strip('"') for k, (a, b) in spec.items()})
    return index


def read_suns(path, tile):
    """{image: (incidence, azimuth)} of one tile from data/images.csv."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r['tile'] == tile:
            out[r['image']] = (float(r['incidence']), float(r['azimuth']))
    return out


# ---------------------------------------------------------------------------
# frames of a tile
# ---------------------------------------------------------------------------

def tile_frames(tile_dir=None, weights=None, frames_list=None, weights_dir=None, weights_suffix='_w.tif'):
    """[(image, weight raster path)] in index order: the sorted weight files of
    the tile (what seams.py indexed), or the lines of a frame list (first
    tab-separated field = image name) with the weights in weights_dir."""
    if frames_list:
        names = [l.split('\t')[0].strip() for l in open(frames_list) if l.strip()]
        wdir = Path(weights_dir or Path(frames_list).parent)
        return [(n, str(wdir / f'{n}{weights_suffix}')) for n in names]
    files = sorted(glob.glob(weights)) if weights else sorted(glob.glob(str(Path(tile_dir) / 'leveled' / f'*{weights_suffix}')))
    if not files:
        raise FileNotFoundError('no weight rasters')
    return [(Path(f).name.removesuffix(weights_suffix), f) for f in files]


def _open(path):
    return rasterio.open(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# census: shares of the images, overlap depth
# ---------------------------------------------------------------------------

def census(domid_path, prov_path, frames, depth_path, rows=ROWS):
    """Per-image pixel counts over the predicted pixels of the tile and the
    overlap-depth raster. Returns (counts, summary). counts[k] = dict of
    n_px_covered, n_sole_px, n_dominant_px, weight_sum for image k."""
    K = len(frames)
    t0 = time.time()
    with rasterio.open(prov_path) as PV, rasterio.open(domid_path) as DM:
        H, W, tr, crs = PV.height, PV.width, PV.transform, PV.crs
        if DM.shape != (H, W):
            raise ValueError('image index and provenance mask differ in size')
        src = [_open(p) for _, p in frames]
        bounds = []
        for ds in src:
            if ds is None:
                bounds.append(None)
            elif ds.crs != crs:
                bounds.append(rasterio.coords.BoundingBox(*rasterio.warp.transform_bounds(ds.crs, crs, *ds.bounds)))
            else:
                bounds.append(ds.bounds)
        dom_cnt = np.zeros(K, np.int64)
        cov_cnt = np.zeros(K, np.int64)
        sole_cnt = np.zeros(K, np.int64)
        wshare = np.zeros(K, np.float64)
        depth_hist = np.zeros(8, np.int64)          # predicted pixels with 0..6, >= 7 images
        n_model = n_noframe = n_tie = n_overlap = n_infill = 0
        profile = dict(driver='GTiff', height=H, width=W, count=1, dtype='int16', crs=crs, transform=tr, nodata=0,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256, bigtiff='if_safer')
        with rasterio.open(depth_path, 'w', **profile) as DP:
            for r0 in range(0, H, rows):
                r1 = min(H, r0 + rows)
                win = Window(0, r0, W, r1 - r0)
                pv = PV.read(1, window=win)
                dm = DM.read(1, window=win)
                model = pv == 0
                n_infill += int((pv == 1).sum())
                top, bottom, left, right = tr.f + r0 * tr.e, tr.f + r1 * tr.e, tr.c, tr.c + W * tr.a
                den = np.zeros((r1 - r0, W), np.float64)
                depth = np.zeros((r1 - r0, W), np.int16)
                wmax = np.zeros((r1 - r0, W), np.float64)
                nmax = np.zeros((r1 - r0, W), np.int16)
                cache = []
                for k in range(K):
                    b = bounds[k]
                    if b is None or b.right <= left or b.left >= right or b.top <= bottom or b.bottom >= top:
                        continue
                    c0 = max(0, int((b.left - left) / tr.a) - 1)
                    c1 = min(W, int((b.right - left) / tr.a) + 2)
                    if c1 <= c0:
                        continue
                    with WarpedVRT(src[k], crs=crs, transform=tr, width=W, height=H,
                                   resampling=Resampling.nearest) as v:
                        w = v.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
                    valid = np.isfinite(w) & (w > 0.0)
                    if src[k].nodata is not None:
                        valid &= w != src[k].nodata
                    w[~valid] = 0.0
                    den[:, c0:c1] += w
                    depth[:, c0:c1] += valid.astype(np.int16)
                    gt = w > wmax[:, c0:c1]
                    eq = valid & (w == wmax[:, c0:c1])
                    wmax[:, c0:c1][gt] = w[gt]
                    nmax[:, c0:c1][gt] = 1
                    nmax[:, c0:c1][eq & ~gt] += 1
                    cache.append((k, c0, c1, w))
                DP.write(depth, 1, window=win)
                nm = int(model.sum())
                n_model += nm
                if nm == 0:
                    continue
                ok = model & (dm >= 0) & (dm < K)
                n_noframe += int((model & ~ok).sum())
                dom_cnt += np.bincount(dm[ok].astype(np.int64), minlength=K)[:K]
                for k, c0, c1, w in cache:
                    m = model[:, c0:c1] & (w > 0)
                    d = den[:, c0:c1]
                    cov_cnt[k] += int(m.sum())
                    wshare[k] += float((w[m] / d[m]).sum())
                    sole_cnt[k] += int((m & (depth[:, c0:c1] == 1)).sum())
                depth_hist += np.bincount(np.minimum(depth[model], 7), minlength=8)[:8]
                n_tie += int((model & (nmax >= 2)).sum())
                n_overlap += int((model & (depth >= 2)).sum())
        for ds in src:
            if ds is not None:
                ds.close()
    counts = [dict(n_px_covered=int(cov_cnt[k]), n_sole_px=int(sole_cnt[k]), n_dominant_px=int(dom_cnt[k]),
                   weight_sum=float(wshare[k])) for k in range(K)]
    nm = max(n_model, 1)
    summary = dict(n_images=K, n_predicted_px=int(n_model), n_infilled_px=int(n_infill),
                   predicted_px_without_dominant_image=int(n_noframe),
                   sum_dominant_share=float(dom_cnt.sum() / nm), sum_weight_share=float(wshare.sum() / nm),
                   overlap_depth_share={(str(i) if i < 7 else '>=7'): float(depth_hist[i] / nm) for i in range(8)},
                   share_px_with_2_or_more_images=float(depth_hist[2:].sum() / nm),
                   tied_largest_weight_share_of_predicted=float(n_tie / nm),
                   tied_largest_weight_share_of_overlap=float(n_tie / max(n_overlap, 1)),
                   census_seconds=round(time.time() - t0))
    return counts, summary


# ---------------------------------------------------------------------------
# footprint polygons
# ---------------------------------------------------------------------------

def valid_mask_pooled(ds, decim=DECIM, rows=ROWS):
    """(number of valid pixels, valid mask max-pooled over decim x decim
    pixels) of a weight raster, read in strips."""
    H, W = ds.height, ds.width
    h, w = math.ceil(H / decim), math.ceil(W / decim)
    pooled = np.zeros((h, w), bool)
    n = 0
    rows -= rows % decim
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        a = ds.read(1, window=Window(0, r0, W, r1 - r0))
        v = np.isfinite(a) & (a > 0)
        if ds.nodata is not None:
            v &= a != ds.nodata
        n += int(v.sum())
        ph, pw = (-v.shape[0]) % decim, (-v.shape[1]) % decim
        v = np.pad(v, ((0, ph), (0, pw)))
        blk = v.reshape(v.shape[0] // decim, decim, v.shape[1] // decim, decim).any(axis=(1, 3))
        pooled[r0 // decim:r0 // decim + blk.shape[0], :] |= blk
    return n, pooled


def footprint(ds, pooled, decim=DECIM, simplify_m=SIMPLIFY_M):
    """Valid-data extent of a weight raster as a shapely (Multi)Polygon in the
    raster's projection, traced from the pooled mask and simplified."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    v = pooled
    if not v.any():
        return None
    T = ds.transform * Affine.scale(decim, decim)
    polys = [shape(g) for g, val in shapes(v.astype(np.uint8), mask=v, transform=T) if val == 1]
    geom = unary_union(polys).simplify(simplify_m, preserve_topology=True)
    return geom if not geom.is_empty else None


def to_lonlat(geom, raster_crs):
    """Project a geometry from the raster's projection to planetocentric
    longitude and latitude on its sphere; longitudes in [-180, 180], a
    footprint crossing the antimeridian is split."""
    from pyproj import CRS, Transformer
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon
    from shapely.ops import clip_by_rect, transform, unary_union
    src = CRS.from_wkt(raster_crs.to_wkt())
    tf = Transformer.from_crs(src, src.geodetic_crs, always_xy=True)
    ref = tf.transform(geom.centroid.x, geom.centroid.y)[0]

    def unwrapped(x, y, z=None):
        lon, lat = tf.transform(x, y)
        lon = ref + (np.asarray(lon) - ref + 180.0) % 360.0 - 180.0     # continuous around the footprint's centre
        return lon, lat

    g = transform(unwrapped, geom)
    minx, _, maxx, _ = g.bounds
    if minx < -180.0 or maxx > 180.0:
        parts = [clip_by_rect(g, -180.0, -90.0, 180.0, 90.0),
                 translate(clip_by_rect(g, 180.0, -90.0, 540.0, 90.0), xoff=-360.0),
                 translate(clip_by_rect(g, -540.0, -90.0, -180.0, 90.0), xoff=360.0)]
        g = unary_union([p for p in parts if not p.is_empty])
    if g.geom_type == 'Polygon':
        g = MultiPolygon([g])
    return g


def area_km2(lonlat_geom):
    """Area on the Mars sphere: each part through a sinusoidal projection
    centred on it (equal-area)."""
    from pyproj import CRS, Transformer
    from shapely.ops import transform
    ll = CRS.from_proj4('+proj=longlat +R=3396190 +no_defs')
    total = 0.0
    for part in getattr(lonlat_geom, 'geoms', [lonlat_geom]):
        tf = Transformer.from_crs(ll, CRS.from_proj4(f'+proj=sinu +lon_0={part.centroid.x} +R=3396190 +units=m +no_defs'),
                                  always_xy=True)
        total += transform(tf.transform, part).area
    return total / 1e6


def geographic_crs_wkt():
    from pyproj import CRS
    try:
        return CRS.from_user_input(MARS_GEOGRAPHIC).to_wkt()
    except Exception:                                   # PROJ without the IAU registry
        return CRS.from_proj4('+proj=longlat +R=3396190 +no_defs').to_wkt()


def write_gpkg(path, records, layer=LAYER):
    """records: [(shapely geometry or None, properties dict)]."""
    import fiona
    from shapely.geometry import mapping
    if os.path.exists(path):
        os.remove(path)
    with fiona.open(path, 'w', driver='GPKG', layer=layer, schema=SCHEMA, crs_wkt=geographic_crs_wkt()) as dst:
        for geom, props in records:
            dst.write({'geometry': mapping(geom) if geom is not None else None,
                       'properties': {k: props.get(k) for k in SCHEMA['properties']}})


def cog(src, dst, gdal_bin='', title=None):
    """Deflate-compressed Cloud Optimized GeoTIFF with nearest-neighbour overviews."""
    cmd = [os.path.join(gdal_bin, 'gdal_translate'), '-of', 'COG', '-co', 'COMPRESS=DEFLATE', '-co', 'LEVEL=6',
           '-co', 'PREDICTOR=NO', '-co', 'BLOCKSIZE=512', '-co', 'BIGTIFF=IF_SAFER', '-co', 'NUM_THREADS=4',
           '-co', 'RESAMPLING=NEAREST', '-co', 'OVERVIEW_RESAMPLING=NEAREST', '-q']
    if title:
        cmd += ['-mo', f'TITLE={title}']
    subprocess.run(cmd + [str(src), str(dst)], check=True)


# ---------------------------------------------------------------------------
# one tile
# ---------------------------------------------------------------------------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tile_product(tile, frames, domid_path, prov_path, suns, cumindex, out_dir, gdal_bin='', simplify_m=SIMPLIFY_M):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    depth_tmp = out / f'.{tile}_overlap_depth.tmp.tif'
    counts, summary = census(domid_path, prov_path, frames, depth_tmp)
    n_model = max(summary['n_predicted_px'], 1)
    with rasterio.open(prov_path) as ref:
        crs = ref.crs
    from .geo import parse_crs
    projection = parse_crs(crs)[0]
    rows, records, no_date, no_sun, no_raster = [], [], [], [], []
    for k, (image, wpath) in enumerate(frames):
        ci = cumindex.get(image[:15], {})
        if not ci.get('IMAGE_TIME'):
            no_date.append(image)
        sun = suns.get(image)
        if sun is None:
            no_sun.append(image)
        ds = _open(wpath)
        geom = n_valid = None
        if ds is None:
            no_raster.append(image)
        else:
            with ds:
                n_valid, pooled = valid_mask_pooled(ds)
                g = footprint(ds, pooled, simplify_m=simplify_m)
                geom = to_lonlat(g, ds.crs) if g is not None else None
        c = counts[k]
        row = dict(tile=tile, frame_index=k, image=image, product_id=ci.get('PRODUCT_ID', ''),
                   image_time_utc=ci.get('IMAGE_TIME', ''), acquisition_date=(ci.get('IMAGE_TIME', '') or '')[:10],
                   incidence_deg=sun[0] if sun else None, sun_azimuth_deg=sun[1] if sun else None,
                   index_incidence_deg=_num(ci.get('INCIDENCE_ANGLE')), emission_deg=_num(ci.get('EMISSION_ANGLE')),
                   phase_deg=_num(ci.get('PHASE_ANGLE')), solar_longitude_deg=_num(ci.get('SOLAR_LONGITUDE')),
                   north_azimuth_deg=_num(ci.get('NORTH_AZIMUTH')), sub_solar_azimuth_deg=_num(ci.get('SUB_SOLAR_AZIMUTH')),
                   orbit_number=int(ci['ORBIT_NUMBER']) if ci.get('ORBIT_NUMBER', '').strip().isdigit() else None,
                   n_px_valid=n_valid, n_px_covered=c['n_px_covered'], n_sole_px=c['n_sole_px'],
                   n_dominant_px=c['n_dominant_px'], dominant_share=c['n_dominant_px'] / n_model,
                   weight_share=c['weight_sum'] / n_model, in_mosaic=1,
                   area_km2=area_km2(geom) if geom is not None else None, projection=projection, note='',
                   weights_file=os.path.relpath(wpath, out) if ds is not None else '')
        rows.append(row)
        records.append((geom, row))
    with open(out / f'{tile}_frames.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    write_gpkg(out / f'{tile}_footprints.gpkg', records)
    cog(domid_path, out / f'{tile}_image_index.tif', gdal_bin,
        f'{tile}: index of the dominant CTX image of each pixel (row frame_index of {tile}_frames.csv), -1 = none')
    cog(depth_tmp, out / f'{tile}_overlap_depth.tif', gdal_bin,
        f'{tile}: number of CTX images (levelled DEMs) blended at each pixel, 0 = none')
    os.remove(depth_tmp)
    dates = [r['acquisition_date'] for r in rows if r['acquisition_date']]
    summary.update(tile=tile, projection=projection, images_without_index_date=no_date, images_without_sun=no_sun,
                   images_without_weight_raster=no_raster,
                   date_range=[min(dates), max(dates)] if dates else None, seconds=round(time.time() - t0),
                   files=[f'{tile}_image_index.tif', f'{tile}_overlap_depth.tif', f'{tile}_frames.csv',
                          f'{tile}_footprints.gpkg'])
    json.dump(summary, open(out / f'{tile}_frames_summary.json', 'w'), indent=1)
    print(f"{tile}: {len(frames)} images, {summary['n_predicted_px']:,} predicted px, dominant shares sum "
          f"{summary['sum_dominant_share']:.4f}, weight shares sum {summary['sum_weight_share']:.4f}, "
          f"{100 * summary['share_px_with_2_or_more_images']:.1f}% of predicted px from 2+ images, "
          f"{len(no_date)} without index date, {len(no_sun)} without sun, {len(no_raster)} without weights; "
          f"{summary['seconds']} s", flush=True)
    return summary


def merge(gpkgs, out_path, layer=LAYER):
    """One GeoPackage from the per-tile files, in the order given."""
    import fiona
    if os.path.exists(out_path):
        os.remove(out_path)
    n = 0
    with fiona.open(out_path, 'w', driver='GPKG', layer=layer, schema=SCHEMA, crs_wkt=geographic_crs_wkt()) as dst:
        for p in gpkgs:
            with fiona.open(p, layer=layer) as src:
                for feat in src:
                    dst.write({'geometry': feat['geometry'],
                               'properties': {k: feat['properties'].get(k) for k in SCHEMA['properties']}})
                    n += 1
    print(f'{out_path}: {n} footprints from {len(gpkgs)} files')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tile', help='the footprint product of one tile')
    t.add_argument('--tile', required=True, help='tile name, e.g. E-030_N-10')
    t.add_argument('--tile-dir', default=None, help='output directory of scripts/run_tile.sh')
    t.add_argument('--domid', default=None, help='image-index raster (default <tile-dir>/deseam_seams_domid.tif)')
    t.add_argument('--prov', default=None, help='provenance mask (default <tile-dir>/infill.prov.tif)')
    t.add_argument('--weights', default=None, help="glob of the weight rasters (default '<tile-dir>/leveled/*_w.tif')")
    t.add_argument('--frames-list', default=None, help='text file, one image per line in index order')
    t.add_argument('--weights-dir', default=None, help='directory of the weight rasters of --frames-list')
    t.add_argument('--weights-suffix', default='_w.tif')
    t.add_argument('--suns', required=True, help='CSV tile,image,incidence,azimuth (data/images.csv)')
    t.add_argument('--cumindex', required=True, help='CTX cumulative index table (.tab, with its .lbl beside it)')
    t.add_argument('--out', required=True, help='output directory')
    t.add_argument('--simplify', type=float, default=SIMPLIFY_M, help='footprint simplification tolerance (m)')
    t.add_argument('--gdal-bin', default='')
    m = sub.add_parser('merge', help='join per-tile GeoPackages into one')
    m.add_argument('--tiles', nargs='+', required=True, help='globs of per-tile GeoPackages')
    m.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.cmd == 'tile':
        td = Path(args.tile_dir) if args.tile_dir else None
        domid = args.domid or str(td / 'deseam_seams_domid.tif')
        prov = args.prov or str(td / 'infill.prov.tif')
        frames = tile_frames(args.tile_dir, args.weights, args.frames_list, args.weights_dir, args.weights_suffix)
        tile_product(args.tile, frames, domid, prov, read_suns(args.suns, args.tile), read_cumindex(args.cumindex),
                     args.out, args.gdal_bin, args.simplify)
    else:
        files = sorted(f for g in args.tiles for f in glob.glob(g))
        merge(files, args.out)


if __name__ == '__main__':
    main()
