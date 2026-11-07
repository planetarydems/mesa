#!/usr/bin/env python3
"""v3 COPY (2026-10-24) of fix_2026-10/rebuild/release2_input_guard.py: conservative input identity checks of the v3 level solve;
never changes scientific products. Changes: the frame-weight CSV comes from V3_WEIGHTS (default <V3D>/weights/<BOX>.csv) and the
solver is <V3D>/code/level_frames_box_v3.py (V3D = env V3D, default global_run/v3_2026-10/downstream)."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, '/home/gridsan/groups/planetary_dtm/mars_dem/global_run/fix_2026-10/rebuild')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release2_context_signature_v3 import validate_capture   # 2026-10-29: digest cache (was release2_context_signature)

BASE = Path('/home/gridsan/groups/planetary_dtm/mars_dem')
REBUILD = BASE / 'global_run/fix_2026-10/rebuild'
V3D = Path(os.environ.get('V3D', BASE / 'global_run/v3_2026-10/downstream'))
V3_WEIGHTS = Path(os.environ.get('V3_WEIGHTS', V3D / 'weights'))
_queue_cache = {}


def digest(path, algorithm='sha256'):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def identity(path):
    p = Path(path)
    s = p.stat()
    if not p.is_file() or s.st_size <= 0:
        raise ValueError(f'input is not a nonempty file: {p}')
    return dict(path=str(p), device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def queue_rows(queue, box):
    key = json.dumps(identity(queue), sort_keys=True)
    if key not in _queue_cache:
        with open(queue) as fh:
            grouped = {}
            for row in csv.DictReader(fh, delimiter='\t'):
                grouped.setdefault(row['box'], []).append(row)
        if json.dumps(identity(queue), sort_keys=True) != key:
            raise ValueError('queue changed while reading it')
        _queue_cache.clear()
        _queue_cache[key] = grouped
    rows = _queue_cache[key].get(box, [])
    if not rows or len({r['stem'] for r in rows}) != len(rows):
        raise ValueError('queue has no rows or duplicate frame stems for this tile')
    if any(not r['stem'] or '/' in r['stem'] or r['stem'] in ('.', '..') for r in rows):
        raise ValueError('invalid frame stem')
    return sorted(rows, key=lambda r: r['stem'])


def equivalent_number(a, b):
    aa, bb = Decimal(str(a)), Decimal(str(b))
    return aa.is_finite() and bb.is_finite() and aa == bb


def collect(box, tag, queue, out, base=BASE, published=True):
    out, base = Path(out), Path(base)
    rows = queue_rows(queue, box)
    frame_list = out / (f'box_dem_{tag}_leveled.frames.txt' if published else f'level_frames_{tag}/FRAMES.txt')
    records = {}
    for line in frame_list.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 4 or parts[0] in records:
            raise ValueError('invalid or duplicate recorded frame-list row')
        records[parts[0]] = parts
    if set(records) != {r['stem'] for r in rows}:
        raise ValueError('recorded frame set differs from current tile queue')
    current = out / f'level_frames_{tag}/FRAMES.txt'
    if current.read_bytes() != frame_list.read_bytes():
        raise ValueError('current warped frame list differs from the published solve')
    leveled = out / f'box_dem_{tag}_leveled.tif'
    solve_mtime = leveled.stat().st_mtime_ns if published else None
    inputs = []
    for row in rows:
        stem = row['stem']
        pred = out / f'preds_{tag}' / stem
        if (pred / '.claim').exists() or (pred / '.fail').exists():
            raise ValueError(f'active or failed prediction: {stem}')
        done_path, guard_path = pred / '.done', pred / 'PRED_REFINE.json'
        done, guard = json.loads(done_path.read_text()), json.loads(guard_path.read_text())
        if (os.path.normpath(done.get('ctx', '')) != os.path.normpath(row['ctx']) or
                os.path.normpath(guard.get('ctx', '')) != os.path.normpath(row['ctx']) or
                guard.get('guard') is not True or guard.get('tag') != tag or guard.get('stem') != stem):
            raise ValueError(f'prediction context/guard differs from current queue: {stem}')
        if not all(equivalent_number(done.get(k), row[k]) for k in ('inc', 'az')):
            raise ValueError(f'prediction incidence/azimuth differs from current queue: {stem}')
        native = pred / 'pred_shot_anchor.tif'
        target = Path(records[stem][1])
        if target not in (native, pred / 'pred_shot_anchor_eqc.tif'):
            raise ValueError(f'frame-list target is not this tagged prediction: {stem}')
        target_id, native_id = identity(target), identity(native)
        if (target_id['size'], target_id['mtime_ns']) != (int(records[stem][2]), int(records[stem][3])):
            raise ValueError(f'prediction changed since the recorded frame list: {stem}')
        link = out / f'level_frames_{tag}' / stem / 'pred_shot_anchor.tif'
        if link.resolve(strict=True) != target.resolve(strict=True):
            raise ValueError(f'warped-frame link differs from recorded target: {stem}')
        files = dict(native=native_id, target=target_id, done=identity(done_path),
                     guard=identity(guard_path))
        if 'context_signature' in done:
            # Only a genuine before/after record from the prediction can grant
            # equivalence across a later byte-identical VRT rewrite. Never
            # synthesize these records for legacy completed predictions.
            signature = validate_capture(done['context_signature'], row['ctx'], native)
            files['ctx'] = dict(path=row['ctx'], context_signature=signature)
        else:
            files['ctx'] = identity(row['ctx'])
            if files['ctx']['mtime_ns'] > native_id['mtime_ns']:
                raise ValueError(f'context changed after prediction: {stem}')
        if solve_mtime is not None and any(v.get('mtime_ns', 0) > solve_mtime for v in files.values()):
            raise ValueError(f'prediction inputs changed after level publication: {stem}')
        if target != native:
            sidecar = pred / 'pred_shot_anchor_eqc.json'
            sc = json.loads(sidecar.read_text())
            if (sc.get('src_size'), sc.get('src_mtime_ns')) != (native_id['size'], native_id['mtime_ns']):
                raise ValueError(f'warped copy is from an older native prediction: {stem}')
            files['warp_sidecar_sha256'] = digest(sidecar)
        inputs.append(dict(stem=stem, files=files, done_sha256=digest(done_path), guard_sha256=digest(guard_path)))
    weights = V3_WEIGHTS / f'{box}.csv'
    orbit = base / f'global_run/shape_error/{box}/orbit_screen_{box}.txt'
    solver = V3D / 'code/level_frames_box_v3.py'
    weight_hash = digest(weights, 'sha1') if weights.exists() and weights.stat().st_size else 'none'
    orbit_hash = digest(orbit, 'sha1') if orbit.exists() and orbit.stat().st_size else 'none'
    solver_hash = digest(solver, 'md5')
    taper = int(os.environ.get('TAPER_PX', '512'))
    reference = out / 'box_dem_ta_implicit_d1_implrel100s1_e088_leveled.tif'
    reference_identity = identity(reference) if reference.exists() and reference.stat().st_size else None
    desired = dict(box=box, tag=tag, rows=rows, frames_sha1=digest(frame_list, 'sha1'), inputs=inputs,
                   frame_weights_sha1=weight_hash, orbit_screen_sha1=orbit_hash,
                   level_script_md5=solver_hash, taper_px=taper, coverage_reference=reference_identity)
    if published:
        provenance = json.loads(out.joinpath(f'box_dem_{tag}_leveled.provenance.json').read_text())
        expected = dict(box=box, tag=tag, frames=len(rows), frame_list_sha1=desired['frames_sha1'],
                        frame_weights_sha1=weight_hash, orbit_screen_sha1=orbit_hash,
                        level_script_md5=solver_hash, taper_px=taper)
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f'level provenance differs from current inputs: {key}')
        if reference_identity is not None:
            if (provenance.get('coverage_reference') != str(reference) or
                    provenance.get('coverage_reference_identity') != reference_identity):
                raise ValueError('level solve did not preserve the current reference coverage grid')
        future_stamp = out / f'box_dem_{tag}_leveled.input_identity.json'
        if future_stamp.exists() and json.loads(future_stamp.read_text()) != desired:
            raise ValueError('published per-tile input identity differs from current inputs')
    return desired


def check(box, tag, queue, out, base=BASE):
    try:
        collect(box, tag, queue, out, base, published=True)
        return True, 'exact tile frame set, source identities/context, weights and solver provenance match'
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        return False, str(exc)


def write_atomic(path, data):
    p = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=p.name + '.', dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(data, fh, sort_keys=True, indent=1)
            fh.write('\n')
        os.replace(temporary, p)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('box')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--queue', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default=str(BASE))
    ap.add_argument('--snapshot', help='capture inputs before a new solve')
    ap.add_argument('--expect-snapshot', help='refuse publication when inputs changed during the solve')
    args = ap.parse_args()
    if args.snapshot or args.expect_snapshot:
        try:
            data = collect(args.box, args.tag, args.queue, args.out, args.base, published=False)
            if args.expect_snapshot and json.loads(Path(args.expect_snapshot).read_text()) != data:
                raise ValueError('inputs changed while the level solve ran; do not publish')
            if args.snapshot:
                write_atomic(args.snapshot, data)
            ok, reason = True, 'current source identities verified'
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            ok, reason = False, str(exc)
    else:
        ok, reason = check(args.box, args.tag, args.queue, args.out, args.base)
    print(json.dumps(dict(ok=ok, box=args.box, reason=reason)))
    raise SystemExit(0 if ok else 6)


if __name__ == '__main__':
    main()
