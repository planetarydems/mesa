#!/usr/bin/env python3
"""Release-2 VRT XML closure plus local raster hashes; never decodes pixels.

VRT mtimes/inodes are deliberately excluded from the signature, allowing an
identical XML rewrite. Leaf rasters and GDAL companions are hashed in full;
their identities also remain strict. Timestamp granularity is not equivalence.
Legacy predictions without a before/after capture gain no equivalence proof.

v3 COPY (2026-10-29) for the v3 input guard; the release-2 original stays untouched. Only change: _file_sha256 reuses a
per-file digest cache keyed by the file's resolved path and stat identity (device, inode, size, mtime_ns, ctime_ns), so
unchanged bytes are not re-read on every guard run. Signatures, digests and recorded captures are byte-identical.
- hit: the stored digest, only when every identity field matches the file's current stat;
- miss: hash as before; store only if the identity was unchanged during hashing AND the file's mtime and ctime are
  older than V3_CTXSIG_MIN_AGE_S (default 300 s; coarse file-system timestamps, clock skew);
- cache = one JSON per file under V3_CTXSIG_CACHE (default <V3D>/ctxsig_cache; empty string disables), written
  atomically (temp + rename); unreadable/malformed entries are ignored and rewritten; write failures are ignored.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

SCHEMA = 'release2_context_closure_v1'
CAPTURE_SCHEMA = 'release2_context_capture_v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def _identity(path):
    p = Path(path)
    s = p.stat()
    if not p.is_file() or s.st_size <= 0:
        raise ValueError(f'context dependency is not a nonempty file: {p}')
    return dict(path=str(p), resolved_path=str(p.resolve(strict=True)), device=s.st_dev,
                inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


CACHE_SCHEMA = 'ctxsig_file_sha256_cache_v1'
CACHE_DIR = os.environ.get('V3_CTXSIG_CACHE', os.path.join(
    os.environ.get('V3D', '/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream'), 'ctxsig_cache'))
CACHE_MIN_AGE_NS = int(float(os.environ.get('V3_CTXSIG_MIN_AGE_S', '300')) * 1e9)
CACHE_STATS = dict(hit=0, miss=0, stored=0, too_recent=0, changed_while_hashing=0, bad_entry=0, store_failed=0)
_HEX = set('0123456789abcdef')


def _stat_key(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _entry_path(real):
    h = hashlib.sha256(real.encode('utf-8', 'surrogateescape')).hexdigest()
    return os.path.join(CACHE_DIR, h[:2], h + '.json')


def _file_sha256(path):
    if not CACHE_DIR:
        return _hash_file(path)
    real = os.path.realpath(path)
    key = _stat_key(os.stat(path))
    entry = _entry_path(real)
    try:
        with open(entry) as fh:
            e = json.load(fh)
        d = e.get('sha256') if isinstance(e, dict) else None
        if (e.get('schema') == CACHE_SCHEMA and e.get('path') == real and e.get('stat') == key and
                isinstance(d, str) and len(d) == 64 and set(d) <= _HEX):
            CACHE_STATS['hit'] += 1
            return d
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError, TypeError):
        CACHE_STATS['bad_entry'] += 1
    CACHE_STATS['miss'] += 1
    digest = _hash_file(path)
    try:
        st = os.stat(path)
        if _stat_key(st) != key:
            CACHE_STATS['changed_while_hashing'] += 1
        elif time.time_ns() - max(st.st_mtime_ns, st.st_ctime_ns) < CACHE_MIN_AGE_NS:
            CACHE_STATS['too_recent'] += 1
        else:
            os.makedirs(os.path.dirname(entry), exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.tmp.', suffix='.json', dir=os.path.dirname(entry))
            try:
                with os.fdopen(fd, 'w') as fh:
                    json.dump(dict(schema=CACHE_SCHEMA, path=real, stat=key, sha256=digest), fh)
                os.replace(tmp, entry)
                CACHE_STATS['stored'] += 1
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    except OSError:
        CACHE_STATS['store_failed'] += 1
    return digest


def _companions(path):
    p = Path(path)
    names = [str(p) + suffix for suffix in ('.aux.xml', '.ovr', '.msk')]
    if p.suffix.lower() in ('.tif', '.tiff'):
        names += [str(p.with_suffix(s)) for s in ('.tfw', '.tifw', '.wld')]
    # Record absence too: adding a previously absent companion changes the digest.
    return {name: dict(identity=_identity(name), sha256=_file_sha256(name))
            if Path(name).exists() else None for name in names}


def capture_context(path):
    root = Path(os.path.abspath(os.fspath(path)))
    nodes, observed, active = {}, {}, set()

    def visit(p, depth=0):
        p = Path(os.path.abspath(p))
        key = str(p)
        if depth > 32 or len(nodes) > 4096:
            raise ValueError('context closure exceeds conservative size/depth limits')
        ident = _identity(p)
        if ident['resolved_path'] in active:
            raise ValueError(f'cyclic context VRT: {p}')
        if key in nodes:
            return
        observed[key] = ident
        with p.open('rb') as fh:
            prefix = fh.read(4096)
            is_vrt = p.suffix.lower() == '.vrt' or b'<VRTDataset' in prefix
            if is_vrt:
                if ident['size'] > 16 * 1024 * 1024:
                    raise ValueError(f'oversized VRT: {p}')
                fh.seek(0)
                data = fh.read()
        companions = _companions(p)
        if not is_vrt:
            if p.suffix.lower() not in ('.tif', '.tiff'):
                raise ValueError(f'only local GeoTIFF raster leaves are supported: {p}')
            nodes[key] = dict(kind='raster', identity=ident, sha256=_file_sha256(p), companions=companions)
            return
        if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
            raise ValueError(f'VRT entities/DTD are unsupported: {p}')
        try:
            tree = ET.fromstring(data)
        except ET.ParseError as exc:
            raise ValueError(f'malformed VRT XML: {p}') from exc
        if tree.tag != 'VRTDataset':
            raise ValueError(f'not a standard VRTDataset: {p}')
        # Warped/geolocation datasets can have dependencies outside these source
        # elements. Fail closed until those schemas are deliberately supported.
        if tree.attrib.get('subClass') or any(e.tag in ('GeoLocation', 'GDALWarpOptions') for e in tree.iter()):
            raise ValueError(f'unsupported VRT dependency schema: {p}')
        sources = [e for e in tree.iter() if e.tag in ('SourceFilename', 'SourceDataset')]
        if not sources:
            raise ValueError(f'VRT has no explicit local source datasets: {p}')
        paths = []
        for element in sources:
            raw = element.text or ''
            if not raw or raw != raw.strip() or raw.startswith('/vsi') or '://' in raw or ':' in raw:
                raise ValueError(f'unsupported VRT source reference in {p}')
            source = Path(raw)
            if not source.is_absolute():
                if element.get('relativeToVRT') != '1':
                    raise ValueError(f'VRT relative source lacks relativeToVRT=1: {p}')
                source = p.parent / source
            paths.append(Path(os.path.abspath(source)))
        nodes[key] = dict(kind='vrt', path=key, resolved_path=ident['resolved_path'],
                          xml_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                          sources=sorted(set(map(str, paths))), companions=companions)
        active.add(ident['resolved_path'])
        for source in paths:
            visit(source, depth + 1)
        active.remove(ident['resolved_path'])

    visit(root)
    # Check every dependency again after the entire traversal, including exact
    # VRT mtimes while reading. Only stable completed captures ignore VRT mtime.
    for name, ident in observed.items():
        if _identity(name) != ident or _companions(name) != nodes[name]['companions']:
            raise ValueError(f'context dependency changed during capture: {name}')
    payload = dict(root=str(root), nodes=[nodes[k] for k in sorted(nodes)])
    return dict(schema=SCHEMA, digest=hashlib.sha256(_canonical(payload)).hexdigest(), payload=payload)


def _validate_signature(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'digest', 'payload'}:
        raise ValueError('malformed context closure signature')
    if value['schema'] != SCHEMA or not isinstance(value['payload'], dict):
        raise ValueError('unsupported context closure signature')
    if hashlib.sha256(_canonical(value['payload'])).hexdigest() != value['digest']:
        raise ValueError('context closure signature digest mismatch')


def validate_capture(record, path, native_path):
    if not isinstance(record, dict) or record.get('schema') != CAPTURE_SCHEMA:
        raise ValueError('missing or unsupported before/after context capture')
    before, after = record.get('before'), record.get('after')
    _validate_signature(before)
    _validate_signature(after)
    if before != after:
        raise ValueError('context dependencies changed during prediction')
    start, end = record.get('before_captured_ns'), record.get('after_captured_ns')
    if (type(start) is not int or type(end) is not int or start <= 0 or end < start or
            end > time.time_ns() + 1_000_000_000):
        raise ValueError('invalid context capture time interval')
    native = _identity(native_path)
    if not start // 1_000_000_000 * 1_000_000_000 <= native['mtime_ns'] <= end:
        raise ValueError('native prediction is outside its context capture interval')
    current = capture_context(path)
    if before != current:
        raise ValueError('current context dependencies differ from recorded prediction capture')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('context')
    args = parser.parse_args()
    print(json.dumps(capture_context(args.context), sort_keys=True))


if __name__ == '__main__':
    main()
