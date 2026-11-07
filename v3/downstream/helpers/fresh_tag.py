#!/usr/bin/env python3
"""fresh_tag.py -- provenance stamps and the freshness gate of the tag-aware downstream chain (review finding D1, 2026-09-17).

Problem it fixes: after a box was re-levelled, ctx_fill_box_tag.sh skipped because a FINAL existed, and deseam_box_tag.sh
skipped mask/census/gate/splice because their files existed. The exports then shipped products built from the PREVIOUS
level solve. Now every derived product records what it was built from, and every skip, de-seam stage and export checks it.

Each derived product P of TAG has a stamp P.inputs.json with the identity (inode, size, mtime_ns) of each DIRECT input at
build time. P is FRESH iff
  * P exists (size > 0), and its stamp exists and names this node and tag;
  * every input exists, still has exactly the identity in the stamp (a re-solve, a re-fill, a re-splice or a restore of an
    older file with mv all change it), and is not newer than P;
  * every input that is itself derived is FRESH (recursive, down to the roots).
Roots (no stamp): box_dem_<TAG>_leveled.tif (guarded by the FRAMES.txt check, exit 6) and ctx_robbins_18m_<TAG>.tif
(guarded by the grid check and the coverage gate in ctx_fill_box_tag.sh). Pseudo-product leveled_<TAG>/SOLVE.json is
written by the level step. It ties the per-frame leveled/weight tifs that the de-seam reads to the mosaic they came from.

CLI (OUT = <boxes>/<BOX>/box_run):
  fresh_tag.py check OUT TAG NODE...                rc 0 iff every NODE is fresh; otherwise prints why (rc 1)
  fresh_tag.py ids OUT TAG NODE...                  current identities of the inputs of NODE... (one line, for --expect)
  fresh_tag.py stamp [--expect IDS] OUT TAG NODE... write the stamps. Refuses (rc 1, nothing written) if an input is
                                                    missing, is newer than the product, or changed since IDS (i.e. while
                                                    the stage was running)
  fresh_tag.py list [--stale-only] OUT TAG          every tagged product derived from the leveled mosaic, with its state
  fresh_tag.py quarantine-stale OUT TAG             move every existing derived product that is NOT fresh (+ companions +
                                                    stamp) to <file>.stale.<epoch>; fresh products stay
  fresh_tag.py quarantine OUT TAG NODE...           move NODE... and everything derived from them aside, fresh or not
  fresh_tag.py export-check OUT TAG deseam|FINAL    rc 0 iff the exported product and its fill mask (prov) are fresh
Stdlib only (anaconda python 3.9 and isis python 3.11).
"""
import collections, json, os, socket, sys, time

PROTECTED_TAGS = ("ta_implicit_d1", "ta_implicit_d1_ftpolar_lr1e-4")   # == tag_guard.sh PROTECTED_TAGS (static test)
STAMP_SFX = ".inputs.json"
# The coarse inpainter weights are a ROOT of the fill products (2026-09-21): a fill made with other weights is stale.
# ONE definition: build_box_final_tag.sh takes its COARSE from `fresh_tag.py ckpt` and exports COARSE_CKPT so every
# check/stamp in that stage sees the same file. Stamps written before this root existed lack it -> stale -> rebuilt.
DEFAULT_COARSE_CKPT = "/home/gridsan/groups/planetary_dtm/mars_dem/src/inpainting/runs/coarse_global_v2/best.pt"
COARSE_CKPT = os.environ.get("COARSE_CKPT", DEFAULT_COARSE_CKPT)
# The 18 m FINE inpainter (2026-09-24, user decision "Deploy as-is") is a second root, of the MODEL surface only (the
# fine pass rewrites box_dem_<TAG>_filled.tif.model.tif; filled/prov stay the coarse composite + hole mask). Exported by
# build_box_final_tag.sh as FINE_CKPT. FINE_CKPT=none switches the fine pass off (rollback) and drops the root.
DEFAULT_FINE_CKPT = "/home/gridsan/groups/planetary_dtm/mars_dem/src/inpainting/runs/fine_global_v1/best.pt"
FINE_CKPT = os.environ.get("FINE_CKPT", DEFAULT_FINE_CKPT)
_MODEL_INPUTS = ("leveled", "ctx", "coarse_ckpt") + (() if FINE_CKPT == "none" else ("fine_ckpt",))

# name: (product, direct inputs, companions moved with the product). Order is topological.
NODES = collections.OrderedDict([
    ("leveled", ("box_dem_{t}_leveled.tif",          (),                                     ())),
    ("ctx",     ("ctx_robbins_18m_{t}.tif",          (),                                     ())),
    ("coarse_ckpt", (COARSE_CKPT,                    (),                                     ())),   # absolute path
    ("fine_ckpt",   (FINE_CKPT,                      (),                                     ())),   # absolute path or "none"
    ("frames",  ("leveled_{t}/SOLVE.json",           ("leveled",),                           ())),
    ("model",   ("box_dem_{t}_filled.tif.model.tif", _MODEL_INPUTS,                          ())),
    ("filled",  ("box_dem_{t}_filled.tif",           ("leveled", "ctx", "coarse_ckpt"),      ())),
    ("prov",    ("box_dem_{t}_filled.tif.prov.tif",  ("leveled", "ctx", "coarse_ckpt"),      ())),
    ("FINAL",   ("box_dem_{t}_FINAL.tif",            ("leveled", "model"),                   ())),
    ("mask",    ("box_seam_{t}_mask.tif",            ("prov", "frames"),                     ("box_seam_{t}_domid.tif",))),
    ("sites",   ("box_seam_{t}_sites.csv",           ("FINAL", "frames"),                    ("box_seam_{t}_strength.tif", "box_seam_{t}_cov.tif"))),
    ("gate",    ("box_deseam_base_{t}.csv",          ("FINAL", "frames", "mask", "sites"),   ())),
    ("deseam",  ("box_dem_{t}_deseam.tif",           ("FINAL", "frames", "mask", "gate"),    ())),
    ("report",  ("box_deseam_report_{t}.csv",        ("FINAL", "deseam", "frames", "mask", "sites"), ())),
])
ROOTS = ("leveled", "ctx", "coarse_ckpt", "fine_ckpt")
NO_QUARANTINE = ROOTS + ("frames",)     # roots have their own guards; SOLVE.json is rewritten only by a re-level
EXPORT_NODES = {"FINAL": ("FINAL", "prov"), "deseam": ("deseam", "prov")}


def path(out, tag, node):
    return os.path.join(out, NODES[node][0].format(t=tag))


def files_of(out, tag, node):
    """product, companions, stamp"""
    p = path(out, tag, node)
    return [p] + [os.path.join(out, c.format(t=tag)) for c in NODES[node][2]] + ([p + STAMP_SFX] if node not in ROOTS else [])


def ident(p):
    s = os.stat(p)
    return dict(ino=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def _idstr(d):
    return f"{d['ino']}:{d['size']}:{d['mtime_ns']}"


def descendants(nodes):
    """nodes plus every node derived from them, in topological order"""
    want = set(nodes)
    for n, (_, ins, _) in NODES.items():
        if any(i in want for i in ins):
            want.add(n)
    return [n for n in NODES if n in want]


def fresh(out, tag, node, _memo=None):
    """-> (True, "fresh") or (False, reason)"""
    memo = {} if _memo is None else _memo
    if node in memo:
        return memo[node]
    p = path(out, tag, node)
    rel = os.path.relpath(p, out)
    if not (os.path.isfile(p) and os.path.getsize(p) > 0):
        r = (False, f"{rel} missing or empty")
    elif node in ROOTS:
        r = (True, "fresh")
    else:
        r = None
        try:
            st = json.load(open(p + STAMP_SFX))
        except (OSError, ValueError):
            r = (False, f"{rel} has no readable provenance stamp {os.path.basename(p)}{STAMP_SFX} (not built by the stamped chain)")
        if r is None and (st.get("node") != node or st.get("tag") != tag):
            r = (False, f"{rel} stamp names node={st.get('node')} tag={st.get('tag')}")
        if r is None:
            pm = os.stat(p).st_mtime_ns
            for i in NODES[node][1]:
                ip = path(out, tag, i)
                irel = os.path.relpath(ip, out)
                rec = st.get("inputs", {}).get(i)
                if not os.path.isfile(ip):
                    r = (False, f"its input {irel} is missing"); break
                cur = ident(ip)
                if rec is None or _idstr(rec) != _idstr(cur):
                    r = (False, f"built from a different {irel} (stamp {_idstr(rec) if rec else 'none'}, now {_idstr(cur)})"); break
                if cur["mtime_ns"] > pm:
                    r = (False, f"{irel} is newer than {rel}"); break
                ok, why = fresh(out, tag, i, memo)
                if not ok:
                    r = (False, f"its input {i} is stale: {why}"); break
            if r is None:
                r = (True, "fresh")
    memo[node] = r
    return r


def input_ids(out, tag, nodes):
    ids = {}
    for n in nodes:
        for i in NODES[n][1]:
            ip = path(out, tag, i)
            ids[i] = _idstr(ident(ip)) if os.path.isfile(ip) else "missing"
    return ",".join(f"{k}={v}" for k, v in sorted(ids.items()))


def stamp(out, tag, nodes, expect=None):
    """-> list of problems (empty = every stamp written)"""
    probs = []
    exp = dict(kv.split("=", 1) for kv in expect.split(",") if kv) if expect else None
    todo = []
    for n in nodes:
        if n in ROOTS:
            probs.append(f"{n} is a root and is never stamped"); continue
        p = path(out, tag, n)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            probs.append(f"{n}: product {p} missing"); continue
        pm = os.stat(p).st_mtime_ns
        rec = {}
        for i in NODES[n][1]:
            ip = path(out, tag, i)
            if not os.path.isfile(ip):
                probs.append(f"{n}: input {ip} missing"); continue
            cur = ident(ip)
            if exp is not None and exp.get(i) != _idstr(cur):
                probs.append(f"{n}: input {os.path.relpath(ip, out)} changed while the stage ran ({exp.get(i)} -> {_idstr(cur)})")
            if cur["mtime_ns"] > pm:
                probs.append(f"{n}: input {os.path.relpath(ip, out)} is newer than the product")
            rec[i] = dict(cur, file=os.path.relpath(ip, out))
        todo.append((n, p, rec))
    if probs:
        return probs
    for n, p, rec in todo:
        doc = dict(node=n, tag=tag, product=os.path.basename(p), inputs=rec, date=time.strftime("%F %T"),
                   host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID", "local"))
        tmp = f"{p}{STAMP_SFX}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, p + STAMP_SFX)
    return []


def states(out, tag):
    """-> [(node, files that exist, 'fresh'|'stale'|'absent', reason)] for every derived node"""
    memo, res = {}, []
    for n in NODES:
        if n in ROOTS:
            continue
        have = [f for f in files_of(out, tag, n) if os.path.lexists(f)]
        ok, why = fresh(out, tag, n, memo)
        res.append((n, have, "fresh" if ok else ("stale" if have else "absent"), why))
    return res


def _move_aside(files, epoch, log):
    for f in files:
        if not os.path.lexists(f):
            continue
        dst = f"{f}.stale.{epoch}"
        k = 1
        while os.path.lexists(dst):
            dst = f"{f}.stale.{epoch}.{k}"; k += 1
        os.rename(f, dst)
        log(f"[stale] moved {f} -> {os.path.basename(dst)}")


def quarantine_stale(out, tag, log=print):
    """move every existing derived product that is not fresh aside; returns the moved node names"""
    epoch, moved = int(time.time()), []
    for n in NODES:
        if n in NO_QUARANTINE:
            continue
        have = [f for f in files_of(out, tag, n) if os.path.lexists(f)]
        if not have:
            continue
        ok, why = fresh(out, tag, n)          # no memo: earlier moves change the answer for descendants
        if not ok:
            log(f"[stale] {n}: {why}")
            _move_aside(have, epoch, log); moved.append(n)
    return moved


def quarantine(out, tag, nodes, log=print):
    epoch, moved = int(time.time()), []
    for n in descendants(nodes):
        if n in NO_QUARANTINE:
            continue
        have = [f for f in files_of(out, tag, n) if os.path.lexists(f)]
        if have:
            _move_aside(have, epoch, log); moved.append(n)
    return moved


def export_problems(out, tag, product):
    memo = {}
    return [f"{n}: {why}" for n in EXPORT_NODES[product] for ok, why in [fresh(out, tag, n, memo)] if not ok]


def main(argv):
    if argv and argv[0] == "ckpt":
        print(COARSE_CKPT); return 0
    if argv and argv[0] == "fine-ckpt":
        print(FINE_CKPT); return 0
    if len(argv) < 3:
        sys.exit(__doc__)
    cmd, args = argv[0], argv[1:]
    expect, stale_only = None, False
    if "--expect" in args:
        k = args.index("--expect"); expect = args[k + 1]; del args[k:k + 2]
    if "--stale-only" in args:
        args.remove("--stale-only"); stale_only = True
    out, tag, rest = os.path.abspath(args[0]), args[1], args[2:]
    bad = [n for n in rest if n not in NODES] if cmd in ("check", "ids", "stamp", "quarantine") else []
    if bad:
        sys.exit(f"unknown node(s) {bad}; nodes: {list(NODES)}")
    if cmd == "check":
        memo, rc = {}, 0
        for n in rest:
            ok, why = fresh(out, tag, n, memo)
            if not ok:
                print(f"[fresh] {n} is STALE: {why}"); rc = 1
        return rc
    if cmd == "ids":
        print(input_ids(out, tag, rest)); return 0
    if cmd == "stamp":
        probs = stamp(out, tag, rest, expect)
        for p in probs:
            print(f"[stamp] REFUSED {p}")
        if not probs:
            print(f"[stamp] {' '.join(rest)} stamped")
        return 1 if probs else 0
    if cmd == "list":
        for n, have, state, why in states(out, tag):
            if stale_only and state != "stale":
                continue
            names = [NODES[n][0].format(t=tag)] + [c.format(t=tag) for c in NODES[n][2]]
            print(f"{state.upper():6s} {n:7s} {' + '.join(names)}" + (f"   ({why})" if state == "stale" else ""))
        if not stale_only:
            print(f"ROOT   ctx     ctx_robbins_18m_{tag}.tif (+ .boxcrs_{tag}.wkt .murray_{tag}.vrt .murray_list_{tag}.txt): "
                  "kept only if it sits on the new leveled grid (ctx_fill_box_tag.sh grid check + coverage gate)")
            print("EXPORT any AOI mosaic exported from these products: re-export into a new --date directory")
        return 0
    if cmd == "quarantine-stale":
        moved = quarantine_stale(out, tag)
        print(f"[stale] quarantine-stale {out} tag={tag}: moved {len(moved)} product(s) {moved}")
        return 0
    if cmd == "quarantine":
        moved = quarantine(out, tag, rest)
        print(f"[stale] quarantine {out} tag={tag} {rest}: moved {len(moved)} product(s) {moved}")
        return 0
    if cmd == "export-check":
        if not rest or rest[0] not in EXPORT_NODES:
            sys.exit("export-check needs PRODUCT deseam|FINAL")
        probs = export_problems(out, tag, rest[0])
        for p in probs:
            print(f"[fresh] STALE {p}")
        return 1 if probs else 0
    sys.exit(f"unknown command {cmd}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
