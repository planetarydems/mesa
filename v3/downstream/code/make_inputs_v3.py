#!/usr/bin/env python3
"""make_inputs_v3.py V3D QUEUE_TSV : the two per-box inputs of the v3 downstream fleet (read only elsewhere).
  V3D/weights/<BOX>.csv  release-2 blend factors (fix_2026-10/blend/weights/<BOX>.csv) with every frame whose reason contains
                         "nontranslational" (unresolved misplacement, align_fix checked_frames: partners > 300 m apart) set to
                         0.0: level_frames_box_v3.py then gives it no weight wherever full-weight coverage reaches 512 px of
                         feather weight, and its full weight where it is the only coverage (hand-over over 512 px).
  V3D/boxes.tsv          box, kind, murray (from the release-2 fleet list), frames (rows of QUEUE_TSV), failed (0), in queue_v3.tsv
                         order (the order boxes are predicted); boxes without queue rows are left out (listed on stdout)."""
import csv, glob, os, sys, collections
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; F = f"{G}/global_run/fix_2026-10"
v3d, queue = sys.argv[1], sys.argv[2]
os.makedirs(f"{v3d}/weights", exist_ok=True); n_ex = 0; n_files = 0
for p in sorted(glob.glob(f"{F}/blend/weights/*.csv")):
    rows = list(csv.DictReader(open(p)))
    with open(f"{v3d}/weights/{os.path.basename(p)}", "w") as o:
        o.write("stem,factor,reason\n")
        for r in rows:
            f, why = float(r["factor"]), r["reason"]
            if "nontranslational" in why: f, why, n_ex = 0.0, why + "+v3_excluded_where_covered", n_ex + 1
            o.write(f"{r['stem']},{f:g},{why}\n")
    n_files += 1
cnt = collections.Counter(); order = []
with open(queue) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["box"] not in cnt: order.append(r["box"])
        cnt[r["box"]] += 1
info = {r["box"]: r for r in csv.DictReader(open(f"{F}/rebuild/downstream/boxes.tsv"), delimiter="\t")}
miss = [b for b in info if cnt[b] == 0]
with open(f"{v3d}/boxes.tsv", "w") as o:                  # queue order: boxes become ready in the order they are predicted
    o.write("box\tkind\tmurray\tframes\tfailed\n")
    for b in order:
        r = info[b]; o.write(f"{b}\t{r['kind']}\t{r['murray']}\t{cnt[b]}\t0\n")
print(f"weights: {n_files} files, {n_ex} frames excluded where covered; boxes.tsv: {sum(1 for _ in open(v3d + '/boxes.tsv')) - 1} boxes; "
      f"no queue rows: {len(miss)} {miss[:10]}")
