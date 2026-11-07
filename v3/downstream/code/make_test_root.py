#!/usr/bin/env python3
"""make_test_root.py ROOT BOX TESTTAG QUEUE_SRC : scratch box root for testing the v3 downstream chain WITHOUT new predictions.
ROOT/boxes/<BOX>/box_run/preds_<TESTTAG>/<stem>/ = symlinks to the release-2 prediction rasters, the R2 .done (unchanged) and a
PRED_REFINE.json whose tag is TESTTAG; box_dem_<R1>_leveled.tif = symlink to the R1 levelled mosaic (coverage grid). Also writes
ROOT/queue_test.tsv (copy of QUEUE_SRC), ROOT/CAMPAIGN_test.json ({"tag": TESTTAG}) and ROOT/boxes.tsv (row of BOX from the R2
fleet list). Nothing outside ROOT is written."""
import csv, json, os, shutil, sys
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; R1, R2 = "ta_implicit_d1_implrel100s1_e088", "ta_implicit_d1_implrel100s1_e088r2"
root, box, tag, qsrc = sys.argv[1:5]
src = f"{G}/global_run/boxes/{box}/box_run"; dst = f"{root}/boxes/{box}/box_run"; os.makedirs(dst, exist_ok=True)
rows = [r for r in csv.DictReader(open(qsrc), delimiter="\t") if r["box"] == box]
for r in rows:
    s, d = f"{src}/preds_{R2}/{r['stem']}", f"{dst}/preds_{tag}/{r['stem']}"; os.makedirs(d, exist_ok=True)
    for f in ("pred_shot_anchor.tif", "pred_feather.tif", "ctx_on_grid.tif", "mola_on_grid.tif"):
        if os.path.exists(f"{s}/{f}") and not os.path.lexists(f"{d}/{f}"): os.symlink(os.path.realpath(f"{s}/{f}"), f"{d}/{f}")
    shutil.copy2(f"{s}/.done", f"{d}/.done")
    g = json.load(open(f"{s}/PRED_REFINE.json")); g["tag"] = tag; g["note"] = "TEST COPY of the R2 guard (v3 chain test)"
    json.dump(g, open(f"{d}/PRED_REFINE.json", "w"), indent=1)
ref = f"{src}/box_dem_{R1}_leveled.tif"
if not os.path.lexists(f"{dst}/box_dem_{R1}_leveled.tif"): os.symlink(ref, f"{dst}/box_dem_{R1}_leveled.tif")
shutil.copy(qsrc, f"{root}/queue_test.tsv")
json.dump({"tag": tag, "note": "v3 downstream chain test on release-2 predictions"}, open(f"{root}/CAMPAIGN_test.json", "w"))
with open(f"{G}/global_run/fix_2026-10/rebuild/downstream/boxes.tsv") as fh, open(f"{root}/boxes.tsv", "w") as o:
    o.write("box\tkind\tmurray\tframes\tfailed\n")
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["box"] == box: o.write(f"{box}\t{r['kind']}\t{r['murray']}\t{len(rows)}\t0\n")
print(box, len(rows), "frames ->", dst)
