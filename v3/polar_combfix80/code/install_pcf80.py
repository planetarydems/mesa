#!/usr/bin/env python3
"""install_pcf80.py BOX PRE_MD5 -- write the verified polar comb filter output BESIDE the box's release surface (coordinator 10-29: never
swap _deseam.tif; the fleet owns it). Writes box_run/box_dem_<V3>_deseam_pcf80.tif (temp + rename, md5 checked) and the marker
box_run/box_dem_<V3>_deseam_pcf80.json (source md5 / mtime / size, output md5, code md5s, ramp, verify result). Consumers (render,
CTX/uncertainty, website) use the pcf80 file only while the marker's source_md5 equals the current _deseam.tif md5.
Refuses when verify.json is not ok, when _deseam.tif changed during staging (PRE_MD5 = md5 taken before the filter ran), or when the
staging output changed."""
import hashlib, json, os, shutil, sys, time
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"
PC = os.environ.get("PC", f"{G}/global_run/v3_2026-10/polar_combfix80"); B, pre = sys.argv[1], sys.argv[2]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


bd = f"{G}/global_run/boxes/{B}/box_run"; src = f"{bd}/box_dem_{V3}_deseam.tif"; dst = f"{bd}/box_dem_{V3}_deseam_pcf80.tif"
mark = f"{bd}/box_dem_{V3}_deseam_pcf80.json"; v = json.load(open(f"{PC}/boxes/{B}/verify.json"))
if not v["ok"]: sys.exit(f"{B} REFUSED: verify not ok")
if not (pre == v["source_md5"] == md5(src)): sys.exit(f"{B} REFUSED: _deseam.tif changed during staging (re-stage)")
if md5(v["output"]) != v["output_md5"]: sys.exit(f"{B} REFUSED: staging output changed")
tmp = f"{bd}/.tmp_pcf80_{os.getpid()}.tif"; shutil.copyfile(v["output"], tmp)
if md5(tmp) != v["output_md5"]: os.remove(tmp); sys.exit(f"{B} FAILED copy check")
os.replace(tmp, dst)
code = {os.path.basename(p): md5(p) for p in (f"{PC}/code/combfix.py", f"{PC}/code/box_correct.py")}
st = os.stat(src)
rec = dict(box=B, written=time.strftime("%Y-%m-%d %H:%M:%S"), source=src, source_md5=v["source_md5"], source_mtime=st.st_mtime, source_size=st.st_size,
           output=dst, output_md5=v["output_md5"], method="combfix v2 + infill notch + k gate + size limit (rule v3, 2026-10-30), frozen copy in polar_combfix80/code",
           code_md5=code, params=json.load(open(f"{PC}/code/params.json")), ramp="0 at |lat| <= 80 deg, 1 at >= 81 deg; output identical to source below 80 deg",
           verify={k: v[k] for k in ("ok", "same_grid", "same_valid_mask", "nonvalid_unchanged", "max_change_below_80", "correction_rms_m", "correction_p99_m",
                                     "correction_max_m", "frac_gt_20m")})
tmpj = mark + ".tmp"; json.dump(rec, open(tmpj, "w"), indent=1); os.replace(tmpj, mark)
print(B, "installed", v["correction_rms_m"], v["correction_max_m"])
