#!/usr/bin/env python3
"""pcf80_state.py -- one line per polar v3 box (rows Np80 + N-90): BOX notdone | current | pending.
notdone: no downstream/state/<BOX>/.done or no _deseam.tif. current: box_dem_<TAG>_deseam_pcf80.tif + .json exist and the marker's source
matches the current _deseam.tif (fast path: same size and mtime; else md5) AND the marker's params / code md5s equal the current
code/params.json and code/*.py (so changing the method re-stages every box). pending: everything else (missing or stale pcf80)."""
import glob, hashlib, json, os, sys
PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"; ST = f"{G}/global_run/v3_2026-10/downstream/state"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


PARAMS = json.load(open(f"{PC}/code/params.json")); CODE = {n: md5(f"{PC}/code/{n}") for n in ("combfix.py", "box_correct.py")}
ONLY = set(sys.argv[1:])                      # optional: restrict to these boxes
for B in sorted(os.path.basename(p) for p in glob.glob(f"{G}/global_run/boxes/E*_Np80") + glob.glob(f"{G}/global_run/boxes/E*_N-90")):
    if ONLY and B not in ONLY: continue
    bd = f"{G}/global_run/boxes/{B}/box_run"; S = f"{bd}/box_dem_{V3}_deseam.tif"; MK = f"{bd}/box_dem_{V3}_deseam_pcf80.json"
    if not (os.path.exists(f"{ST}/{B}/.done") and os.path.exists(S)): print(B, "notdone"); continue
    cur = False
    if os.path.exists(MK) and os.path.exists(f"{bd}/box_dem_{V3}_deseam_pcf80.tif"):
        m = json.load(open(MK)); st = os.stat(S)
        same_method = m.get("params") == PARAMS and m.get("code_md5") == CODE
        cur = same_method and ((m["source_size"] == st.st_size and abs(m["source_mtime"] - st.st_mtime) < 1e-3) or md5(S) == m["source_md5"])
    print(B, "current" if cur else "pending", flush=True)
