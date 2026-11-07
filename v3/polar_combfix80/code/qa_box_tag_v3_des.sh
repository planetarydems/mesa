#!/usr/bin/env bash
# qa_box_tag_v3.sh <BOX> : per-box acceptance record of the v3 chain (2026-10-24), read only on products.
#   1. code/eval_box.py: R1 de-seamed tile vs the v3 surface (box_dem_<TAG>_deseam.tif) in the box's own cell: one-pixel steps
#      > 20 m (N-S and E-W), straight-line cells, PEDR census cells failing the census rule, shot MAE.
#   2. code/lattice_box.py: lattice metric of the v3 surface vs R1 before (deseam) and after (deseam_dlc2) the lattice filter.
#   3. (2026-10-27) code/frame_outlier_v3.py --parity all --no-score: REPORT of PEDR tracks along which a CTX-derived stretch misses
#      the shots by > max(50 m, 250 tan slope) for >= 10 shots, and (polar boxes) frame edges > 150 m off MOLA; regions sized, written
#      to $QA_DIR/outliers/<BOX>_all_segments.tsv; never part of the verdict, never a product change.
# Writes $QA_DIR/<BOX>.txt (QA_DIR default <V3D>/qa). The verdict is recorded, never enforced here (exit 0 unless a tool fails).
#   PASS when straight-line cells (N-S + E-W) <= R1's, step pixels <= R1's + max(20, 10 % of R1's) (2026-10-28 floor; before:
#   <= R1's), the PEDR cells failing in v3 only do NOT significantly
#   outnumber those failing in R1 only (REVIEW only if one-sided sign test p < 0.01 AND v3-only - R1-only > max(5, 5 % of R1's
#   failing cells); cells scored by both), and the shot MAE is not worse than R1's by > 0.25 m. (2026-10-27; before: failing PEDR
#   cells <= R1's + max(2, 2 %).)
# Env: TAG, V3D, BOXES_ROOT, QA_DIR, NP (eval workers, default 16).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
NAME=${1:?box}; TAG=${TAG:?TAG}; V3D=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}
BASE=/home/gridsan/groups/planetary_dtm/mars_dem
OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$NAME/box_run
QA=${QA_DIR:-$V3D/qa}; mkdir -p "$QA"; REC=$QA/$NAME.txt
DES=${DES:-$OUT/box_dem_${TAG}_deseam.tif}
[ -s "$DES" ] || { echo "[qa] no $DES"; exit 2; }
unset PROJ_DATA PROJ_LIB
{ echo "# qa $NAME tag=$TAG $(date '+%F %T')"
  NP=${NP:-16} ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} "$V3D/code/eval_box.py" "$NAME" r1 "$DES" || echo "EVAL_FAILED"
  N=${LATTICE_N:-300} ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} "$V3D/code/lattice_box.py" "$NAME" "v3=$DES" r1 r1dlc2 || echo "LATTICE_FAILED"
  NP=${NP:-16} ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} "$V3D/code/frame_outlier_v3.py" "$NAME" "$QA/outliers" --parity all \
      --tag "$TAG" --no-score 2>&1 | grep -v "Warning\|warn(" | grep "^OUTLIERS" || echo "OUTLIERS $NAME: check failed (report only)"
} > "$REC.partial" 2>&1
python3 - "$REC.partial" <<'PY' >> "$REC.partial"
import math, re, sys
T = open(sys.argv[1]).read(); L = [l for l in T.splitlines() if "cell-core px" in l]
def f(l):
    g = lambda p: int(re.search(p, l).group(1))
    return dict(ns=g(r"N-S\s+(\d+) E-W"), ew=g(r"E-W\s+(\d+) \|"), lns=g(r"line cells N-S\s+(\d+)"), lew=g(r"line cells N-S\s+\d+ E-W\s+(\d+)"),
                fail=g(r"fail (\d+)"), mae=float(re.search(r"shot MAE ([0-9.]+)", l).group(1)))
fl = re.search(r"PEDR flips last vs first: only-last (\d+), only-first (\d+), both (\d+)", T)
if len(L) < 2 or not fl: print("VERDICT UNKNOWN (eval incomplete)"); sys.exit(0)
r1, v3 = f(L[0]), f(L[1]); vo, ro = int(fl.group(1)), int(fl.group(2)); n = vo + ro
p = sum(math.comb(n, i) for i in range(vo, n + 1)) / (1 << n) if n else 1.0       # one-sided sign test: P(X >= vo | n, 1/2)
pedr_bad = p < 0.01 and (vo - ro) > max(5, 0.05 * r1["fail"])
mae_bad = v3["mae"] - r1["mae"] > 0.25
s3, s1 = v3["ns"] + v3["ew"], r1["ns"] + r1["ew"]
ok = (v3["lns"] + v3["lew"] <= r1["lns"] + r1["lew"]) and (s3 <= s1 + max(20, 0.1 * s1)) and not pedr_bad and not mae_bad   # step floor 2026-10-28
print(f"VERDICT {'PASS' if ok else 'REVIEW'} lines {v3['lns'] + v3['lew']} vs R1 {r1['lns'] + r1['lew']}; step px {v3['ns'] + v3['ew']} vs "
      f"{r1['ns'] + r1['ew']}; PEDR failing cells {v3['fail']} vs {r1['fail']} (v3-only {vo}, R1-only {ro}, sign test p {p:.3g}"
      f"{', REVIEW' if pedr_bad else ''}); shot MAE {v3['mae']:.2f} vs {r1['mae']:.2f}{' REVIEW' if mae_bad else ''} [rule 2026-10-28]")
PY
mv -f "$REC.partial" "$REC"
grep -h "VERDICT\|^OUTLIERS" "$REC"
grep -q "EVAL_FAILED\|LATTICE_FAILED" "$REC" && exit 3
exit 0
