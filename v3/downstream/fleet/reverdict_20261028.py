#!/usr/bin/env python3
"""reverdict_20261028.py QA_DIR STATE_DIR [--apply] : recompute the v3 QA verdict of every finished box (STATE_DIR/<B>/.done) under
the 2026-10-28 rule from the numbers already in its record (QA_DIR/<B>.txt: the two eval_box 'cell-core px' lines R1 then v3 and the
'PEDR flips' line), exactly as stages/qa_box_tag_v3.sh [rule 2026-10-28] computes it:
  PASS = lines v3 <= R1, step px v3 <= R1 + max(20, 10 % of R1), NOT (sign test p < 0.01 AND v3-only - R1-only > max(5, 5 % of R1
  failing cells)), shot MAE v3 - R1 <= 0.25 m.
Without --apply: prints old -> new per box and the counts (read only). With --apply: replaces the VERDICT line (old one kept as
'# superseded: ...'), atomically. Records without the eval lines are listed (re-run the QA stage for them)."""
import glob, math, os, re, sys
Q, ST = sys.argv[1], sys.argv[2]; apply = "--apply" in sys.argv; out = []
for d in sorted(glob.glob(f"{ST}/*/.done")):
    b = os.path.basename(os.path.dirname(d)); p = f"{Q}/{b}.txt"
    if not os.path.exists(p): out.append((b, "-", "no record")); continue
    T = open(p).read()
    if "[rule 2026-10-28]" in T: out.append((b, "already", re.search(r"VERDICT (\w+)", T).group(1))); continue
    L = [l for l in T.splitlines() if "cell-core px" in l]; fl = re.search(r"PEDR flips last vs first: only-last (\d+), only-first (\d+)", T)
    ov = re.search(r"^VERDICT (\w+).*$", T, re.M)
    if len(L) < 2 or not fl or not ov: out.append((b, ov.group(1) if ov else "-", "eval lines missing: re-run QA")); continue
    def f(l):
        g = lambda pat: int(re.search(pat, l).group(1))
        return dict(ns=g(r"N-S\s+(\d+) E-W"), ew=g(r"E-W\s+(\d+) \|"), lns=g(r"line cells N-S\s+(\d+)"), lew=g(r"line cells N-S\s+\d+ E-W\s+(\d+)"),
                    fail=g(r"fail (\d+)"), mae=float(re.search(r"shot MAE ([0-9.]+)", l).group(1)))
    r1, v3 = f(L[0]), f(L[1]); vo, ro = int(fl.group(1)), int(fl.group(2)); n = vo + ro
    p_ = sum(math.comb(n, i) for i in range(vo, n + 1)) / (1 << n) if n else 1.0
    pedr_bad = p_ < 0.01 and (vo - ro) > max(5, 0.05 * r1["fail"]); mae_bad = v3["mae"] - r1["mae"] > 0.25
    s3, s1 = v3["ns"] + v3["ew"], r1["ns"] + r1["ew"]
    ok = (v3["lns"] + v3["lew"] <= r1["lns"] + r1["lew"]) and (s3 <= s1 + max(20, 0.1 * s1)) and not pedr_bad and not mae_bad
    line = (f"VERDICT {'PASS' if ok else 'REVIEW'} lines {v3['lns'] + v3['lew']} vs R1 {r1['lns'] + r1['lew']}; step px {s3} vs {s1}; PEDR failing "
            f"cells {v3['fail']} vs {r1['fail']} (v3-only {vo}, R1-only {ro}, sign test p {p_:.3g}{', REVIEW' if pedr_bad else ''}); shot MAE "
            f"{v3['mae']:.2f} vs {r1['mae']:.2f}{' REVIEW' if mae_bad else ''} [rule 2026-10-28]")
    out.append((b, ov.group(1), "PASS" if ok else "REVIEW", s3, s1))
    if apply:
        new = T.replace(ov.group(0), f"# superseded: {ov.group(0)}\n{line}", 1)
        with open(p + ".tmp", "w") as fh: fh.write(new)
        os.replace(p + ".tmp", p)
ch = [o for o in out if len(o) == 5 and o[1] != o[2]]; rv = [o for o in out if len(o) == 5 and o[2] == "REVIEW"]
print(f"{len(out)} finished boxes; recomputed {sum(len(o) == 5 for o in out)}; old REVIEW {sum(1 for o in out if len(o) == 5 and o[1] == 'REVIEW')}"
      f" -> new REVIEW {len(rv)}; changed: " + ", ".join(f"{o[0]} {o[1]}->{o[2]} ({o[3]} vs {o[4]} step px)" for o in ch))
print("still REVIEW: " + ", ".join(f"{o[0]} ({o[3]} vs {o[4]})" for o in rv))
for o in out:
    if len(o) == 3: print(f"  {o[0]}: {o[2]}")
print("APPLIED" if apply else "dry run: nothing written")
