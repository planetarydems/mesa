#!/usr/bin/env python3
"""summ_notch.py DIR -- staged combfix v2 (pcf80) vs v2 + 50 % infill notch (cfn80), both against v3, windows / shots >= 80 deg,
each box's own 10 x 10 deg. DIR/<BOX>/{pcf80,cfn80}_eval.json and _windows.tsv (copied from the sandbox)."""
import csv, glob, json, os, sys
D = sys.argv[1]; boxes = sorted(os.path.basename(p) for p in glob.glob(f"{D}/*") if os.path.isdir(p))
def ld(b, t):
    try: return json.load(open(f"{D}/{b}/{t}_eval.json"))
    except Exception: return None
def win(b, t):
    try: return {(int(r["r0"]), int(r["c0"])): r for r in csv.DictReader(open(f"{D}/{b}/{t}_windows.tsv"), delimiter="\t")}
    except Exception: return {}
print("| box | split | windows | comb amp p95: v3 / v2 / v2+notch | comb amp max | band RMS p95 | windows band > 2 m |")
print("|---|---|---|---|---|---|---|")
for b in boxes:
    P, N = ld(b, "pcf80"), ld(b, "cfn80")
    if not (P and N): print(f"| {b} | missing |"); continue
    for lab in ("frames", "infill"):
        p, n = P.get(f"windows_{lab}"), N.get(f"windows_{lab}")
        if not (p and n): continue
        print(f"| {b} | {lab} | {n['n']} | {p['comb_p95'][0]:.2f} / {p['comb_p95'][1]:.2f} / {n['comb_p95'][1]:.2f} | {p['comb_max'][0]:.2f} / {p['comb_max'][1]:.2f} / {n['comb_max'][1]:.2f} | "
              f"{p['band_rms_p95'][0]:.2f} / {p['band_rms_p95'][1]:.2f} / {n['band_rms_p95'][1]:.2f} | {p['n_band_gt2m'][0]} / {p['n_band_gt2m'][1]} / {n['n_band_gt2m'][1]} |")
print()
print("Windows where v2+notch is worse than v2 (comb amp or band RMS higher by > 0.05 m), and better:")
for b in boxes:
    wp, wn = win(b, "pcf80"), win(b, "cfn80"); k = sorted(set(wp) & set(wn))
    if not k: continue
    worse = [x for x in k if float(wn[x]["comb_fix"]) > float(wp[x]["comb_fix"]) + 0.05 or float(wn[x]["band_fix"]) > float(wp[x]["band_fix"]) + 0.05]
    better = [x for x in k if float(wn[x]["comb_fix"]) < float(wp[x]["comb_fix"]) - 0.05 or float(wn[x]["band_fix"]) < float(wp[x]["band_fix"]) - 0.05]
    print(f"  {b}: {len(k)} windows, worse {len(worse)}, better {len(better)}" + (f"; worse at " + ", ".join(f"({wn[x]['lat']},{wn[x]['lon']}, infill {wn[x]['infill']}: comb {wp[x]['comb_fix']}->{wn[x]['comb_fix']}, band {wp[x]['band_fix']}->{wn[x]['band_fix']})" for x in worse[:5]) if worse else ""))
print()
print("| box | shots | n | MAE v3 / v2 / v2+notch | along-track hp RMS v3 / v2 / v2+notch |")
print("|---|---|---|---|---|")
for b in boxes:
    P, N = ld(b, "pcf80"), ld(b, "cfn80")
    if not (P and N): continue
    for k in ("all/all", "all/steep20", "all/flat5"):
        p, n = P.get("shots", {}).get(k), N.get("shots", {}).get(k)
        if not (p and n): continue
        tp, tn = p.get("track_hp_rms") or ["-", "-"], n.get("track_hp_rms") or ["-", "-"]
        print(f"| {b} | {k} | {n['n']} | {p['mae'][0]:.3f} / {p['mae'][1]:.3f} / {n['mae'][1]:.3f} | {tp[0]} / {tp[1]} / {tn[1]} |")
print()
print("| box | N-S sections | crest change vs v3 (mean / p90 abs / max abs): v2 ; v2+notch | layering RMS v3 / v2 / v2+notch |")
print("|---|---|---|---|")
for b in boxes:
    P, N = ld(b, "pcf80"), ld(b, "cfn80")
    if not (P and N): continue
    p, n = P.get("sections_NS"), N.get("sections_NS")
    if p and n: print(f"| {b} | {n['n']} | {p['dcrest_mean']:+.2f} / {p['dcrest_abs_p90']:.2f} / {p['dcrest_absmax']:.2f} ; {n['dcrest_mean']:+.2f} / {n['dcrest_abs_p90']:.2f} / {n['dcrest_absmax']:.2f} | {p['layer'][0]:.3f} / {p['layer'][1]:.3f} / {n['layer'][1]:.3f} |")
