#!/usr/bin/env python3
"""predict_queue_v3.py --queue TSV --model-dir DIR --out-root DIR [--nproc 8] [--threads 5] [--gpus 0|2] [--limit N]

Lean frame-prediction runner for the v3 checks (HiRISE test of a new model, 2026-10-24). Any number of workers on any partition share
one queue (TSV with the R2 queue columns idx box stem ctx inc az ...): a frame is claimed by an atomic mkdir of
OUT_ROOT/claims/<box>__<stem>. Each frame runs the production prediction command of the R2 worker (worker_r2.py) unchanged except for
the adapter, predict_frame_v3.py (resampling guards for the lattice-free decoders), and the model folder:
  python3 predict_frame_v3.py --ctx <ctx> --checkpoint <m>/checkpoints/best.pt --model-config <m>/config.json --mola-tiff <463 m mosaic>
          --pedr-db <PEDR sqlite> --inc I --az A --conf-source pedr --conf-d0 2000 --conf-gamma 1.0 --methods shot_anchor,feather
          --out OUT_ROOT/<box>/<stem>
On success (exit 0, the four rasters present, "[relief-done]" in the log) it writes .done (R2 fields + model, checkpoint md5);
otherwise .fail with the exit code and the log tail. Writes nothing outside OUT_ROOT."""
import argparse, csv, hashlib, json, os, socket, subprocess, sys, time

B = "/home/gridsan/groups/planetary_dtm/mars_dem"
MOLA = f"{B}/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff"; PEDR = f"{B}/Stored_Data_for_Codes/MOLA_SQL_Table/mola"
ADAPTER = os.environ.get("V3_ADAPTER", f"{B}/global_run/v3_2026-10/code/predict_frame_v3.py")
RASTERS = ("pred_shot_anchor.tif", "pred_feather.tif", "mola_on_grid.tif", "ctx_on_grid.tif")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True); ap.add_argument("--model-dir", required=True); ap.add_argument("--out-root", required=True)
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--threads", type=int, default=5); ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of the queue (dry runs)")
    a = ap.parse_args()
    ckpt, cfg = f"{a.model_dir}/checkpoints/best.pt", f"{a.model_dir}/config.json"
    for p in (ckpt, cfg, ADAPTER, MOLA): assert os.path.exists(p), p
    ck_md5 = md5(ckpt); tag = os.path.basename(a.model_dir.rstrip("/"))
    rows = list(csv.DictReader(open(a.queue), delimiter="\t"))
    if a.limit: rows = rows[:a.limit]
    os.makedirs(f"{a.out_root}/claims", exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local"); host = socket.gethostname(); part = os.environ.get("SLURM_JOB_PARTITION", "")
    running = {}; it = iter(rows); slot_free = list(range(a.nproc)); exhausted = False

    def start(r, slot):
        d = f"{a.out_root}/{r['box']}/{r['stem']}"
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
                   CUDA_VISIBLE_DEVICES=(str(slot % a.gpus) if a.gpus else ""))
        env.pop("PROJ_DATA", None); env.pop("PROJ_LIB", None)
        cmd = ["python3", ADAPTER, "--ctx", r["ctx"], "--checkpoint", ckpt, "--model-config", cfg, "--mola-tiff", MOLA, "--pedr-db", PEDR,
               "--inc", r["inc"], "--az", r["az"], "--conf-source", "pedr", "--conf-d0", "2000", "--conf-gamma", "1.0",
               "--methods", "shot_anchor,feather", "--out", d]
        log = open(f"{d}/.predict.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=B, stdin=subprocess.DEVNULL)
        running[p] = (r, d, slot, time.time(), log)

    while True:
        while slot_free and not exhausted:
            try: r = next(it)
            except StopIteration: exhausted = True; break
            d = f"{a.out_root}/{r['box']}/{r['stem']}"
            if os.path.exists(f"{d}/.done"): continue
            try: os.mkdir(f"{a.out_root}/claims/{r['box']}__{r['stem']}")
            except FileExistsError: continue
            start(r, slot_free.pop())
        if not running: break
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            r, d, slot, t0, log = running.pop(p); log.close(); slot_free.append(slot)
            txt = open(f"{d}/.predict.log", errors="replace").read()
            ok = p.returncode == 0 and all(os.path.exists(f"{d}/{f}") for f in RASTERS) and "[relief-done]" in txt
            import re as _re                                  # v3 guard: a confidence input of ~0 means the PEDR query failed
            cm = [float(x) for x in _re.findall(r"\[conf\] source=pedr .*?mean=([0-9.]+)", txt)]
            if ok and (not cm or min(cm) < 0.05):
                ok = False; reason = f"confidence mean {cm} (< 0.05 or missing): PEDR query failure"
            else:
                reason = ""
            rec = dict(job=job, host=host, partition=part, slot=slot, secs=round(time.time() - t0), inc=r["inc"], az=r["az"], ctx=r["ctx"],
                       model=tag, checkpoint=ckpt, checkpoint_md5=ck_md5, adapter=ADAPTER, rc=p.returncode, date=time.strftime("%Y-%m-%d %H:%M"))
            rec["conf_means"] = cm
            if ok: json.dump(rec, open(f"{d}/.done", "w"), indent=1)
            else:
                rec["log_tail"] = txt[-2000:]; rec["reason"] = reason; json.dump(rec, open(f"{d}/.fail", "w"), indent=1)
            print(f"{r['box']} {r['stem']} {'ok' if ok else 'FAIL rc=%s' % p.returncode} {rec['secs']}s", flush=True)
    print("worker done", flush=True)


if __name__ == "__main__":
    main()
