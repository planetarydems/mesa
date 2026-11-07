#!/usr/bin/env python3
# v3 COPY (2026-10-24) of fix_2026-10/rebuild/downstream/worker_r2.py: tag from the v3 CAMPAIGN.json (env V3_CAMPAIGN, default
# global_run/v3_2026-10/predict/CAMPAIGN.json), queue global_run/v3_2026-10/predict/queue_v3.tsv (env V3_QUEUE_TSV), fleet dir = V3D
# (env, default global_run/v3_2026-10/downstream), job prefix v3d_. Stages level -> fill -> mif -> qa:
#   level  V3D/stages/level_box_tag_v3.sbatch (level_frames_box_v3.py: feathered demotion, raster border as edge, v3 weights)
#   fill   V3D/stages/ctx_fill_{gpu,cpu}_tag_v3.sbatch (2026-10-28 copy of predict_v4s3d_e098/downstream's: Murray ctx, coarse + fine
#          infill, join with V3D/code/seamline_merge_v3.py + FINAL.merge.json, so the mif stage skips its re-join)
#   mif    V3D/stages/mif_box_tag_v3.sh -> box_dem_<TAG>_deseam.tif = FINAL + infill-vs-MOLA rule (NO de-seam splice)
#   qa     V3D/stages/qa_box_tag_v3.sh -> V3D/qa/<BOX>.txt (PEDR cells + step lines vs R1, lattice metric); never fails the box
#          on its verdict (only on a tool error).
# Freshness: V3D/code/fresh_tag_v3.py (deseam = FINAL + prov). Input guard: V3D/code/v3_input_guard.py. Test hook V4D_BOXES (box root).
# Previous header: fix_2026-10 RELEASE-2 COPY (2026-10-16) of /home/gridsan/groups/planetary_dtm/mars_dem/global_run/repredict_2026-10/downstream/worker.py: tag ta_implicit_d1_implrel100s1_e088r2, fleet /home/gridsan/groups/planetary_dtm/mars_dem/global_run/fix_2026-10/rebuild/downstream, queue /home/gridsan/groups/planetary_dtm/mars_dem/global_run/fix_2026-10/rebuild/queue_r2_full.tsv, level stage = /home/gridsan/groups/planetary_dtm/mars_dem/global_run/fix_2026-10/rebuild/stages/level_box_tag_r2.sbatch, job prefix r2d_.
"""worker.py --fill gpu|cpu --jobs N --deadline EPOCH [--max-boxes K] [--list F] [--state D] [--dry-run]

Per-node dispatcher of the GLOBAL downstream chain for ta_implicit_d1_implrel100s1_e088, the relief-model re-prediction
(user order 2026-10-02). = predict_v4s3d_e098/downstream/fleet/worker.py with TAG, the fleet dir (lists, state, logs)
and the job-name prefix (rld_) changed, and QUEUE_TSV = this campaign's frozen queue passed to every stage. The STAGE
SCRIPTS are the e098 chain's own, unchanged, in predict_v4s3d_e098/downstream/ (they take TAG from the environment).
Original header (user order 2026-09-20:
"run the infill and levelling pipeline across all available compute"). One box at a time on this node:

    level   bash level_box_tag.sbatch BOX          preflight quick -> warp/frame list -> preflight full -> level_only
    fill    bash ctx_fill_{gpu,cpu}_tag.sbatch BOX Murray ctx on the leveled grid -> fill_holes_coarse -> FINAL
    deseam  bash deseam_box_tag.sbatch BOX all     seam census/gate/splice -> box_dem_<TAG>_deseam.tif

The stage scripts are the ones the AOI run used, unchanged, run as plain bash inside this allocation instead of as
three chained sbatch jobs: 648 boxes x 3 jobs is far past MaxJobs 240, and a claim fleet needs no dependencies.
Each script already skips work that is fresh (fresh_tag.py stamps), so a re-run of a half-done box is cheap.

Claims (same protocol as the prediction fleet, which ran 53k frames on 12 nodes without a lost row):
  state/<BOX>/.claim   O_EXCL, holds "RUN HOST PID EPOCH"; RUN = <jobid>.<restart>, so a requeued job's own
                       orphaned claims are stale. Heartbeat-touched every GRACE/2 while a stage runs.
  state/<BOX>/.done    the box is finished (or was already fresh: skipped=true)
  state/<BOX>/.fail    a stage failed; the box is not retried by this fleet (fix it, remove .fail, rescan)
A claim is taken over only if it is older than GRACE (600 s) AND a fresh squeue shows its owner dead; since 2026-10-30 the
  renamed file must also still be that claim (inode + age re-check after the rename) so two workers cannot both take it.
Boxes whose de-seam is already fresh are marked done without running anything (fresh_tag.py check).

Per-box ALLOW_FAILED comes from boxes.tsv (the .fail frames of that box), never a fleet-wide number.
Signal deaths (rc < 0) and Lustre errors retry the stage once after 60 s; everything else fails the box.
Drain files: STOP or DRAIN_<partition> in the fleet dir. YIELD: if another job of this user is pending on this
partition for the node cap and this worker is the youngest v4d_* worker there, finish the box and exit 12.
Exit codes: 0 list exhausted, 10 deadline, 11 FS errors, 12 yield, 20 drain, 21 low disk space, 75 broken node.
SPACE GUARD (2026-10-05): before each new box the group quota is read; below MIN_FREE_TIB (default 1.5, env
RLD_MIN_FREE_TIB) the worker finishes the box in flight, claims nothing more and exits 21 (no resubmit).
"""
import argparse, fcntl, glob, json, os, re, signal, socket, subprocess, sys, time

B = "/home/gridsan/groups/planetary_dtm/mars_dem"
V = f"{B}/global_run/v3_2026-10"
TAG = json.load(open(os.environ.get("V3_CAMPAIGN", f"{V}/predict/CAMPAIGN.json")))["tag"]
assert TAG not in ("ta_implicit_d1_implrel100s1_e088", "ta_implicit_d1_implrel100s1_e088r2"), "v3 tag must be new"
DS = f"{B}/global_run/predict_v4s3d_e098/downstream"
FLEET = os.environ.get("V3D", f"{V}/downstream")
sys.path.insert(0, f"{FLEET}/code")
from v3_input_guard import check as level_inputs_match
QUEUE = os.environ.get("V3_QUEUE_TSV", f"{V}/predict/queue_v3.tsv")
LUSTRE_RE = re.compile(r"Errno 108|transport endpoint shutdown|Object is remote|No such device|"
                       r"Input/output error|Stale file handle|Cannot send after transport")
GRACE = float(os.environ.get("V4D_GRACE", 600))
MIN_FREE_TIB = float(os.environ.get("RLD_MIN_FREE_TIB", 1.5))   # space guard (2026-10-05): stop claiming boxes below this
JOB = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
RUN = f"{JOB}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
PART = os.environ.get("SLURM_JOB_PARTITION", "local")
HOST = socket.gethostname()
# per-stage wall-clock kill (the scripts' own SLURM limits were 8 h level / 8-12 h fill / 8 h deseam; the AOI boxes,
# the largest in the fleet, took 1.3 h / 0.7 h / 0.1 h)
TIMEOUT = {"level": 6 * 3600, "ctx": 3 * 3600, "fill": 10 * 3600, "mif": 2 * 3600, "qa": 2 * 3600}
# post-condition of each stage: rc 0 is not enough (build_box_final_tag.sh level_only WARNs and exits 0 when the mosaic
# exists but leveled_<TAG>/SOLVE.json is missing, e.g. after a kill between the publish and the stamp), and a de-seam
# whose diagnostic report step fails still leaves a complete, stamped raster.
STAGE_NODES = {"level": ("frames",), "ctx": ("ctx",), "fill": ("FINAL", "prov"), "mif": ("deseam",), "qa": ()}
LOG = None
# Readiness gate (v3, 2026-10-24): the downstream fleet trails the prediction fleet, so a box is claimed only when every queue frame
# of it is final in preds_<TAG> (.done or .fail, no .claim); the preflight then decides (ALLOW_FAILED). Checked at most every
# READY_RECHECK s per box; ready boxes are cached.
READY_RECHECK = float(os.environ.get("V3D_READY_RECHECK", 600))
CHECK_DONE = os.environ.get("V3D_CHECK_DONE") == "1"     # 2026-10-28: re-validate finished boxes' inputs on every pass (slow audit)
_ready, _ready_t, _qstems = set(), {}, None


def preds_ready(box):
    global _qstems
    if box in _ready:
        return True
    if time.time() - _ready_t.get(box, 0.0) < READY_RECHECK:
        return False
    _ready_t[box] = time.time()
    if _qstems is None:
        import csv
        _qstems = {}
        with open(QUEUE) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                _qstems.setdefault(r["box"], []).append(r["stem"])
    stems = _qstems.get(box, [])
    if not stems:
        return False
    pd = f"{box_run(box)}/preds_{TAG}"
    for st in stems:
        d = f"{pd}/{st}"
        if os.path.exists(f"{d}/.claim") or not (os.path.exists(f"{d}/.done") or os.path.exists(f"{d}/.fail")):
            return False
    _ready.add(box)
    say(f"READY {box}: all {len(stems)} queue frames final")
    return True


def say(msg):
    try:
        LOG.write(f"[{time.strftime('%F %T')}] {msg}\n"); LOG.flush()
    except (OSError, AttributeError):
        pass


def box_run(box):
    return f"{os.environ.get('V4D_BOXES', B + '/global_run/boxes')}/{box}/box_run"


_alive = {"t": 0.0, "v": None}


def alive_jobs(max_age=30.0):
    if time.time() - _alive["t"] < max_age and _alive["v"] is not None:
        return _alive["v"]
    v = None
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", "wmitchell"), "-h", "-o", "%A %i"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            v = set(r.stdout.split()) | {JOB} | (set(os.environ.get("V4D_TEST_ALIVE", "").split(",")) - {""})
    except Exception:
        v = None
    _alive.update(t=time.time(), v=v)
    return v


def should_yield():
    """True if a non-rld job of this user waits on our partition for the node cap and we are the youngest rld worker."""
    user = os.environ.get("USER", "wmitchell")
    try:
        pend = subprocess.run(["squeue", "-u", user, "-h", "-t", "PENDING", "-p", PART, "-o", "%i|%j|%r"],
                              capture_output=True, text=True, timeout=60)
        run = subprocess.run(["squeue", "-u", user, "-h", "-t", "RUNNING", "-p", PART, "-o", "%A|%j|%S"],
                             capture_output=True, text=True, timeout=60)
        if pend.returncode or run.returncode:
            return False, ""
        waiting = [l for l in pend.stdout.splitlines() if l.count("|") == 2 and not l.split("|")[1].startswith("v3d_")
                   and "AssocGrpNodeLimit" in l.split("|")[2]]
        mine = sorted((l.split("|")[2], l.split("|")[0]) for l in run.stdout.splitlines()
                      if l.count("|") == 2 and l.split("|")[1].startswith("v3d_"))
        return bool(waiting) and bool(mine) and mine[-1][1] == JOB, ";".join(waiting)[:200]
    except Exception:
        return False, ""


def free_tib():
    """Free space under the group quota in TiB, or None if it cannot be read (never stop on an unreadable quota)."""
    try:
        r = subprocess.run(["lfs", "quota", "-q", "-g", "planetary_dtm", "/home/gridsan/wmitchell"],
                           capture_output=True, text=True, timeout=120)
        nums = [int(t) for t in r.stdout.replace("*", " ").split() if t.isdigit()]   # kbytes, quota, limit, files, ...
        if r.returncode != 0 or len(nums) < 3:
            return None
        used, limit = nums[0], nums[2]                 # kbytes used, soft quota, hard limit
        if limit <= 0 or used <= 0:
            return None
        return max(limit - used, 0) / 1024.0 ** 3
    except Exception:
        return None


def owner_alive(run, alive):
    job = run.split(".")[0]
    if job == JOB:
        return run == RUN or "." not in run            # same job id, another run = requeued = dead
    return job in alive


def try_claim(d):
    """-> 'final' | 'held_live' | 'held_young' | 'claimed'. Raises OSError on file-system trouble."""
    if os.path.exists(f"{d}/.fail"):
        return "final"
    if os.path.exists(f"{d}/.done"):
        # 2026-10-28: a finished box is final; its inputs are NOT re-validated on every pass (v3_input_guard's context-signature
        # check costs 10-20 min per levelled box, so a scan over the finished boxes took hours before the first claim). Input
        # changes reach a finished box through the redo procedure (redo_box_v3.sh clears .done). V3D_CHECK_DONE=1 restores the audit.
        if not CHECK_DONE:
            return "final"
        return "final" if already_fresh(os.path.basename(d)) else "stale_final"
    os.makedirs(d, exist_ok=True)
    cp = f"{d}/.claim"
    for _ in range(3):
        try:
            fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            try:
                st0 = os.stat(cp); mt = st0.st_mtime; words = open(cp).read().split()
            except FileNotFoundError:
                continue                                # released meanwhile: try again
            if time.time() - mt < GRACE:
                return "held_young"
            alive = alive_jobs()
            if alive is None:
                return "held_young"                     # cannot tell: never take over
            if words and owner_alive(words[0], alive):
                return "held_live"
            try:                                        # dead or unparseable owner: one atomic rename wins
                stale = f"{cp}.stale.{RUN}.{os.getpid()}.{int(time.time())}"
                os.rename(cp, stale)
            except OSError:
                return "held_young"
            # 2026-10-30: alive_jobs() above can take seconds (squeue), so another worker may have taken the dead claim
            # over and written its own fresh .claim in between; the rename then moved THAT file (seen at the 15:03 restart:
            # 5 boxes got a spurious .fail from the loser's stage-lock rc=7). Check the inode and age of what was renamed
            # and give it back (link, never overwrite) when it is not the stale claim judged above.
            try:
                st1 = os.stat(stale)
            except OSError:
                return "held_young"
            if st1.st_ino != st0.st_ino or time.time() - st1.st_mtime < GRACE:
                try:
                    os.link(stale, cp); os.unlink(stale)
                except OSError:
                    pass
                say(f"TAKEOVER aborted (claim re-taken by another worker meanwhile): {d}")
                return "held_young"
            say(f"TAKEOVER claim {words[:2] if words else 'EMPTY'}: {d}")
            continue
        try:
            os.write(fd, f"{RUN} {HOST} {os.getpid()} {int(time.time())}\n".encode())
            os.close(fd)
        except OSError:
            try: os.close(fd)
            except OSError: pass
            try: os.remove(cp)
            except OSError: pass
            raise
        if os.path.exists(f"{d}/.done") or os.path.exists(f"{d}/.fail"):
            os.remove(cp)
            return "final" if not os.path.exists(f"{d}/.done") or not CHECK_DONE or already_fresh(os.path.basename(d)) else "stale_final"
        return "claimed"
    return "held_young"


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def fresh_nodes(box, nodes):
    """(fresh?, why) for fresh_tag.py products of this box. Freshness is recursive down to the roots."""
    try:
        if not nodes:
            return True, "no freshness node"
        r = subprocess.run(["python3", f"{FLEET}/code/fresh_tag_v3.py", "check", box_run(box), TAG, *nodes],
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]
    except Exception as e:
        return False, f"fresh_tag check failed: {e}"


def already_fresh(box):
    """True if this box's de-seam is fresh, so every upstream product is fresh too: nothing to do.
    2026-10-28: cheap tests first (QA record, then the de-seam freshness stamp); the input guard (10-20 min on a levelled box)
    runs only when both pass. Same answer as before (QA and de-seam and guard), without the guard's cost on unfinished boxes."""
    qa = f"{os.environ.get('QA_DIR', FLEET + '/qa')}/{box}.txt"
    if not os.path.exists(qa) or not fresh_nodes(box, ("deseam",))[0]:
        return False
    ok, why = level_inputs_match(box, TAG, QUEUE, box_run(box))
    if not ok:
        say(f"INPUT MISMATCH {box}: {why}")
        return False
    return True


def clear_my_locks(box):
    """Drop tag locks of this box that THIS job owns. A SIGKILL skips the stage scripts' EXIT trap, and tag_guard only
    takes a lock over when its owner job is gone from squeue -- so an orphan of a still-running worker would make the
    box unusable for every other worker until this job ends."""
    for k in ("level", "fill"):
        lk = f"{box_run(box)}/.{k}_{TAG}.lock"
        try:
            if open(f"{lk}/owner").read().split()[0] == JOB:
                os.remove(f"{lk}/owner")
                os.rmdir(lk)
                say(f"released orphaned {k} lock of {box}")
        except OSError:
            pass


def stage_cmd(stage, row, args):
    """(argv, extra env) for one stage of one box."""
    box = row["box"]
    env = {"TAG": TAG, "ALLOW_FAILED": row["failed"], "QUEUE_TSV": QUEUE, "V3D": FLEET}
    if os.environ.get("V4D_BOXES"):
        env["BOXES_ROOT"] = os.environ["V4D_BOXES"]
    if stage == "level":
        env["SLURM_CPUS_PER_TASK"] = str(args.jobs)
        return ["bash", f"{FLEET}/stages/level_box_tag_v3.sbatch", box], env
    if stage in ("fill", "ctx"):
        env.update(MURRAY_DIR=row["murray"], WALL_H=str(int(TIMEOUT[stage] // 3600)))
        if stage == "ctx":                       # Murray ctx on the leveled grid only (inpainter retrain, 2026-09-21)
            env["CTX_ONLY"] = "1"
        return ["bash", f"{FLEET}/stages/ctx_fill_{args.fill}_tag_v3.sbatch", box], env   # v3 copy (2026-10-28): joins with seamline_merge_v3
    if stage == "mif":
        return ["bash", f"{FLEET}/stages/mif_box_tag_v3.sh", box], env
    env.update(NP=str(min(args.jobs, 16)))
    return ["bash", f"{FLEET}/stages/qa_box_tag_v3.sh", box], env


def pilot_stage_image_qa(row, stage, args):
    """Immediate real-raster snapshots, diagnostic only; never fail a stage."""
    root = f"{B}/global_run/fix_2026-10/rebuild"
    if (QUEUE != f"{root}/pilot_guardfix_20261017/queue.tsv" or
        not args.state.endswith("/state_pilot_guardfix_20261017")):
        return
    try:
        with open(f"{root}/pilot_guardfix_20261017/image_qa.lock", "a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                say(f"WARN pilot image QA already running; refresh deferred after {row['box']}/{stage}")
                return
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            command = ["python3", f"{root}/quality_monitor/render_qa.py",
                       f"{root}/quality_monitor/config.json", f"{root}/quality_monitor/snapshots"]
            say(f"PILOT image QA after fresh {row['box']}/{stage}")
            result = subprocess.run(command, stdout=LOG, stderr=subprocess.STDOUT, env=env, timeout=300)
            if result.returncode:
                say(f"WARN pilot image QA rc={result.returncode}; stage remains valid, refresh deferred")
    except Exception as exc:
        say(f"WARN pilot image QA failed; refresh deferred: {exc}")


class Box:
    """Runs level -> fill -> deseam for one box, one stage at a time, heartbeating its claim."""

    def __init__(self, row, args, state):
        self.row, self.args, self.d = row, args, f"{state}/{row['box']}"
        self.log = f"{FLEET}/logs/box_{row['box']}.log"
        self.stages = list(args.stages)
        self.times, self.t0, self.hb = {}, time.time(), 0.0
        self.attempt, self.retry_at, self.p, self.fh, self.stage, self.rc = 0, None, None, None, None, 0
        self.start_stage(self.stages.pop(0))

    def start_stage(self, stage=None):
        """stage given: begin it (attempt 1). stage omitted: retry the current one."""
        if stage is not None:
            self.stage, self.attempt = stage, 1
        else:
            self.attempt += 1
        self.retry_at = None
        cmd, extra = stage_cmd(self.stage, self.row, self.args)
        env = dict(os.environ, **extra)
        env.pop("PROJ_DATA", None); env.pop("PROJ_LIB", None)
        self.heartbeat(force=True)
        self.fh = open(self.log, "a", buffering=1)
        self.fh.write(f"\n===== {self.stage} {self.row['box']} attempt={self.attempt} run={RUN} host={HOST} "
                      f"{time.strftime('%F %T')} =====\n")
        self.mark = self.fh.tell()
        self.p = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.STDOUT, env=env, cwd=B,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        self.ts = time.time()
        say(f"STAGE {self.stage} {self.row['box']} attempt={self.attempt} pid={self.p.pid}")

    def heartbeat(self, force=False):
        if force or time.time() - self.hb > GRACE / 2:
            try: os.utime(f"{self.d}/.claim")
            except OSError: pass
            self.hb = time.time()

    def tail(self):
        """The WHOLE output of the current attempt (LUSTRE_RE must see an error printed hours ago, not just the end)."""
        try:
            with open(self.log, errors="replace") as fh:
                fh.seek(self.mark)
                return fh.read()
        except OSError:
            return ""

    def poll(self):
        """None while running; True when the box is done; False when it failed."""
        self.heartbeat()
        if self.retry_at is not None:
            if time.time() < self.retry_at:
                return None
            self.start_stage(); return None
        rc = self.p.poll()
        if rc is None:
            if time.time() - self.ts > TIMEOUT[self.stage]:
                try: os.killpg(self.p.pid, signal.SIGKILL)
                except OSError: pass
                try: self.p.wait(timeout=120)
                except subprocess.TimeoutExpired: say(f"UNKILLABLE pid {self.p.pid} (D-state?) {self.row['box']}")
                self.fh.close()
                clear_my_locks(self.row["box"])
                return self.fail(rc=-9, why="timeout")
            return None
        self.fh.close()
        el = round(time.time() - self.ts)
        if rc == 0:
            ok, why = fresh_nodes(self.row["box"], STAGE_NODES[self.stage])
            if not ok:
                return self.fail(rc=0, why=f"{self.stage} exited 0 but {'+'.join(STAGE_NODES[self.stage])} is not fresh: {why}")
            self.times[self.stage] = el
            say(f"OK {self.stage} {self.row['box']} {el}s")
            if self.stages:
                self.start_stage(self.stages.pop(0))
                return None
            return self.succeed()
        if self.stage == "deseam" and fresh_nodes(self.row["box"], STAGE_NODES["deseam"])[0]:
            self.times[self.stage] = el          # the splice finished and is stamped; only the diagnostic report failed
            say(f"WARN deseam {self.row['box']} rc={rc} but the de-seam raster is fresh (report stage); accepting the box")
            return self.succeed()
        txt = self.tail()
        if (rc < 0 or LUSTRE_RE.search(txt)) and self.attempt < 2:
            say(f"RETRY {self.stage} {self.row['box']} rc={rc} ({'signal' if rc < 0 else 'lustre'})")
            self.retry_at = time.time() + 60
            return None
        return self.fail(rc=rc, why="error")

    def succeed(self):
        el = round(time.time() - self.t0)
        if os.environ.get("V4D_DELETE_STALE") == "1":
            # re-fill pass (2026-09-21): products quarantined by fresh_tag.py (<file>.stale.<epoch>) are reproducible
            # copies of ~3.5 GB each; once this box's chain is fresh again they only cost quota
            n = 0
            for f in glob.glob(f"{box_run(self.row['box'])}/*{TAG}*.stale.*"):
                try:
                    if os.path.isfile(f): os.remove(f); n += 1
                except OSError: pass
            say(f"deleted {n} stale product file(s) of {self.row['box']}")
        write_json(f"{self.d}/.done", dict(job=JOB, run=RUN, host=HOST, partition=PART, fill=self.args.fill,
                                           secs=el, stages=self.times, frames=self.row["frames"],
                                           failed_frames=self.row["failed"], date=time.strftime("%F %T")))
        self.release()
        say(f"DONE {self.row['box']} {el}s " + " ".join(f"{k}={v}s" for k, v in self.times.items()))
        return True

    def fail(self, rc, why):
        el = round(time.time() - self.t0)
        self.rc = rc
        tail = " | ".join(self.tail().strip().splitlines()[-4:])[-600:]
        write_json(f"{self.d}/.fail", dict(stage=self.stage, rc=rc, why=why, job=JOB, run=RUN, host=HOST,
                                           secs=el, stages=self.times, tail=tail, date=time.strftime("%F %T")))
        self.release()
        say(f"FAIL {self.row['box']} stage={self.stage} rc={rc} {why} {el}s :: {tail[:300]}")
        return False

    def abandon(self):
        """Stop this box without a verdict: the claim goes, no .fail, so another worker redoes it (stamps make it cheap)."""
        try:
            if self.p is not None and self.p.poll() is None:
                os.killpg(self.p.pid, signal.SIGKILL); self.p.wait(timeout=60)
        except Exception:
            pass
        try: self.fh.close()
        except Exception: pass
        clear_my_locks(self.row["box"])
        self.release()

    def release(self):
        try: os.remove(f"{self.d}/.claim")
        except OSError: pass


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--jobs", type=int, required=True, help="cores for the level solve and the de-seam")
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--max-boxes", type=int, default=0)
    ap.add_argument("--list", default=f"{FLEET}/boxes.tsv")
    ap.add_argument("--state", default=f"{FLEET}/state")
    ap.add_argument("--stages", default="level,fill,mif,qa")
    ap.add_argument("--est-secs", type=float, default=2.0 * 3600, help="initial per-box estimate for the deadline")
    ap.add_argument("--dry-run", action="store_true", help="claim nothing; print what would run")
    args = ap.parse_args()
    args.stages = [s for s in args.stages.split(",") if s]
    unknown = [s for s in args.stages if s not in TIMEOUT]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; known: {','.join(TIMEOUT)}")
    LOG = open(f"{FLEET}/logs/worker_{PART}_{JOB}{os.environ.get('V3D_WORKER', '')}.log", "a", buffering=1)   # 2 workers per GPU node

    rows = []
    with open(args.list) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if line.strip():
                rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    for f in (f"{FLEET}/stages/level_box_tag_v3.sbatch", f"{FLEET}/stages/ctx_fill_{args.fill}_tag_v3.sbatch", f"{FLEET}/stages/mif_box_tag_v3.sh",
              f"{FLEET}/stages/qa_box_tag_v3.sh", f"{FLEET}/code/fresh_tag_v3.py", args.list, QUEUE):
        if not os.access(f, os.R_OK):
            say(f"PREFLIGHT unreadable {f} on {HOST}"); sys.exit(75)
    pf = subprocess.run(["python3", "-c", "import numpy, scipy, rasterio; print('ok')"],
                        capture_output=True, text=True, cwd=B, timeout=600)
    if pf.returncode != 0:      # node-local anaconda breakage (d-5-14-3: libffi / libsqlite3 'file too short')
        say(f"PREFLIGHT import failed on {HOST}: {(pf.stderr.strip().splitlines() or [pf.returncode])[-1]}")
        sys.exit(75)
    say(f"START v3d tag={TAG} queue={QUEUE} run={RUN} part={PART} host={HOST} fill={args.fill} jobs={args.jobs} stages={','.join(args.stages)} "
        f"deadline={time.strftime('%F %T', time.localtime(args.deadline))} boxes={len(rows)}")

    cur, cursor, started, times, stop_reason, code = None, 0, 0, [], None, None
    fs_errors, passes, pass_claimed, pass_young, pass_fs, fast_fails = 0, 1, 0, 0, 0, 0
    last_yield_check = time.time()
    stale_final_rows = set()
    while True:
        if os.path.exists(f"{FLEET}/STOP") or os.path.exists(f"{FLEET}/DRAIN_{PART}"):
            stop_reason = stop_reason or "drain"
        if stop_reason is None and time.time() - last_yield_check > 600 and os.environ.get("V3D_NO_YIELD") != "1":
            last_yield_check = time.time(); y, who = should_yield()
            if y:
                stop_reason = "yield"; say(f"YIELD: waiting job(s) {who}; finishing this box then exit 12")
        est = max(sorted(times)[int(len(times) * 0.9)] * 1.3 if len(times) >= 4 else args.est_secs, 1800)
        if stop_reason is None and time.time() + est > args.deadline:
            stop_reason = "deadline"
        if args.max_boxes and started >= args.max_boxes:
            stop_reason = stop_reason or "max-boxes"
        if stop_reason is None and cur is None and not args.dry_run:
            ft = free_tib()
            if ft is not None and ft < MIN_FREE_TIB:
                stop_reason = "space"; say(f"SPACE: group quota has {ft:.2f} TiB free (< {MIN_FREE_TIB}); claiming no more boxes, exit 21")

        while cur is None and stop_reason is None:
            if cursor >= len(rows):
                break
            row = rows[cursor]; cursor += 1
            d = f"{args.state}/{row['box']}"
            if not args.dry_run and "level" in args.stages and not os.path.exists(f"{d}/.done") and not os.path.exists(f"{d}/.fail") \
                    and not preds_ready(row["box"]):
                pass_young += 1                         # predictions still running: not claimable yet, keep rescanning
                continue
            try:
                st = try_claim(d) if not args.dry_run else "dry"
                fs_errors = 0
            except OSError as e:
                fs_errors += 1; pass_fs += 1; say(f"FS-ERROR claim {row['box']}: {e}")
                if fs_errors >= 20: stop_reason = "fs-errors"
                break
            if st == "stale_final":
                stale_final_rows.add(row["box"])
                write_json(f"{d}/manifest_mismatch_{JOB}.json", dict(box=row["box"], queue=QUEUE,
                    job=JOB, reason="existing .done does not match current input identity or fresh products; use a new manifest-specific state and review/quarantine stale r2 outputs", date=time.strftime("%F %T")))
                say(f"REFUSE {row['box']}: stale .done retained; no products changed")
                continue
            if st == "held_young":
                pass_young += 1
            if st == "dry":
                _ready_t.pop(row["box"], None)
                print(f"would run {args.stages} on {row['box']} (frames {row['frames']}, allow_failed {row['failed']}, "
                      f"fill {args.fill}, preds_ready={preds_ready(row['box'])}, fresh={already_fresh(row['box'])})")
                continue
            if st != "claimed":
                continue
            if already_fresh(row["box"]):
                write_json(f"{d}/.done", dict(skipped=True, reason="deseam already fresh", job=JOB, run=RUN,
                                              host=HOST, date=time.strftime("%F %T")))
                try: os.remove(f"{d}/.claim")
                except OSError: pass
                say(f"SKIP {row['box']}: de-seam already fresh")
                continue
            try:
                cur = Box(row, args, args.state)
            except OSError as e:
                fs_errors += 1; pass_fs += 1; say(f"FS-ERROR start {row['box']}: {e}")
                try: os.remove(f"{d}/.claim")
                except OSError: pass
                if fs_errors >= 20: stop_reason = "fs-errors"
                break
            started += 1; pass_claimed += 1

        if cur is not None:
            try:
                res = cur.poll()
            except OSError as e:
                say(f"FS-ERROR poll {cur.row['box']}: {e}"); cur.abandon(); cur = None
                fs_errors += 1; pass_fs += 1
                if fs_errors >= 20: stop_reason = stop_reason or "fs-errors"
                res = None
            if res is not None:
                if res:
                    times.append(round(time.time() - cur.t0))
                elif time.time() - cur.t0 < 300 and (cur.rc < 0 or cur.rc in (1, 126, 127)):
                    fast_fails += 1     # node breakage looks like this; the stage scripts' data refusals (2,4,5,6,7,10,11,12) do not
                cur = None
        if cur is not None and time.time() > args.deadline:
            # the stage kill budget (up to 10 h) is far longer than the wall-clock margin: never let SLURM kill this job,
            # or the resubmit at the end of the sbatch never runs and the fleet loses a node for good.
            say(f"DEADLINE with {cur.row['box']} in stage {cur.stage}: abandoning it (claim released, no verdict)")
            cur.abandon(); cur = None
            stop_reason = stop_reason or "deadline"
        if not times and fast_fails >= 3 and stop_reason is None:
            stop_reason = "broken-node"
            say(f"CIRCUIT BREAKER: {fast_fails} fast failures and no success on {HOST}")
        if cur is None and stop_reason:
            break
        if cur is None and cursor >= len(rows):
            # a clean exit needs a pass that claimed nothing, saw no young claim AND hit no FS error: a row skipped
            # by an FS error is neither .done nor .fail, so exiting 0 here would drop it for the whole campaign
            if pass_claimed == 0 and pass_young == 0 and pass_fs == 0:
                code = 0; stop_reason = "list exhausted"; break
            say(f"PASS {passes} done: claimed {pass_claimed}, young/uncertain claims {pass_young}, fs-errors {pass_fs}; rescanning")
            if pass_claimed == 0:
                time.sleep(120)
            cursor, passes, pass_claimed, pass_young, pass_fs = 0, passes + 1, 0, 0, 0
            continue
        time.sleep(5)
    if code is None:
        code = {"broken-node": 75, "drain": 20, "deadline": 10, "fs-errors": 11, "yield": 12, "space": 21, "max-boxes": 0}.get(stop_reason, 0)
    # Exhausting the list is not success when data/preflight failures were recorded.
    # 30 is a durable data failure, not a crash to resubmit on another node.
    failed_rows = sum(os.path.exists(f"{args.state}/{row['box']}/.fail") for row in rows) + len(stale_final_rows)
    if code == 0 and failed_rows:
        code = 30
        stop_reason += " with failed boxes"
    # The queued pilot sbatch reads this Python worker at execution time. Run
    # full-tile diagnostics inside that same compute allocation, only after both
    # frozen pilot chains are complete. Statistics never grant release acceptance.
    pilot_queue = f"{B}/global_run/fix_2026-10/rebuild/pilot_guardfix_20261017/queue.tsv"
    pilot_boxes = {"E-140_Np80", "Ep170_N-10"}
    if (code == 0 and QUEUE == pilot_queue and
        args.state.endswith("/state_pilot_guardfix_20261017") and
        rows and {r["box"] for r in rows}.issubset(pilot_boxes) and
        all(os.path.exists(f"{args.state}/{box}/.done") and
            not os.path.exists(f"{args.state}/{box}/.fail") for box in pilot_boxes)):
        say("PILOT full-tile QA starting; acceptance remains pending manual review")
        try:
            qa = subprocess.run(["python3", f"{B}/global_run/fix_2026-10/rebuild/qa_pilot_fullscan.py"],
                                stdout=LOG, stderr=subprocess.STDOUT, timeout=1800)
            if qa.returncode:
                code = 31
                stop_reason += "; quantitative QA could not validate all required metadata"
        except Exception as exc:
            say(f"PILOT full-tile QA failed: {exc}")
            code = 31
            stop_reason += "; quantitative QA execution failed"
    say(f"END run={RUN} started={started} ok={len(times)} failed_rows={failed_rows} passes={passes} reason={stop_reason} exit={code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
