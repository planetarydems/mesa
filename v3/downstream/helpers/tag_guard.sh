# tag_guard.sh -- helpers SOURCED by the tag-aware downstream chain (never executed on its own).
# global_run/predict_v4s3d_e098/downstream, 2026-09-16.
#
#   tag_check_writable TAG     refuse empty/odd tags and the existing production tags (this chain never writes their products)
#   tag_refuse_shift OUT TAG   refuse if an output-side PEDR shift exists for TAG (preds_shifted_TAG*, leveled_TAG.SHIFTED*):
#                              the new predictions were made from aligned_pedr inputs, i.e. they are ALREADY in the PEDR
#                              frame; build_box_final.sh would silently level from those VRTs = a double registration
#   box_kind BOX               eq | polar (same rule as build_queue.py: -30 <= south edge <= 20 is equatorial global eqc;
#                              everything else is a per-box sinusoidal "polar-band" box)
#   murray_dir_for BOX         murray_lab (5 m, equatorial) | murray_lab_18m (polar campaign store)
#   tag_lock DIR / tag_unlock DIR   mkdir lock with an owner file; a lock whose owner SLURM job is gone is taken over.
#                              Owner token = $TAG_LOCK_OWNER (exported by an outer holder) or $SLURM_JOB_ID or local<pid>;
#                              a child process with the same token re-enters the lock and its tag_unlock leaves it alone.
#                              Takeover (review finding D2, 2026-09-17) is serialised by a second mkdir mutex DIR.takeover:
#                              under it the owner line is re-read and must still be the dead owner's line read before
#                              (else another job already took the lock -> refuse); the lock dir itself is never removed,
#                              the owner file is replaced atomically (write owner.new.<pid>, mv). A takeover is also refused
#                              when the squeue listing does not contain this job's own SLURM_JOB_ID (listing unreliable).
#                              A DIR.takeover left by a job killed inside the few-ms critical section blocks takeovers
#                              (fail-safe): remove it by hand once no job is taking the lock over.
# Freshness of derived products (review finding D1) lives in fresh_tag.py (provenance stamps + quarantine).
DS_BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS_DIR=$DS_BASE/global_run/predict_v4s3d_e098/downstream
DEFAULT_TAG=ta_implicit_d1_v4s3d_e098
PROTECTED_TAGS="ta_implicit_d1 ta_implicit_d1_ftpolar_lr1e-4"

tag_check_writable() {
  local t=${1:-}
  [ -n "$t" ] || { echo "FATAL: empty TAG" >&2; return 2; }
  case "$t" in *[!A-Za-z0-9._-]*) echo "FATAL: TAG has characters outside [A-Za-z0-9._-]: '$t'" >&2; return 2 ;; esac
  local p
  for p in $PROTECTED_TAGS; do
    if [ "$t" = "$p" ]; then
      echo "FATAL: TAG=$t is an existing production tag; the tag-aware chain never writes its products" >&2
      return 2
    fi
  done
  return 0
}

tag_refuse_shift() {
  local out=${1:?box_run dir} t=${2:?tag} hits
  hits=$( { compgen -G "$out/preds_shifted_${t}*"; compgen -G "$out/leveled_${t}.SHIFTED*"; } 2>/dev/null )
  if [ -n "$hits" ]; then
    echo "FATAL: output-side PEDR shift artefacts exist for TAG=$t. Predictions from aligned_pedr are already in the" >&2
    echo "       PEDR frame; levelling from shifted VRTs would register them twice. Refusing. Found:" >&2
    echo "$hits" | sed 's/^/         /' >&2
    return 4
  fi
  return 0
}

box_kind() {
  local s=${1##*_N}
  s=${s/p/}
  if [[ "$s" =~ ^-?[0-9]+$ ]] && [ "$s" -ge -30 ] && [ "$s" -le 20 ]; then echo eq; else echo polar; fi
}

murray_dir_for() {
  if [ "$(box_kind "$1")" = eq ]; then echo "$DS_BASE/murray_lab"; else echo "$DS_BASE/murray_lab_18m"; fi
}

tag_lock() {
  local lk=${1:?lock dir} owner line me=${TAG_LOCK_OWNER:-${SLURM_JOB_ID:-local$$}}
  if mkdir "$lk" 2>/dev/null; then
    echo "$me $(hostname) $$ $(date +%s)" > "$lk/owner"
    return 0
  fi
  line=$(head -n 1 "$lk/owner" 2>/dev/null)
  owner=$(awk '{print $1; exit}' <<< "$line")
  local opid; opid=$(awk '{print $3; exit}' <<< "$line")
  if [ -n "$owner" ] && [ "$owner" = "$me" ]; then
    if [ "$opid" = "$$" ]; then return 0; fi                    # this very shell already holds it
    TAG_LOCK_REENTERED="${TAG_LOCK_REENTERED:-} $lk"          # held by our outer process: re-enter, never release it here
    echo "[lock] re-entering $lk (owner $me)" >&2
    return 0
  fi
  local q
  if [[ "$owner" =~ ^[0-9]+$ ]] && q=$(squeue -h -u "${USER:-wmitchell}" -o %A 2>/dev/null); then   # only SLURM owners
    if [ -n "${SLURM_JOB_ID:-}" ] && ! grep -qx "$SLURM_JOB_ID" <<< "$q"; then
      echo "[lock] squeue listing does not contain this job ($SLURM_JOB_ID) -> not trusting it for a takeover of $lk" >&2
    elif ! grep -qx "$owner" <<< "$q"; then
      # D2: serialise takeovers. Two jobs that both read the dead owner must not both win: under the mutex the owner
      # line must still be exactly the dead line read above, and the lock dir is never removed (no window for mkdir).
      local mx="$lk.takeover" cur rc=1
      if mkdir "$mx" 2>/dev/null; then
        cur=$(head -n 1 "$lk/owner" 2>/dev/null)
        if [ -d "$lk" ] && [ -n "$cur" ] && [ "$cur" = "$line" ]; then
          echo "$me $(hostname) $$ $(date +%s)" > "$lk/owner.new.$$" && mv -f "$lk/owner.new.$$" "$lk/owner" && rc=0
          rm -f "$lk/owner.new.$$" 2>/dev/null
        fi
        rmdir "$mx" 2>/dev/null
        if [ "$rc" -eq 0 ]; then
          echo "[lock] took over $lk (owner job $owner is not in squeue)" >&2
          return 0
        fi
        echo "[lock] takeover of $lk lost the race: owner is now '$(awk '{print $1; exit}' <<< "$cur")' -- refusing" >&2
        return 1
      fi
      echo "[lock] another job is taking over $lk right now ($mx exists; if no job is, remove it by hand) -- refusing" >&2
      return 1
    fi
  fi
  echo "[lock] $lk is held by '${owner:-unknown}' -- refusing" >&2
  return 1
}

tag_unlock() {
  local lk=${1:?lock dir}
  case " ${TAG_LOCK_REENTERED:-} " in *" $lk "*) return 0 ;; esac
  rm -f "$lk/owner" "$lk"/owner.new.* 2>/dev/null; rmdir "$lk" 2>/dev/null || true
}
