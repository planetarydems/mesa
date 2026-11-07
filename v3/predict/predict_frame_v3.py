#!/usr/bin/env python3
"""predict_frame_v3.py -- v3 copy (2026-10-24) of global_run/repredict_2026-10/predict/predict_frame_relief.py for the lattice-free
decoders (CSAIL lattice_dec_2026-10-23: upsample_mode nearest/bilinear + 3x3 conv, optional down_mode anti-aliased max-pool).
Changes: (a) down_mode is a constructor key when config.json sets it; (b) after building, the network is rebuilt WITHOUT
upsample_mode / down_mode and the run stops if either setting did not change the architecture (the config-not-reaching-the-
constructor bug that kept the lattice-free decoder from ever being trained in August), and it stops if a non-deconv decoder
still contains ConvTranspose2d layers. Everything else is identical to the e088 production path.

Original docstring: predict_frame_relief.py -- the PRODUCTION frame adapter (src/testing/predict_single_ctx_mosaic_frame.py, run unmodified)
for the relief-conditioned model (2026-09-28).

The adapter is imported as a module and its main() is called with the caller's arguments. Two things are replaced, both
by attribute, so nothing on disk is edited:
  1. build_model_cfg.build_from_config -> build_relief(): the network is built from the RELIEF training code (RELIEF_CODE,
     default <model dir>/code/ablation = CSAIL src_v4rel/ablation, md5-checked copy) with every constructor key the relief
     eval_cells.load_model restores from config.json (_CTOR_KEYS: relief_film, relief_solve, film_condition_dim, cg_iters,
     ...). The adapter then loads the checkpoint with its own STRICT load_into, so a missing or unexpected tensor stops it.
  2. mosaic_single_ctx.predict_patch -> the same function with the relief scalar appended to the sun vector:
        rho = (log10(mstd) - 2.0) / 1.5,   mstd = max(std(MOLA tile), 5 m)
     i.e. the floored MOLA standard deviation that predict_patch itself uses to z-score the tile, exactly as the training
     loader appends relief_rho(mola_std) (src_v4rel/ablation/mars_ctx_dataset.py, STD_FLOOR_M = 5.0).
Everything else (CTX resampling, 463 m MOLA mosaic sampler, PEDR d1 conf channel, 384/192 Hann tiling, PEDR shot anchoring
and its gates, output rasters) is the production code path of the e098 product.

Usage: identical to the adapter, e.g.
  predict_frame_relief.py --ctx <aligned_pedr vrt> --checkpoint <m>/checkpoints/best.pt --model-config <m>/config.json
      --mola-tiff ... --pedr-db ... --inc I --az A --conf-source pedr --conf-d0 2000 --conf-gamma 1.0
      --methods shot_anchor,feather --out DIR
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np

B = "/home/gridsan/groups/planetary_dtm/mars_dem"
ADAPTER = f"{B}/src/testing/predict_single_ctx_mosaic_frame.py"


def _arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


CFG_PATH = _arg("--model-config")
if not CFG_PATH:
    raise SystemExit("--model-config is required (the relief flags come from config.json)")
RELIEF_CODE = os.environ.get("RELIEF_CODE", os.path.join(os.path.dirname(CFG_PATH), "code", "ablation"))
for f in ("film_unet.py", "implicit_solve.py", "joint_shape_albedo.py", "eval_cells.py"):
    if not os.path.exists(os.path.join(RELIEF_CODE, f)):
        raise SystemExit(f"missing {f} in RELIEF_CODE={RELIEF_CODE}")

# ---- the production adapter, imported (its bootstrap puts src/testing and the rest of src on sys.path) ----
_spec = importlib.util.spec_from_file_location("predict_single_ctx_mosaic_frame", ADAPTER)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
M = A.M

# ---- 1. relief model builder in place of build_model_cfg.build_from_config ----
sys.path.insert(0, f"{B}/inference")
import build_model_cfg as BMC  # noqa: E402

_CTOR = None
# verbatim from the relief eval_cells.py (src_v4rel, md5 f8c3d58b), lines 78-86: the constructor keys load_model restores
# from config.json. Copied rather than imported because eval_cells pulls in the loader and shot_supervision at import.
_CTOR_KEYS = {
    'skip_blur_sigma': 6.0,   'sigma_lo':   12.0,  'sigma_hi':      35.0,
    'conf_gamma':      1.0,   'cg_iters':   25,    'cg_mu':         0.05,
    'freeze_mu':       False, 'grad_lambda': 0.1,  'grad_cg_iters': 50,
    'trust_hidden':    32,    'ell_init':   None,
    'upsample_mode':   'deconv',
    'down_mode':       None,      # v3: passed when config.json sets it (maxpool | blurpool); bilinear_conv / spline_conv share
                                  # parameter shapes, so the decoder comes from config.json, never from the state dict
    'film_condition_dim': 3,
    'relief_film': False, 'relief_solve': False,
}


def _check_resampling(jsa, cfg, in_ch, depths, dropout, use_attention, ctor, model):
    """v3 guard: the decoder/encoder resampling settings of config.json must reach the network."""
    import torch
    kw = dict(in_channels=in_ch, out_channels=1, depths=depths if depths is not None else cfg.get("depths", [64, 128, 256, 512]),
              dropout=float(dropout if dropout is not None else cfg.get("dropout", 0.1)),
              use_attention=bool(use_attention if use_attention is not None else cfg.get("use_attention", True)),
              skip_mode=cfg.get("skip_mode", "dense"))
    up = ctor.get("upsample_mode", "deconv")
    n_tconv = sum(isinstance(m, torch.nn.ConvTranspose2d) for m in model.modules())
    if up != "deconv" and n_tconv:
        raise RuntimeError(f"upsample_mode={up} but the network has {n_tconv} ConvTranspose2d layers")
    if up == "deconv" and not n_tconv:
        raise RuntimeError("upsample_mode=deconv but the network has no ConvTranspose2d layer")
    for key, default in (("upsample_mode", "deconv"), ("down_mode", "maxpool")):   # the code's defaults (exporter, 10-24)
        if ctor.get(key, default) in (default, None):
            continue
        ref = jsa.FiLMUNetDualHead(**kw, **{k: v for k, v in ctor.items() if k != key})
        if str(ref) == str(model):
            raise RuntimeError(f"{key}={ctor[key]} did not change the network: the constructor ignored it")
    print(f"[arch] v3 resampling upsample_mode={up} down_mode={ctor.get('down_mode')} ConvTranspose2d={n_tconv} (checked)", flush=True)


def build_relief(config_path, in_channels=None, depths=None, use_attention=None, dropout=None, verbose=True):
    global _CTOR
    cfg = json.load(open(config_path))
    if not (cfg.get("relief_film") or cfg.get("relief_solve")):
        raise RuntimeError(f"{config_path} is not a relief model (relief_film/relief_solve both off)")
    # register the relief modules under their bare names BEFORE anything imports them (src/training holds namesakes)
    BMC._load("implicit_solve", os.path.join(RELIEF_CODE, "implicit_solve.py"))
    fu = BMC._load("film_unet", os.path.join(RELIEF_CODE, "film_unet.py"))
    jsa = BMC._load("joint_shape_albedo", os.path.join(RELIEF_CODE, "joint_shape_albedo.py"))
    for mod in (fu, jsa):
        if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.abspath(RELIEF_CODE):
            raise RuntimeError(f"{mod.__name__} resolved to {mod.__file__}, expected {RELIEF_CODE}")
    ctor = {k: cfg.get(k, d) for k, d in _CTOR_KEYS.items() if cfg.get(k, d) is not None}
    _CTOR = ctor
    in_ch = int(in_channels if in_channels is not None else cfg["in_channels"])
    model = jsa.FiLMUNetDualHead(
        in_channels=in_ch, out_channels=1,
        depths=depths if depths is not None else cfg.get("depths", [64, 128, 256, 512]),
        dropout=float(dropout if dropout is not None else cfg.get("dropout", 0.1)),
        use_attention=bool(use_attention if use_attention is not None else cfg.get("use_attention", True)),
        skip_mode=cfg.get("skip_mode", "dense"), **ctor)
    if not (getattr(model, "relief_film", False) and getattr(model, "relief_solve", False)):
        raise RuntimeError("model built without relief_film/relief_solve")
    _check_resampling(jsa, cfg, in_ch, depths, dropout, use_attention, ctor, model)
    if verbose:
        print(f"[arch] RELIEF {cfg.get('ablation_name', '?')} skip_mode={cfg.get('skip_mode')} in_ch={in_ch} "
              f"relief_film={ctor.get('relief_film')} relief_solve={ctor.get('relief_solve')} "
              f"film_condition_dim={ctor.get('film_condition_dim')} cg_iters={ctor.get('cg_iters')} "
              f"(code {RELIEF_CODE})", flush=True)
    return model, cfg


BMC.build_from_config = build_relief

# ---- 2. predict_patch with the relief scalar appended ----
_predict_patch = M.predict_patch
RELIEF_LOG_CENTER, RELIEF_LOG_SCALE = 2.0, 1.5
_RHO = []


def predict_patch_relief(model, ctx_p, mola_p, valid_p, sun_vec, device, conf_p=None):
    mstd = max(float(mola_p.std()), M.STD_FLOOR_M)          # the value predict_patch z-scores MOLA with
    rho = (math.log10(max(mstd, 1e-3)) - RELIEF_LOG_CENTER) / RELIEF_LOG_SCALE
    sv = np.concatenate([np.asarray(sun_vec, np.float32)[:3], np.array([rho], np.float32)]).astype(np.float32)
    pred, mmean, mstd2 = _predict_patch(model, ctx_p, mola_p, valid_p, sv, device, conf_p=conf_p)
    if abs(mstd2 - mstd) > 1e-6 * max(1.0, mstd):
        raise RuntimeError(f"predict_patch MOLA std {mstd2} != relief std {mstd}")
    _RHO.append(rho)
    return pred, mmean, mstd2


M.predict_patch = predict_patch_relief

# ---- 3. v3: PEDR query near the poles (bug found 2026-10-24 in R2 frame J06_046962_0925, 86 S) ----
# PedrShots.query_box folds lonmin/lonmax with % 360; a frame rotated near a pole gives a longitude span >= 360 deg (472 deg there),
# which collapsed the query to one sector: the confidence input was 0 everywhere and the shot anchoring lost its shots. The leveller
# already guards against this; the predictor did not. Here any box spanning >= 360 deg of longitude or reaching within 2 deg of a
# pole queries ALL longitudes of its latitude band (extra shots off the frame are dropped by both callers' grid tests).
_query_box = M.PedrShots.query_box
_FULL_LON = []


def query_box_v3(self, latmin, latmax, lonmin, lonmax):
    if (lonmax - lonmin) >= 359.999 or latmax >= 88.0 or latmin <= -88.0:
        _FULL_LON.append((latmin, latmax, lonmin, lonmax))
        print(f"[pedr-v3] full-longitude query: lat[{latmin:.3f},{latmax:.3f}] requested lon[{lonmin:.3f},{lonmax:.3f}]", flush=True)
        return _query_box(self, latmin, latmax, 0.0, 359.9999999)
    return _query_box(self, latmin, latmax, lonmin, lonmax)


M.PedrShots.query_box = query_box_v3

if __name__ == "__main__":
    if M.STD_FLOOR_M != 5.0:
        raise SystemExit(f"mosaic_single_ctx.STD_FLOOR_M = {M.STD_FLOOR_M}, the training loader floors at 5.0")
    A.main()
    if _RHO:
        r = np.array(_RHO)
        print(f"[relief] rho over {r.size} tiles: min {r.min():.3f} median {np.median(r):.3f} max {r.max():.3f} "
              f"(MOLA std {10 ** (r.min() * 1.5 + 2):.1f}-{10 ** (r.max() * 1.5 + 2):.1f} m)", flush=True)
    print("[relief-done] v3", flush=True)
