# ta_impl_spline100_s1_fp32_e098

MESA model from the lattice-free decoder study (CSAIL, 2026-10-23/24; revision_work/lattice_free_decoder_2026-10-23).
PRODUCTION CANDIDATE: from scratch with the production recipe (100 epochs, lr 1e-4 cosine to 1e-6, batch 16, fp32, 1 GPU, relief into FiLM + solve), checkpoint = lowest validation DEM MAE epoch (the rule that picked epoch 88). Decoder chosen by the from-scratch rule fixed 2026-10-24 (accuracy within 0.05 m of the deconv seeds, then lowest lattice share); exported 2026-10-25 ahead of FINAL_SUMMARY.md because the rule's inputs (test evals + lattice_eval) were final. (seed 1)

- Source checkpoint: `/data/scratch-fast/whamitch/lattice_dec_2026-10-23/runs/spline100_s1/abl-ta_impl_spline_20261025_100037_s1_lr0.0001_bs16_g0.5_p0.3_tv0.01_d64x128x256x512_selmae/checkpoints/best.pt` (epoch 98 of a 100-epoch run), copied to checkpoints/best.pt (full checkpoint: model +
  optimizer state; the weights are under the key 'model').
- Cell: `ta_impl_spline` = production ta_implicit_d1 (implicit solve) + relief into FiLM and the solve, with ONLY the
  resampling changed: **upsample_mode = `spline_conv`**, **down_mode = `maxpool`** (both in config.json).
  deconv = ConvTranspose2d(k=2, s=2) (epoch 88); bilinear_conv = bilinear x2 + 3x3 conv; spline_conv = fixed
  degree-4 B-spline x2 + 3x3 conv; resize_conv = nearest x2 + 3x3 conv; blurpool = anti-aliased max-pool.
- Warm start (init_from): null
- Load: with code/ablation on sys.path, `eval_cells.load_model('<this dir>', device)` builds FiLMUNetDualHead from
  config.json (its _CTOR_KEYS include upsample_mode and down_mode) and loads strictly. strict load OK: FiLMUNetDualHead, upsample_mode=spline_conv, down_mode=maxpool, 23958187 params
  bilinear_conv and spline_conv have IDENTICAL parameter shapes: always build from config.json, never infer the
  decoder from the state dict (a spline model loaded as bilinear would run without error and be wrong).
- Inference inputs are exactly those of epoch 88 (CTX, standardised MOLA, d_shot = exp(-d1/2000 m), sun vector,
  relief rho; 384 px windows, any multiple of 16), output z denormalised as before. Only film_unet.py (new
  upsampler/downsampler classes), ablation_config.py (cells 44-46, down_mode), train_mars_ctx.py (--init-from) and
  eval_cells.py (down_mode ctor key) differ from the epoch-88 code (src_v4abl2); see MD5SUMS.
