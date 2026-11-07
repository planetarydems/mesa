# Provenance of the v3 chain (release/mader/v3)
Every file under `v3/` is a verbatim copy of the installed file on SuperCloud (project directory
`/home/gridsan/groups/planetary_dtm/mars_dem`, written `$G` below), taken on 2026-11-06 (EST) after the
seam campaign had closed the per-box chain. The md5 is the digest of the copy. The digest of the source
was checked on the cluster with `md5sum -c` before the commit (`v3/SOURCES.tsv`, the same table in
machine-readable form). Nothing under `v3/` was edited: the stage scripts keep their cluster paths,
their headers and their change notes.
The weight tables (586 CSV files) are listed in the table once. Their individual digests are in
`v3/downstream/weights/MD5SUMS`.
The model weights are not in this repository (287.6 MB). The checkpoint is
`$G/models/v3_models/ta_impl_spline100_s1_fp32_e098/checkpoints/best.pt`, md5
`96de121dea9e271212a93df39d6b7f45` (`v3/model/.../MD5SUMS`, also recorded in `v3/predict/CAMPAIGN.json`
and in every frame's `.done` record).

## Prediction (stage 1)
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/predict/CAMPAIGN.json` | `$G/global_run/v3_2026-10/predict/CAMPAIGN.json` | `8d2097c10ae53c638b8a3dce69fafc68` |
| `v3/predict/predict_frame_v3.py` | `$G/global_run/v3_2026-10/code/predict_frame_v3.py` | `157e5f0b7d3d0dcd9983168155016bb9` |
| `v3/predict/predict_queue_v3.py` | `$G/global_run/v3_2026-10/code/predict_queue_v3.py` | `eca4fde0cd7da308f10e419706d1443a` |
| `v3/predict/predict_queue_v3.sbatch` | `$G/global_run/v3_2026-10/code/predict_queue_v3.sbatch` | `9a91d97ea5289c90f03a4087ed48c9ae` |

## Model package (metadata only)
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/model/ta_impl_spline100_s1_fp32_e098/MD5SUMS` | `$G/models/v3_models/ta_impl_spline100_s1_fp32_e098/MD5SUMS` | `5d41cc9d226c1fca50c1816adf23de48` |
| `v3/model/ta_impl_spline100_s1_fp32_e098/README.md` | `$G/models/v3_models/ta_impl_spline100_s1_fp32_e098/README.md` | `f67714e16294f32cc8ca615360faadde` |
| `v3/model/ta_impl_spline100_s1_fp32_e098/config.json` | `$G/models/v3_models/ta_impl_spline100_s1_fp32_e098/config.json` | `73bc7aaa91e3563967d7de5538b6c7ec` |
| `v3/model/ta_impl_spline100_s1_fp32_e098/summary.json` | `$G/models/v3_models/ta_impl_spline100_s1_fp32_e098/summary.json` | `b263dbecd05d4eb915f641bc5930e09a` |

## Per-box chain: code
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/downstream/code/eval_box.py` | `$G/global_run/v3_2026-10/downstream/code/eval_box.py` | `9288356207db1ce481e1542ce8c210af` |
| `v3/downstream/code/frame_outlier_v3.py` | `$G/global_run/v3_2026-10/downstream/code/frame_outlier_v3.py` | `f41748312831f1092d933117128a08e1` |
| `v3/downstream/code/fresh_tag_v3.py` | `$G/global_run/v3_2026-10/downstream/code/fresh_tag_v3.py` | `9bba1d1254650ce89e1734b6518655d5` |
| `v3/downstream/code/join_check.py` | `$G/global_run/v3_2026-10/downstream/code/join_check.py` | `60a9fd661262525e59ae71d182d76e65` |
| `v3/downstream/code/lattice_box.py` | `$G/global_run/v3_2026-10/downstream/code/lattice_box.py` | `1f61142579f65ee93ee128c71b3d82f7` |
| `v3/downstream/code/level_frames_box_v3.py` | `$G/global_run/v3_2026-10/downstream/code/level_frames_box_v3.py` | `e1a5d191ee4bf131f2f5a85a78a3f3da` |
| `v3/downstream/code/make_inputs_v3.py` | `$G/global_run/v3_2026-10/downstream/code/make_inputs_v3.py` | `cfca93b5244c291fde15fe20d5ff9283` |
| `v3/downstream/code/make_pass2.py` | `$G/global_run/v3_2026-10/downstream/code/make_pass2.py` | `ea930d0185296bf3b92693dcf4070432` |
| `v3/downstream/code/make_test_root.py` | `$G/global_run/v3_2026-10/downstream/code/make_test_root.py` | `fee2e485304fee614af1fac12bc7c334` |
| `v3/downstream/code/mola_infill_fix.py` | `$G/global_run/v3_2026-10/downstream/code/mola_infill_fix.py` | `012ef2804a8fe80f957cc4972ea1b399` |
| `v3/downstream/code/release2_context_signature_v3.py` | `$G/global_run/v3_2026-10/downstream/code/release2_context_signature_v3.py` | `56addd9a3738803bc1b767cd3518f648` |
| `v3/downstream/code/seam_pass2.py` | `$G/global_run/v3_2026-10/downstream/code/seam_pass2.py` | `5a477b15c15631b2d87eb78c9153b220` |
| `v3/downstream/code/seam_relax_v3.py` | `$G/global_run/v3_2026-10/downstream/code/seam_relax_v3.py` | `8b8f9501d75a9102c6229cb017a215d7` |
| `v3/downstream/code/seam_stage_v3.py` | `$G/global_run/v3_2026-10/downstream/code/seam_stage_v3.py` | `cf033b0886731d86b3bee9ab9f6537f2` |
| `v3/downstream/code/seamline_merge_v3.py` | `$G/global_run/v3_2026-10/downstream/code/seamline_merge_v3.py` | `4bc43e4968cb07d470ca0c24e65b00c8` |
| `v3/downstream/code/v3_input_guard.py` | `$G/global_run/v3_2026-10/downstream/code/v3_input_guard.py` | `45c9ebabc89e5cc986a51d86560e81b7` |
| `v3/downstream/code/worker_v3d.py` | `$G/global_run/v3_2026-10/downstream/code/worker_v3d.py` | `35ce8949c0e04b2e168c73d9b61691f5` |

## Per-box chain: stage scripts
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/downstream/stages/build_box_final_tag_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/build_box_final_tag_v3.sh` | `a38e473d3142f0183d9d60f97c4900f2` |
| `v3/downstream/stages/ctx_fill_box_tag_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/ctx_fill_box_tag_v3.sh` | `d09db02ab4108474ce8fa33fc448d886` |
| `v3/downstream/stages/ctx_fill_cpu_tag_v3.sbatch` | `$G/global_run/v3_2026-10/downstream/stages/ctx_fill_cpu_tag_v3.sbatch` | `c7a718763860cb72053be44c0b1fdb45` |
| `v3/downstream/stages/ctx_fill_gpu_tag_v3.sbatch` | `$G/global_run/v3_2026-10/downstream/stages/ctx_fill_gpu_tag_v3.sbatch` | `e0290f58d2fed6d339fd1a0c43877003` |
| `v3/downstream/stages/level_box_tag_v3.sbatch` | `$G/global_run/v3_2026-10/downstream/stages/level_box_tag_v3.sbatch` | `6e7cdde3240a62e5edc98fc32003d47f` |
| `v3/downstream/stages/mif_box_tag_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/mif_box_tag_v3.sh` | `29cb731140e5bb146f362e8c38dc67a6` |
| `v3/downstream/stages/pass2_box_tag_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/pass2_box_tag_v3.sh` | `682dfc4726f6c03d4f44d04c3354ee83` |
| `v3/downstream/stages/qa_box_tag_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/qa_box_tag_v3.sh` | `de7901a4972caf97ff16484f051d71cb` |
| `v3/downstream/stages/rejoin_box_v3.sh` | `$G/global_run/v3_2026-10/downstream/stages/rejoin_box_v3.sh` | `a67b43012a74d7a31280fe6f97feecaf` |

## Per-box chain: helpers sourced from the release-2 chain
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/downstream/helpers/fresh_tag.py` | `$G/global_run/predict_v4s3d_e098/downstream/fresh_tag.py` | `12822e403a75a08e971665473fdf665e` |
| `v3/downstream/helpers/tag_guard.sh` | `$G/global_run/predict_v4s3d_e098/downstream/tag_guard.sh` | `be5cf990a13000e1f8635150695091d5` |

## Per-box chain: fleet worker and sweeps
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/downstream/fleet/boxes.tsv` | `$G/global_run/v3_2026-10/downstream/boxes.tsv` | `8ee209c01f1370f158cb7969ccea10cd` |
| `v3/downstream/fleet/pass2_sweep.sbatch` | `$G/global_run/v3_2026-10/downstream/pass2_sweep.sbatch` | `fab7e528ef7833ee795fc2566c70b001` |
| `v3/downstream/fleet/redo_box_v3.sh` | `$G/global_run/v3_2026-10/downstream/redo_box_v3.sh` | `a870ff287fe3af4479ac230f3df59240` |
| `v3/downstream/fleet/redo_watch.sbatch` | `$G/global_run/v3_2026-10/downstream/redo_watch.sbatch` | `0481c5d8f3e5488539b950a6f598d785` |
| `v3/downstream/fleet/rejoin_worker.sbatch` | `$G/global_run/v3_2026-10/downstream/rejoin_worker.sbatch` | `1a265d18c0f092dbdd6546c3066875cf` |
| `v3/downstream/fleet/remif_sweep.sbatch` | `$G/global_run/v3_2026-10/downstream/remif_sweep.sbatch` | `8d1a79cee97924ed588ee8199fe5d67f` |
| `v3/downstream/fleet/reqa_sweep.sbatch` | `$G/global_run/v3_2026-10/downstream/reqa_sweep.sbatch` | `c1445ac679fdd185ee8c6290470a570b` |
| `v3/downstream/fleet/reqa_sweep_20261028.sbatch` | `$G/global_run/v3_2026-10/downstream/reqa_sweep_20261028.sbatch` | `fef429bcab962d913c1e09ae260975b5` |
| `v3/downstream/fleet/reverdict_20261028.py` | `$G/global_run/v3_2026-10/downstream/reverdict_20261028.py` | `e3d910670a5ca5c88fbf8e7a76ecf396` |
| `v3/downstream/fleet/seam_sweep.sbatch` | `$G/global_run/v3_2026-10/downstream/seam_sweep.sbatch` | `74cf8b4ca51087d892990cc38fd0d7cd` |
| `v3/downstream/fleet/worker_v3d_p8.sbatch` | `$G/global_run/v3_2026-10/downstream/worker_v3d_p8.sbatch` | `c90ac11dcff37d3967bdfcb8cbfbc2c4` |
| `v3/downstream/fleet/worker_v3d_volta.sbatch` | `$G/global_run/v3_2026-10/downstream/worker_v3d_volta.sbatch` | `84203431f6a91146bf81eca40ce1a688` |

## Per-box frame weights (586 boxes)
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/downstream/weights/<BOX>.csv` (586 files) | `/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/weights/<BOX>.csv` | see `v3/downstream/weights/MD5SUMS` (written in this repository) |

## Polar MOLA-comb correction (pcf80): code
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/polar_combfix80/code/FROZEN.md5` | `$G/global_run/v3_2026-10/polar_combfix80/code/FROZEN.md5` | `b7683d07d69a8d6dccd5a91eec40978b` |
| `v3/polar_combfix80/code/bigC_diag.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/bigC_diag.py` | `2e74012d90ad94260944694dc8fc2061` |
| `v3/polar_combfix80/code/box_correct.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/box_correct.py` | `4a0fc574a98cb8fadaa93025e1305b4e` |
| `v3/polar_combfix80/code/combfix.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/combfix.py` | `265992c8fcec8872ef690286c78f5b47` |
| `v3/polar_combfix80/code/install_pcf80.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/install_pcf80.py` | `b396e68f68578e8847c9c297e945f423` |
| `v3/polar_combfix80/code/limit_test.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/limit_test.py` | `623fbc472ab5b5601fe5ee0fde844bab` |
| `v3/polar_combfix80/code/limit_test.sbatch` | `$G/global_run/v3_2026-10/polar_combfix80/code/limit_test.sbatch` | `7fe43af298da04eb4950a56670e9ed76` |
| `v3/polar_combfix80/code/notch_eval.sbatch` | `$G/global_run/v3_2026-10/polar_combfix80/code/notch_eval.sbatch` | `41221f7ec76a482fbb39533b0f641753` |
| `v3/polar_combfix80/code/params.json` | `$G/global_run/v3_2026-10/polar_combfix80/code/params.json` | `6701b93f861b7a53b060181ca9102ad1` |
| `v3/polar_combfix80/code/pcf80_boxes.sbatch` | `$G/global_run/v3_2026-10/polar_combfix80/code/pcf80_boxes.sbatch` | `93ca49f5b04f5658eb1616fbce7f29cd` |
| `v3/polar_combfix80/code/pcf80_state.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/pcf80_state.py` | `f8321105b7005e9009ae00553dcd91c3` |
| `v3/polar_combfix80/code/pcf80_sweep.sbatch` | `$G/global_run/v3_2026-10/polar_combfix80/code/pcf80_sweep.sbatch` | `bba4c1966d0dd69c096ac3b288d95d7c` |
| `v3/polar_combfix80/code/pick_part.sh` | `$G/global_run/v3_2026-10/polar_combfix80/code/pick_part.sh` | `83a66f400506d425ffb84bad6263d2ef` |
| `v3/polar_combfix80/code/qa_box_tag_v3_des.sh` | `$G/global_run/v3_2026-10/polar_combfix80/code/qa_box_tag_v3_des.sh` | `f1531ba663ac7f2103a22162aec04780` |
| `v3/polar_combfix80/code/qa_pcf80.sbatch` | `$G/global_run/v3_2026-10/polar_combfix80/code/qa_pcf80.sbatch` | `c5e40dfbd20485d9f869a9b72b13a068` |
| `v3/polar_combfix80/code/summ_notch_pmc.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/summ_notch_pmc.py` | `bab7027b87c58d41b2248b89da3d5bcd` |
| `v3/polar_combfix80/code/verify_pcf80.py` | `$G/global_run/v3_2026-10/polar_combfix80/code/verify_pcf80.py` | `d3ba0755c31ca8da097c607aea1991ad` |

## Polar MOLA-comb correction (pcf80): records
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/polar_combfix80/INSTALL_LOG.txt` | `$G/global_run/v3_2026-10/polar_combfix80/INSTALL_LOG.txt` | `a27916e25716b71d813a76caeee57eae` |
| `v3/polar_combfix80/README.md` | `$G/global_run/v3_2026-10/polar_combfix80/README.md` | `612dd7e092c0d2f11b0f77a472cbf520` |

## Reviewed local corrections
| release path | source on SuperCloud | md5 |
|---|---|---|
| `v3/corrections/LATEST.json` | `$G/global_run/v3_2026-10/site_corrections_global_2026-11-03/LATEST.json` | `24e22a3d4f338b4e1185690331bad755` |
| `v3/corrections/apply_global.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/apply_global.py` | `e02835622edd8504bc1c858940a4dc5e` |
| `v3/corrections/apply_reviewed.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/apply_reviewed.py` | `7621486b6c8dee6f42623b815e7d8fe9` |
| `v3/corrections/campaign_decisions.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/campaign_decisions.py` | `ca6686a2b49d7e919d3a0177a190a4bf` |
| `v3/corrections/edge_certificate.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/edge_certificate.py` | `a22bb1666e836399b67e6b72638a2057` |
| `v3/corrections/release_lock.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/release_lock.py` | `d951bfe3bf32e7c4cfc036857430879b` |
| `v3/corrections/resolve_global.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/resolve_global.py` | `8f1196737b4136b2fe9ef324ccaf1eab` |
| `v3/corrections/test_global_apply.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/test_global_apply.py` | `2bf691a47ab881896fee09af311272ed` |
| `v3/corrections/test_global_resolver.py` | `$G/global_run/fix_2026-10/r2_delivery/failure_fixes/seam_remaining/pilot/global_20261103/code/test_global_resolver.py` | `d8aba6635d0a8c8576dc177f9bcaabae` |
