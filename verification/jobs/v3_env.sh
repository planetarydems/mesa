# Paths shared by the v3 verification jobs (SuperCloud). Everything outside $R is read only.
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mesa_v3; O=$R/out_v3; WT=$R/out/weights
M=$G/models/v3_models/ta_impl_spline100_s1_fp32_e098; ORIG=$M/code/ablation   # the model of the paper and its training code
V=$G/data/mars_ctx_dem_v4_pedr
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff; PEDR=$G/Stored_Data_for_Codes/MOLA_SQL_Table/mola
TAG=ta_implicit_d1_spline100s1_e098v3; B=$G/global_run/boxes                  # the v3 product frames
X=$G/experiments/v3_manuscript_results_2026-11-05                            # the manuscript's v3 evaluation
T11=$G/experiments/revision_2026-10-14/task11_restore359; WF=$G/experiments/implrel_e088_paper/fullframe
