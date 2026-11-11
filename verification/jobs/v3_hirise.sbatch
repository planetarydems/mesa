#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 06:00:00
#SBATCH -J mesa_v3_hirise
#SBATCH -o %x_%j.out
# Tables 3 and 4 and the HiRISE numbers of Section 3.2 for the model of the paper: the input tables rebuilt from the
# manuscript's v3 scoring (the windows of release 1 and their co-registration outcome, the v3 product frames, the full
# stereo DEMs), every window scored by the released code and compared with the manuscript's v3 records, per-DEM
# values and aggregate. Python environment of the manuscript's HiRISE scoring (GDAL 3.6.2).
source /etc/profile 2>/dev/null
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
N=$G/experiments/hirise_newmodel_test_2026-10-24; FIX=$X/hirise/fixed_r1
PY=$HOME/.conda/envs/planetary_dtm/bin/python
OVR='{"B20_017508_1878+D20_035006_1877": "../stereo_dems_rerun_2026-10-15/B20_017508_1878_XI_07N194W_D20_035006_1877_XN_07N195W", "B08_012873_2045+P08_004052_2045": "../stereo_dems_rerun_2026-10-15/P08_004052_2045_XN_24N019W_B08_012873_2045_XN_24N019W"}'
D=$O/hirise; mkdir -p $D/data $D/inputs; cd $C
echo "=== $(hostname) $(date)"
$PY -c "import rasterio, scipy, numpy, torch; print('rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"
[ -s $D/inputs/hirise_windows.csv ] || $PY verification/make_hirise_inputs_v3.py $FIX $N/defs/sites_ff_r2.csv $N/defs/frames_r2.csv \
  $N/defs/r2_frames.tsv $R/ref/hirise_placement/ori_pick.csv $R/ref/hirise_sample/sel_r1r2.csv \
  $R/ref/hirise_placement/modelfree_exclude.txt $G $WF/labels_full "$OVR" $TAG $D/data $D/inputs
$PY verification/check_hirise.py --ref $FIX --inputs $D/inputs --sample $R/ref/hirise_sample --group $G \
  --out $D --ref-prefix eval --jobs ${JOBS:-40} ${ONLY:+--only $ONLY}
echo "=== done $(date)"
