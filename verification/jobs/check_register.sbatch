#!/bin/bash
#SBATCH -p debug-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH -t 02:00:00
#SBATCH -J mesa_register
#SBATCH -o %x_%j.out
# Registration (similarity + refinement) of the released code vs production, one equatorial and one polar tile.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 GDAL_CACHEMAX=1024
export PROJ_DATA=$HOME/miniforge/envs/isis/share/proj PROJ_LIB=$HOME/miniforge/envs/isis/share/proj
export PATH=$HOME/miniforge/envs/isis/bin:$PATH        # the environment production registration ran in
G=/home/gridsan/groups/planetary_dtm/mars_dem; R=$G/release_check
cd $R/mader
echo "=== $(hostname) $(date)"
python3 verification/check_register.py E-030_N-10 robbins $G/global_run/boxes $R/out/register 5 $G $G/global_run/robbins_quads
python3 verification/check_register.py Ep090_N-70 murray $G/global_run/boxes $R/out/register 5 $G $G/global_run/polar_prep/field/polar_corrections.csv
echo "=== done $(date)"
