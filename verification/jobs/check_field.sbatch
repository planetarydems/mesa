#!/bin/bash
#SBATCH -p debug-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -J mesa_field
#SBATCH -o %x_%j.out
# Global CTX Mosaic corrections and the PEDR field (fit + apply) of the released code vs production.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
G=/home/gridsan/groups/planetary_dtm/mars_dem/global_run
cd /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mader
python3 verification/check_field.py $G/pedr_field $G/polar_prep/field $G/boxes E-030_N-10 Ep090_N-70 Ep050_Np30
