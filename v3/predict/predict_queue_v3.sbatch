#!/bin/bash
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -J v3_pred
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/logs/%x_%j.out
# predict_queue_v3.sbatch -- one worker of predict_queue_v3.py (shared mkdir-claim queue). Env: QUEUE, MODEL (model folder), OUT
# (out root), NPROC (8), THREADS (5), LIMIT (optional), WAIT_MODEL_MIN (minutes to wait for MODEL/MD5SUMS to appear and verify).
# Partition / GPUs at submission (xeon-p8: CPU; xeon-g6-volta: add --gres=gpu:volta:2).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
C=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/code
echo "=== $(hostname) $(date) job $SLURM_JOB_ID model $MODEL queue $QUEUE out $OUT"
for i in $(seq 0 ${WAIT_MODEL_MIN:-0}); do
  if [ -s "$MODEL/MD5SUMS" ] && (cd "$MODEL" && md5sum -c --quiet MD5SUMS); then echo "model verified $(date)"; break; fi
  [ "$i" -ge "${WAIT_MODEL_MIN:-0}" ] && { echo "FATAL: model not present/verified: $MODEL"; exit 2; }
  sleep 60
done
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
python3 -u $C/predict_queue_v3.py --queue "$QUEUE" --model-dir "$MODEL" --out-root "$OUT" --nproc ${NPROC:-8} --threads ${THREADS:-5} \
    --gpus $GPUS ${LIMIT:+--limit $LIMIT}
echo "=== done $(date): done $(ls $OUT/*/*/.done 2>/dev/null | wc -l) fail $(ls $OUT/*/*/.fail 2>/dev/null | wc -l)"
