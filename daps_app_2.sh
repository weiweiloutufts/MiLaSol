#!/bin/bash
#SBATCH -J app  # Job name
#SBATCH --time=00-48:00:00
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
# --constraint=a100
#SBATCH --mem=64G
#SBATCH --output=logs/sa/%j.app.%A_%a.out
#SBATCH --error=logs/sa/%j.app.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
#SBATCH --array=0-8


# Load required modules
set -euo pipefail

# Load required modules
module purge
module load modtree/deprecated
module load anaconda/2024.10
conda activate biosolu

export PYTHONUNBUFFERED=1

GPU_WAIT_SECONDS=30
PY=/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python

echo "CVD=${CUDA_VISIBLE_DEVICES:-<empty>}"
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$PY -c "import esm; from raygun.pretrained import raygun_4_4mil_800M; print('deps OK')"



DATA_FILE="/cluster/home/wlou01/milasol/data/soludb_wild_insol_seq.txt"
LINES_PER_JOB=275
TOTAL_LINES=$(wc -l < "$DATA_FILE")
JOB_ID=${SLURM_ARRAY_TASK_ID}
START_LINE=$(( JOB_ID * LINES_PER_JOB + 1 ))

if [ "$START_LINE" -gt "$TOTAL_LINES" ]; then
    echo "Job $JOB_ID has no sequences to process. Exiting."
    exit 0
fi

if [ "$JOB_ID" -lt 8 ]; then
    END_LINE=$(( START_LINE + LINES_PER_JOB - 1 ))
    if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
        END_LINE=$TOTAL_LINES
    fi
else
    END_LINE=$TOTAL_LINES
fi

TMP_FILE=$(mktemp "${TMPDIR:-/tmp}/daps_sa_${SLURM_JOB_ID}_${JOB_ID}_XXXX.txt")
sed -n "${START_LINE},${END_LINE}p" "$DATA_FILE" > "$TMP_FILE"

if [ ! -s "$TMP_FILE" ]; then
    echo "No sequences extracted for job $JOB_ID (start=$START_LINE, end=$END_LINE). Exiting."
    rm -f "$TMP_FILE"
    exit 0
fi

PART_ID=$(printf "%02d" ${JOB_ID})

export PYTHONPATH=$(pwd)/src
$PY -u src/milasol/app/app2.py \
    --data-path "$TMP_FILE" \
    --model-path checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth \
    --out-dir  design \
    --fileid "$PART_ID" \
    --n-steps 5 \
    --n-restarts 100 \
    --chunk-size 32

rm -f "$TMP_FILE"

