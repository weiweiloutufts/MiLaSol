#!/bin/bash
#SBATCH -J app_so2  # Job name
#SBATCH --time=05-12:00:00
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --mem=64G
#SBATCH --output=logs/sa/%j.app20.out
#SBATCH --error=logs/sa/%j.app20.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu


# Load required modules


module purge
module load ngc/1.0
module load pytorch/2.5.1-cuda12.1-cudnn9



PY=/cluster/tufts/ngc/tools/pytorch/2.5.1-cuda12.1-cudnn9/bin/python
export PYTHONUNBUFFERED=1

echo "CVD=${CUDA_VISIBLE_DEVICES:-<empty>}"; nvidia-smi | head -n 3
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$PY -c "import esm; from raygun.pretrained import raygun_4_4mil_800M; print('deps OK')"



DATA_FILE="/cluster/home/wlou01/prot-solubility/data/sample_32.txt"

export PYTHONPATH=$(pwd)/src
$PY -u src/milasol/app/app2.py \
    --data-path "$DATA_FILE" \
    --model-path checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth \
    --out-dir  /cluster/tufts/cowenlab/wlou01/datasets/outputs/sample_32_10 \
    --fileid "00" \
    --n-steps 10 \
    --n-restarts 100 \
    --chunk-size 32
echo "Job finished at $(date)"

