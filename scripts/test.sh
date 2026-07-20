#!/bin/bash
#SBATCH -J test  # Job name
#SBATCH --time=00-12:00:00
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
# --constraint=a100
#SBATCH --mem=64G
#SBATCH --output=%j.test.%A_%a.out
#SBATCH --error=%j.test%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu


# # Load required modules


set -euo pipefail

# Load required modules
module purge
module load modtree/deprecated
module load anaconda/2024.10
conda activate biosolu

export PYTHONUNBUFFERED=1



PY=/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python

#echo "CVD=${CUDA_VISIBLE_DEVICES:-<empty>}"; nvidia-smi | head -n 3
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$PY -c "import esm; from raygun.pretrained import raygun_4_4mil_800M; print('deps OK')"




export PYTHONPATH=/cluster/home/wlou01/milasol/src
$PY -u test.py 



