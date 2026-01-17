#!/bin/bash
#SBATCH -J test  # Job name
#SBATCH --time=05-12:00:00
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --mem=64G
#SBATCH --output=%j.test.%A_%a.out
#SBATCH --error=%j.test%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu


# # Load required modules


module purge
module load ngc/1.0
module load pytorch/2.5.1-cuda12.1-cudnn9



PY=/cluster/tufts/ngc/tools/pytorch/2.5.1-cuda12.1-cudnn9/bin/python
export PYTHONUNBUFFERED=1

echo "CVD=${CUDA_VISIBLE_DEVICES:-<empty>}"; nvidia-smi | head -n 3
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$PY -c "import esm; from raygun.pretrained import raygun_4_4mil_800M; print('deps OK')"



export PYTHONPATH=$(pwd)/src
$PY -u test.py 



