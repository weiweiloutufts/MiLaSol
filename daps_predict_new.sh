#!/bin/bash
#SBATCH -J dapspredict   # Job name
#SBATCH --time=00-12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=logs/%j.predict.%A_%a.out
#SBATCH --error=logs/%j.predict.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
#SBATCH --array=1-10

# Load required modules



module purge  
module load cuda/11.0
module load anaconda/2024.10
module load ngc/1.0


#module load pytorch/2.5.1-cuda12.1-cudnn9 





export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

JOB_ID=${SLURM_ARRAY_TASK_ID}

# if your parts are named seq_list_part1.txt ... seq_list_part10.txt
source_dir="/cluster/home/wlou01/prot-solubility/outputs/soludb_fixed/part_${JOB_ID}/"
output_dir="/cluster/home/wlou01/prot-solubility/outputs/soludb/part_${JOB_ID}"
mkdir -p "$output_dir"

# show what this task is doing in the logs
echo "[task ${JOB_ID}] source_data=${source_data}"
echo "[task ${JOB_ID}] output_dir=${output_dir}"

# make python print immediately
export PYTHONUNBUFFERED=1

/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python -u -m milasol.models.predict_new \
  --modelname checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth \
  --out_dir "$output_dir" \
  --cache_dir /cluster/tufts/cowenlab/wlou01/modelcache/ \
  --batch_size 8 \
  --sequence_file "${source_dir}seqs.txt" \
  --esm_file "${source_dir}esm2.csv" \
  --prot_file "${source_dir}prott5.csv" \
  --ray_file "${source_dir}raygun.csv"

  

