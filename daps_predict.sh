#!/bin/bash
#SBATCH -J dapspredict   # Job name
#SBATCH --time=00-12:00:00
#SBATCH -p batch
# --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.predict.%A_%a.out
#SBATCH --error=logs/%j.predict.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu


# Load required modules

#exclusive
# --constraint="a100-80G"

#module purge  # Clears any previously loaded modules
module load cuda/11.0
module load anaconda/2024.10
module load ngc/1.0


module load pytorch/2.5.1-cuda12.1-cudnn9 
#best_model.pth: att26_0.2_0.1_0.1_0.0_64_2.0_1.0.pth
#conda activate biosolu
#conda install -c conda-forge lightning

#pip install -e .
export PYTHONPATH=$(pwd)/src

/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python -m milasol.models.predict_new \
  --modelname checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth \
  --out_dir outputs/ \
  --cache_dir /cluster/tufts/cowenlab/wlou01/modelcache/ \
  --label_file /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/test_tgt.txt \
  --sequence_file /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/test_src.txt \
  --esm_file /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/test_src_esm_embeddings.csv \
  --prot_file /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/test_src_prot_embeddings.csv \
  --ray_file /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/test_src_raygun_embeddings_v2.csv \
