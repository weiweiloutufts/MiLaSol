#!/bin/bash
#SBATCH -J test
#SBATCH --time=00-1:00:00
#SBATCH -p batch
#SBATCH --mem=16g
#SBATCH --output=result/MyJob.%j.out
#SBATCH --error=result/MyJob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu

# Load required modules
module purge 
module load anaconda/2024.10
export PYTHONNOUSERSITE=1

#pip install -e .
# Run the Python script with current hyperparameter combination
python src/milasol/scripts/getgold.py /cluster/home/wlou01/prot-solubility/logs/daps1 /cluster/home/wlou01/prot-solubility/result/daps1.csv

