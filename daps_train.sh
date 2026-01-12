#!/bin/bash
#SBATCH -J daps1
#SBATCH --time=00-12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
# --constraint="a100-40G"
#SBATCH --mem=64G
#SBATCH --output=logs/daps1/%j.train.%A_%a.out
#SBATCH --error=logs/daps1/%j.train.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
#SBATCH --array=0-26

# Load required modules

#exclusive
# --constraint="a100-80G"

#module purge  # Clears any previously loaded modules
module load cuda/11.0
module load anaconda/2024.10
module load ngc/1.0


module load pytorch/2.5.1-cuda12.1-cudnn9 




# Read parameters from file
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params34.txt)
set -- $PARAMS
# Extract parameters
# batch_size kernel_size num_filters lstm_hidden_dim num_lstm_layers contrastive_weight learning_rate
batch_size=$1
kernel_size=$2
num_filters=$3
lstm_hidden_dim=$4
num_lstm_layers=$5
learning_rate=$6
latent_dim=$7
contrastive_weight=$8
rec_loss_weight=$9
entropy_weight=${10}
triplet_weight=${11}
proto_weight=${12}
pos_rate=${13}



echo "Running with batch_size=$batch_size,kernel_size=$kernel_size,filters=$num_filters, lstm_hidden_dim=$lstm_hidden_dim, lstm_layers=$num_lstm_layers,learning_rate=$learning_rate, latent_dim=$latent_dim,contrastive_weight=$contrastive_weight, rec_loss_weight=$rec_loss_weight, entropy_weight=$entropy_weight,  
triplet_weight=$triplet_weight, proto_weight=$proto_weight, pos_rate=$pos_rate"

pip install -e .
python -m milasol.models.train \
  --batch_size $batch_size \
  --kernel_size $kernel_size \
  --num_filters $num_filters \
  --lstm_hidden_dim $lstm_hidden_dim \
  --num_lstm_layers $num_lstm_layers \
  --learning_rate $learning_rate \
  --latent_dim $latent_dim \
  --contrastive_weight $contrastive_weight \
  --rec_loss_weight $rec_loss_weight \
  --entropy_weight $entropy_weight \
  --triplet_weight $triplet_weight \
  --proto_weight $proto_weight \
  --pos_rate $pos_rate \
 
  