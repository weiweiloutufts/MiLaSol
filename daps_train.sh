#!/bin/bash
#SBATCH -J noseq
#SBATCH --time=00-12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=pax052
# --constraint="a100-40G"
#SBATCH --mem=64G
#SBATCH --output=logs/noseq/%j.train.%A_%a.out
#SBATCH --error=logs/noseq/%j.train.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
#SBATCH --array=0-26

# Load required modules
set -euo pipefail

# Load required modules
module purge
module load modtree/deprecated
module load anaconda/2024.10
conda activate biosolu

export PYTHONUNBUFFERED=1

GPU_WAIT_SECONDS=30
py=/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python

echo "CVD=${CUDA_VISIBLE_DEVICES:-<empty>}"
nvidia-smi | head -n 3 || true
$py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$py -c "import esm; from raygun.pretrained import raygun_4_4mil_800M; print('deps OK')"


if ! $py -c "import torch; raise SystemExit(0 if torch.version.cuda is not None else 1)"; then
  echo "Loaded Python has CPU-only PyTorch. Exiting." >&2
  exit 1
fi

while ! $py -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; do
  echo "CUDA is not available yet. Waiting ${GPU_WAIT_SECONDS}s before retrying."
  nvidia-smi || true
  sleep "$GPU_WAIT_SECONDS"
done

echo "CUDA is available. Starting training."

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

DATASET_LAYOUT="deepsol"
MODEL_DIR="/cluster/tufts/cowenlab/wlou01/model_noseq"

if [ "$DATASET_LAYOUT" = "plm_sol" ]; then
  DATA_ROOT="/cluster/tufts/cowenlab/wlou01/datasets/plm_sol"
  TRAIN_SEQUENCE_FILE="$DATA_ROOT/train/train_seqs.txt"
  TRAIN_ESM_FILE="$DATA_ROOT/train/train_esm2.csv"
  TRAIN_PROT_FILE="$DATA_ROOT/train/train_prott5.csv"
  TRAIN_RAYGUN_FILE="$DATA_ROOT/train/train_raygun.csv"
  TRAIN_LABEL_FILE="$DATA_ROOT/train_tgt.txt"
  VAL_SEQUENCE_FILE="$DATA_ROOT/validation/validation_seqs.txt"
  VAL_ESM_FILE="$DATA_ROOT/validation/validation_esm2.csv"
  VAL_PROT_FILE="$DATA_ROOT/validation/validation_prott5.csv"
  VAL_RAYGUN_FILE="$DATA_ROOT/validation/validation_raygun.csv"
  VAL_LABEL_FILE="$DATA_ROOT/val_tgt.txt"
  TEST_SEQUENCE_FILE="$DATA_ROOT/test/test_seqs.txt"
  TEST_ESM_FILE="$DATA_ROOT/test/test_esm2.csv"
  TEST_PROT_FILE="$DATA_ROOT/test/test_prott5.csv"
  TEST_RAYGUN_FILE="$DATA_ROOT/test/test_raygun.csv"
  TEST_LABEL_FILE="$DATA_ROOT/test_tgt.txt"
elif [ "$DATASET_LAYOUT" = "deepsol" ]; then
  DATA_ROOT="/cluster/tufts/cowenlab/wlou01/datasets/deepsol_data"
  TRAIN_SEQUENCE_FILE="$DATA_ROOT/train_src.txt"
  TRAIN_ESM_FILE="$DATA_ROOT/train_src_esm_embeddings.csv"
  TRAIN_PROT_FILE="$DATA_ROOT/train_src_prot_embeddings.csv"
  TRAIN_RAYGUN_FILE="$DATA_ROOT/train_src_raygun_embeddings_v2.csv"
  TRAIN_LABEL_FILE="$DATA_ROOT/train_tgt.txt"
  VAL_SEQUENCE_FILE="$DATA_ROOT/val_src.txt"
  VAL_ESM_FILE="$DATA_ROOT/val_src_esm_embeddings.csv"
  VAL_PROT_FILE="$DATA_ROOT/val_src_prot_embeddings.csv"
  VAL_RAYGUN_FILE="$DATA_ROOT/val_src_raygun_embeddings_v2.csv"
  VAL_LABEL_FILE="$DATA_ROOT/val_tgt.txt"
  TEST_SEQUENCE_FILE="$DATA_ROOT/test_src.txt"
  TEST_ESM_FILE="$DATA_ROOT/test_src_esm_embeddings.csv"
  TEST_PROT_FILE="$DATA_ROOT/test_src_prot_embeddings.csv"
  TEST_RAYGUN_FILE="$DATA_ROOT/test_src_raygun_embeddings_v2.csv"
  TEST_LABEL_FILE="$DATA_ROOT/test_tgt.txt"
else
  echo "Unknown DATASET_LAYOUT: $DATASET_LAYOUT" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

export PYTHONPATH=$(pwd)/src
echo "Testing dataset visibility from training python"
$py -c "from pathlib import Path; p=Path('/cluster/tufts/cowenlab/wlou01/datasets/plm_sol/train/train_seqs.txt'); print(p.exists(), p.is_file(), p)"
$py -c "open('/cluster/tufts/cowenlab/wlou01/datasets/plm_sol/train/train_seqs.txt').readline(); print('read ok')"

$py -u -m  milasol.models.train \
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
  --model_dir "$MODEL_DIR" \
  --train_sequence_file "$TRAIN_SEQUENCE_FILE" \
  --train_esm_file "$TRAIN_ESM_FILE" \
  --train_prot_file "$TRAIN_PROT_FILE" \
  --train_raygun_file "$TRAIN_RAYGUN_FILE" \
  --train_label_file "$TRAIN_LABEL_FILE" \
  --val_sequence_file "$VAL_SEQUENCE_FILE" \
  --val_esm_file "$VAL_ESM_FILE" \
  --val_prot_file "$VAL_PROT_FILE" \
  --val_raygun_file "$VAL_RAYGUN_FILE" \
  --val_label_file "$VAL_LABEL_FILE" \
  --test_sequence_file "$TEST_SEQUENCE_FILE" \
  --test_esm_file "$TEST_ESM_FILE" \
  --test_prot_file "$TEST_PROT_FILE" \
  --test_raygun_file "$TEST_RAYGUN_FILE" \
  --test_label_file "$TEST_LABEL_FILE"

  