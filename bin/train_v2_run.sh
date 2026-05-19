#!/bin/bash
#SBATCH --job-name=vjepa_fusion
#SBATCH --output=/home/lucas.ueda/slurm/vjepa_fusion_%j.out
#SBATCH --error=/home/lucas.ueda/slurm/vjepa_fusion_%j.err
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mail-user=l156368@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate kitchens

# Execute the Fusion Training Script
# Note: --encoder_lr 0.0 ensures the encoder remains frozen.
# Note: Batch size is reduced to 16 (vs 32) because we are processing 3x the video data.
python ~/github/AIMS_3rd_EgoVis/train_v2.py \
    --train_csv /hadatasets/EPIC-KITCHENS/EPIC_100_train.csv \
    --val_csv   /hadatasets/EPIC-KITCHENS/EPIC_100_validation.csv \
    --rgb_dir   /hadatasets/EPIC-KITCHENS_rgb_crops \
    --flow_u_dir /hadatasets/EPIC-KITCHENS_rgb_crops_flow_u \
    --flow_v_dir /hadatasets/EPIC-KITCHENS_rgb_crops_flow_v \
    --model_path ./experiments/vjepa2_flow_fusion \
    --batch_size 4 \
    --accumulation_steps 8 \
    --epochs 30 \
    --lr 1e-4 \
    --encoder_lr 0.0 \
    --weighted_loss