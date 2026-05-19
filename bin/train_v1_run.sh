#!/bin/bash
#SBATCH --job-name=jepa
#SBATCH --output=/home/lucas.ueda/slurm/jepa%j.out
#SBATCH --error=/home/lucas.ueda/slurm/jepa%j.err
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00  # Maximum 2 days as per your cluster limit
#SBATCH --mem=128G         # Increased memory for audio processing
#SBATCH --partition=l40s   # Selecting the h100 partition
#SBATCH --gres=gpu:1
#SBATCH --mail-user=l156368@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate kitchens  # Replace with your environment name


python ~/github/AIMS_3rd_EgoVis/train_v1.py \
    --train_csv /hadatasets/EPIC-KITCHENS/EPIC_100_train.csv \
    --val_csv   /hadatasets/EPIC-KITCHENS/EPIC_100_validation.csv \
    --data_dir  /hadatasets/EPIC-KITCHENS_rgb_crops \
    --model_path ./experiments/vjepa2_epic \
    --batch_size 32 --accumulation_steps 2 \
    --epochs 30 --lr 1e-3 --weighted_loss