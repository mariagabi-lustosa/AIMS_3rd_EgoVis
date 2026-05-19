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
#SBATCH --cpus-per-task=32

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate kitchens_tmp  # Replace with your environment name


python ~/github/AIMS_3rd_EgoVis/process_optical.py \
    --rgb_dir    /hadatasets/EPIC-KITCHENS_rgb_crops \
    --video_path /hadatasets/EPIC-KITCHENS_384