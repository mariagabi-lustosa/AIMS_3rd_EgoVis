#!/bin/bash
#SBATCH --job-name=vjepa_ext_v
#SBATCH --output=/home/lucas.ueda/slurm/vjepa_v_%j.out
#SBATCH --error=/home/lucas.ueda/slurm/vjepa_v_%j.err
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mail-user=l156368@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate kitchens

# Extracting Flow-V Embeddings
python ~/github/AIMS_3rd_EgoVis/extract_embeddings.py \
    --input_dir /hadatasets/EPIC-KITCHENS_rgb_crops_flow_v \
    --cache_dir /home/lucas.ueda/github/AIMS_3rd_EgoVis/cache_models