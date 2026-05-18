#!/bin/bash
set -e

source ~/miniconda3/bin/activate

conda create -n kitchens python=3.10 -y
conda activate kitchens