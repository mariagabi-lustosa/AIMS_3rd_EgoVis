# Project Overview

This repository contains experiments and training pipelines for video representation learning using JEPA and precomputed motion features.

## Repository Structure

### `src/preprocessing/`
Scripts used to pre-extract:
- **RAFT-based optical flow**
- **JEPA embeddings**

These preprocessing steps are used to reduce computational cost during training by avoiding repeated feature extraction.

---

### `bin/`
Contains:
- SLURM execution scripts
- Training launch scripts
- Experimental training pipelines

#### Important Files
- `train_v1.py` — Initial JEPA training implementation
- `train_v2.py` — Improved/alternative JEPA training attempt

> Both training versions are still experimental and may require additional verification and cleanup.

---

### Environment Setup

#### `make_env.sh`
Creates the Conda environment required for the project.

#### `requirements_tmp.txt`
Contains the current Python package dependencies.

---
