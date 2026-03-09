# Short-Term Object Interaction Anticipation with DETR, SigLIP and Cosmos Tokenizer

This project implements a short-term anticipation (STA) system for object detection using DETR (Detection Transformer) with SigLIP backbone and Cosmos Tokenizer for temporal context encoding on the Ego4D dataset.

## Architecture Overview

The system combines visual and temporal features to predict future object bounding boxes and classes:

1. **SigLIP Query Encoder**: Extracts visual features from the query frame
2. **Cosmos Tokenizer**: Encodes context frames into temporal features
3. **DETR Decoder**: Fuses both feature streams to predict bounding boxes and object classes


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mariagabi-lustosa/AIMS_3rd_EgoVis.git
cd AIMS_3rd_EgoVis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
# or
conda create -n sta
conda activate sta
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Cosmos Tokenizer

```bash
# Clone the Cosmos Tokenizer repository
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer

# Install git-lfs and download models
git lfs install
git lfs pull

# Install the package
pip install -e .
cd ..
```

**Download pre-trained Cosmos models:**

```python
from huggingface_hub import login, snapshot_download

# Login to HuggingFace (you'll need a token)
login()

# Download the Cosmos model
model_name = "Cosmos-Tokenizer-DV8x8x8"
snapshot_download(
    repo_id=f"nvidia/{model_name}", 
    local_dir=f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}"
)
```

## Dataset Configuration

### Expected Data Structure

The project expects the Ego4D dataset organized as follows:

```
/hadatasets/ego4d_data/v1/
├── annotations/
│   └── fho_sta_train.json
└── video_540ss/
    ├── <video_uid_1>.mp4
    ├── <video_uid_2>.mp4
    └── ...
```

### Downloading Ego4D Dataset

1. Register at https://ego4d-data.org/
2. Download Short-Term Anticipation (STA) annotations
3. Download videos in 540p resolution (video_540ss)

### Path Configuration

Configure data paths in the training scripts:

- `bbox_prediction/train/train_detr_siglip.py`
- `bbox_prediction/train/train_detr_siglip_optimized.py`
- `bbox_prediction/cosmos/cosmos_tokenizer.py`

```python
sta_json = Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
full_scale_dir = Path("/hadatasets/ego4d_data/v1/video_540ss")
```

## Complete Usage Pipeline

### Step 1: Pre-compute Cosmos Features

The Cosmos Tokenizer processes 16 context frames around each query frame. Pre-compute these features before training:

```bash
python -m sta.bbox_prediction.cosmos.cosmos_tokenizer
```

**Configuration in `cosmos_tokenizer.py`:**
```python
CACHE_DIR = Path("/your/path/cosmos_cache_sta_train")
STA_JSON = Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
FULL_SCALE_DIR = Path("/hadatasets/ego4d_data/v1/video_540ss")
MODEL_DIR = Path("/work/your.user/sta/Cosmos-Tokenizer/pretrained_ckpts")
```

This script will:
- Load Ego4D videos
- Extract 16 context frames around each query frame (stride=2)
- Process frames through Cosmos Tokenizer
- Apply spatial pooling to get compact temporal features [16, 3]
- Save features as `.pt` files in the cache directory

**Features format:**
```python
cosmos_features: torch.Tensor  # Shape: [16, 3] per sample
# 16 temporal frames, 3 feature dimensions (spatially pooled)
```

### Step 2: Training with SigLIP + Cosmos

Train the complete model with both visual (SigLIP) and temporal (Cosmos) features:

```bash
python -m sta.bbox_prediction.train.train_detr_siglip
```

**Configure the dataset to use Cosmos features:**

In `train_detr_siglip.py`:
```python
# Uncomment line ~33 to enable Cosmos features
ds = STANounDetectionDataset(
    paths, 
    transform_query=tf, 
    min_box_size=1, 
    keep_metadata=False, 
    cosmos_cache_dir=Path("/your/path/cosmos_cache_sta_train")
)

# Uncomment line ~36 to use SiglipCosmosDETR model
model = SiglipCosmosDETR(
    siglip_name=model_name, 
    num_classes=num_nouns, 
    num_queries=50, 
    train_backbone=False
)
```



### Step 3: Monitor Training with TensorBoard

Training metrics are automatically logged:

```bash
tensorboard --logdir=runs
```

Access http://localhost:6006 to visualize:
- Total loss
- Classification loss
- Bounding box L1 loss
- GIoU loss
- Learning rate schedule

### Step 4: Evaluation and Visualization

#### Visualize Predictions

View model predictions on test samples:

```bash
python -m sta.bbox_prediction.eval.view_nouns_debug
```

**Configure checkpoint path:**
```python
ckpt = Path("checkpoints/detr_siglip_cosmos_best.pt")
```

This script:
- Loads a trained checkpoint
- Processes test samples
- Displays predicted vs. ground truth bounding boxes
- Shows object class labels

#### Plot Training Logs

Analyze training progression:

```bash
python -m sta.bbox_prediction.eval.plot_training_log
```

Generates plots for:
- Loss curves over epochs
- Learning dynamics
- Convergence analysis