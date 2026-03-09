# STA - Short-Term Anticipation with DETR and SigLIP

This project implements an object detection and short-term anticipation system using DETR (Detection Transformer) with SigLIP backbone for the Ego4D dataset.


## Installation

### 1. Clone the repository

```bash
git clone <your-repository>
cd sta
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
conda create -m venv
conda activate venv
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Cosmos Tokenizer

NVIDIA's Cosmos Tokenizer is included as a submodule. To install it:

```bash
cd Cosmos-Tokenizer
pip install -e .
cd ..
```

## Dataset Configuration

### Expected Data Structure

The project expects the Ego4D dataset to be organized as follows:

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

Data paths are configured in the training scripts. You can modify them in:

- `bbox_prediction/train/train_detr_siglip.py`
- `bbox_prediction/train/train_detr_siglip_optimized.py`

Change the following lines to point to your data:

```python
sta_json = Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
full_scale_dir = Path("/hadatasets/ego4d_data/v1/video_540ss")
```

## Usage

### Training

#### DETR + SigLIP Training (Optimized)

Run the optimized training script:

```bash
python -m sta.bbox_prediction.train.train_detr_siglip_optimized
```

This script includes:
- Mixed precision training (FP16)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- TensorBoard logging

#### DETR + SigLIP Training (Basic version)

```bash
python -m sta.bbox_prediction.train.train_detr_siglip
```

### Training Parameters

You can adjust the following hyperparameters in the training scripts:

- `BATCH_SIZE`: Batch size (default: 16)
- `NUM_WORKERS`: Number of DataLoader workers (default: 4)
- `NUM_EPOCHS`: Number of epochs (default: 20)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps (default: 2)

### Monitoring with TensorBoard

During training, metrics are automatically saved. To visualize them:

```bash
tensorboard --logdir=../runs
```

Access http://localhost:6006 in your browser.

### Evaluation and Visualization

#### Visualize Predictions

```bash
python -m sta.bbox_prediction.eval.view_nouns_debug
```

#### Plot Training Logs

```bash
python -m sta.bbox_prediction.eval.plot_training_log
```

#### Dataset Sanity Check

```bash
python -m sta.bbox_prediction.eval.sanity_check
```

## Models

### Available Architectures

1. **SiglipDETR**: DETR with SigLIP base backbone
   - Vision encoder: `google/siglip-base-patch16-224`
   - Decoder: 6-layer transformer
   - Queries: 50 object queries

2. **SiglipCosmosDETR**: DETR with SigLIP + Cosmos Tokenizer
   - Includes Cosmos features for temporal representation

### Components

- **SiglipQueryEncoder**: Query frame encoder using SigLIP
- **SetCriterion**: Loss function for DETR training
- **HungarianMatcher**: Bipartite matching between predictions and ground truth

## Outputs

### Checkpoints

Checkpoints are saved in `../checkpoints/` with the following format:
- `detr_siglip_optimized_epoch{N}.pt`: Checkpoint for each epoch
- `detr_siglip_optimized_best.pt`: Best model (lowest loss)

### Logs

- JSON logs: `../runs/train_log_optimized_{N}.jsonl`
- TensorBoard: `../runs/detr_siglip_optimized/`

## Output Data Structure

### Prediction Format

```python
{
    'pred_logits': torch.Tensor,  # [B, num_queries, num_classes+1]
    'pred_boxes': torch.Tensor,   # [B, num_queries, 4] (cx, cy, w, h) normalized
}
```

### Ground Truth Format

```python
{
    'labels': torch.Tensor,  # [num_boxes] class indices
    'boxes': torch.Tensor,   # [num_boxes, 4] (cx, cy, w, h) normalized
}
```
# AIMS_3rd_EgoVis
# AIMS_3rd_EgoVis
