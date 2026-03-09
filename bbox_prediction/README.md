## Model Architectures

### 1. SiglipCosmosDETR (Main Model with Temporal Context)

The complete model combining visual and temporal information:

**Components:**
- **SigLIP Backbone**: `google/siglip-base-patch16-224` (frozen)
- **Query Encoder**: Processes query frame into visual tokens
- **Cosmos Features**: Pre-computed temporal context [16, 3]
- **Projection Layers**: 
  - SigLIP: Linear(768 → 256)
  - Cosmos: Linear(3 → 256)
- **Memory Fusion**: Concatenates projected features
- **DETR Decoder**: 6-layer transformer decoder
- **Prediction Heads**:
  - Classification: Linear(256 → num_classes + 1)
  - Bounding box: MLP(256 → 4) with sigmoid

```python
model = SiglipCosmosDETR(
    siglip_name="google/siglip-base-patch16-224",
    num_classes=581,  # Ego4D noun categories
    num_queries=50,
    d_model=256,
    cosmos_dim=3,
    train_backbone=False
)
```

**Forward Pass:**
1. Query frame → SigLIP → visual tokens [B, N, 768]
2. Load pre-computed Cosmos features [B, 16, 3]
3. Project both to d_model=256
4. Concatenate: memory = [siglip_tokens | cosmos_tokens]
5. DETR decoder processes with learnable queries
6. Predict classes and bounding boxes

### 2. SiglipDETR (without Temporal Context)

Simplified version for comparison:

```python
model = SiglipDETR(
    siglip_name="google/siglip-base-patch16-224",
    num_classes=581,
    num_queries=50,
    num_decoder_layers=6,
    dim_feedforward=2048,
    train_backbone=False
)
```

Uses only query frame visual features without Cosmos temporal context.

### 3. SetCriterion (Loss Function)

Combines three loss components after Hungarian matching:

```python
criterion = SetCriterion(
    num_classes=num_nouns,
    matcher=HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    ),
    eos_coef=0.1  # Weight for "no object" class
)
```## Model Architectures

### 1. SiglipCosmosDETR (Main Model with Temporal Context)

The complete model combining visual and temporal information:

**Components:**
- **SigLIP Backbone**: `google/siglip-base-patch16-224` (frozen)
- **Query Encoder**: Processes query frame into visual tokens
- **Cosmos Features**: Pre-computed temporal context [16, 3]
- **Projection Layers**: 
  - SigLIP: Linear(768 → 256)
  - Cosmos: Linear(3 → 256)
- **Memory Fusion**: Concatenates projected features
- **DETR Decoder**: 6-layer transformer decoder
- **Prediction Heads**:
  - Classification: Linear(256 → num_classes + 1)
  - Bounding box: MLP(256 → 4) with sigmoid

```python
model = SiglipCosmosDETR(
    siglip_name="google/siglip-base-patch16-224",
    num_classes=581,  # Ego4D noun categories
    num_queries=50,
    d_model=256,
    cosmos_dim=3,
    train_backbone=False
)
```

**Forward Pass:**
1. Query frame → SigLIP → visual tokens [B, N, 768]
2. Load pre-computed Cosmos features [B, 16, 3]
3. Project both to d_model=256
4. Concatenate: memory = [siglip_tokens | cosmos_tokens]
5. DETR decoder processes with learnable queries
6. Predict classes and bounding boxes

### 2. SiglipDETR (without Temporal Context)

Simplified version for comparison:

```python
model = SiglipDETR(
    siglip_name="google/siglip-base-patch16-224",
    num_classes=581,
    num_queries=50,
    num_decoder_layers=6,
    dim_feedforward=2048,
    train_backbone=False
)
```

Uses only query frame visual features without Cosmos temporal context.

### 3. SetCriterion (Loss Function)

Combines three loss components after Hungarian matching:

```python
criterion = SetCriterion(
    num_classes=num_nouns,
    matcher=HungarianMatcher(
        cost_class=1.0,

**Loss Components:**
- **Classification Loss**: Focal cross-entropy for class prediction
- **L1 Bbox Loss**: L1 distance on normalized bbox coordinates
- **GIoU Loss**: Generalized Intersection over Union

## Data Formats

### Input

**Query Frame:**
```python
pixel_values: torch.Tensor  # [B, 3, 224, 224] - normalized image
```

**Cosmos Features:**
```python
cosmos_feats: torch.Tensor  # [B, 16, 3] - pre-computed temporal features
```

**Targets (Ground Truth):**
```python
{
    'labels': torch.Tensor,  # [num_boxes] - class indices (0 to num_classes-1)
    'boxes': torch.Tensor,   # [num_boxes, 4] - (cx, cy, w, h) normalized to [0, 1]
}
```

### Model Output

```python
{
    'pred_logits': torch.Tensor,  # [B, num_queries, num_classes+1] - classification logits
    'pred_bboxes': torch.Tensor,  # [B, num_queries, 4] - bbox (cx, cy, w, h) in [0, 1]
}
```

### Evaluation Metrics

**Logged during training:**
- Total loss
- Classification loss
- L1 bbox loss
- GIoU loss
- Learning rate

**Standard detection metrics:**
- **mAP** (mean Average Precision): Primary detection metric
- **Recall**: Percentage of objects correctly detected
- **Precision**: Percentage of correct detections

**Note:** For complete mAP evaluation needs to be implemented.

## Outputs and Checkpoints

### Checkpoints

Saved in `../checkpoints/`:

```
checkpoints/
├── detr_siglip_cosmos_epoch1.pt
├── detr_siglip_cosmos_epoch2.pt
├── ...
├── detr_siglip_cosmos_epoch20.pt
└── detr_siglip_cosmos_best.pt  # Best model (lowest validation loss)
```

### Logs

**JSON Logs:** `../runs/train_log.jsonl`
```json
{"step": 0, "epoch": 0, "loss": 15.234, "loss_ce": 8.5, "loss_bbox": 4.2, "loss_giou": 2.5, "lr": 0.0001}
```

**TensorBoard:** `../runs/detr_siglip_cosmos/`
