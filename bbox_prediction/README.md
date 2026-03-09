## Model Architectures

### 1. Siglip + Cosmos + DETR

The complete model combining visual and temporal information:

**Components:**
- **SigLIP Backbone**: `google/siglip-base-patch16-224` (frozen)
- **Query Encoder**: Processes query frame into visual tokens
- **Cosmos Features**: Pre-computed temporal context
- **Projection Layers**: 
  - SigLIP
  - Cosmos
- **Memory Fusion**: Concatenates projected features
- **DETR Decoder**: transformer decoder
- **Prediction Heads**:
  - Classification
  - Bounding box: MLP with sigmoid

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

**Pipeline:**
1. Query frame → SigLIP → visual tokens [B, N, 768]
2. Load pre-computed Cosmos features [B, 16, 3]
3. Project both to d_model=256
4. Concatenate: memory = [siglip_tokens | cosmos_tokens]
5. DETR decoder processes with learnable queries
6. Predict classes and bounding boxes

### 2. Siglip + DETR (without Temporal Context)

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
```

**Loss Components:**
- **Classification Loss**: Cross-entropy for class prediction
- **L1 Bbox Loss**: L1 distance on normalized bbox coordinates
- **GIoU Loss**: Generalized Intersection over Union


## Evaluation Metrics

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

**Note:** For complete evaluation, mAP needs to be implemented.
