# Neural Watermark Recovery Network

## Overview

A neural network that recovers Legendre watermarks from JPEG-attacked images,
enabling detection of wallet addresses even after compression.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Original Image                                                         │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────┐                                                        │
│   │  Legendre   │  ← Random wallet address                               │
│   │  Embedding  │                                                        │
│   └─────────────┘                                                        │
│        │                                                                 │
│        ▼                                                                 │
│   Watermarked Image ──────────────────────┐                              │
│        │                                  │                              │
│        ▼                                  │                              │
│   ┌─────────────┐                         │                              │
│   │ JPEG Attack │  Q50-Q95 random         │                              │
│   └─────────────┘                         │                              │
│        │                                  │                              │
│        ▼                                  ▼                              │
│   Attacked Image              Target: Watermark Region DFT               │
│        │                                  │                              │
│        ▼                                  │                              │
│   ┌─────────────┐                         │                              │
│   │   Neural    │                         │                              │
│   │   Network   │                         │                              │
│   └─────────────┘                         │                              │
│        │                                  │                              │
│        ▼                                  │                              │
│   Predicted DFT ◄─────── L1 Loss ─────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture: Frequency Recovery Network (FRNet)

### Design Choices

1. **Output**: Directly predict DFT coefficients (not full image)
   - More efficient — only predict what we need
   - Size: p×p×2 (real + imaginary) where p ≈ 500-600

2. **Input**: Grayscale image (H×W×1)
   - Matches our embedding (grayscale/V-channel)

3. **Architecture**: U-Net style with frequency-aware layers
   - Encoder: Extract features at multiple scales
   - Decoder: Reconstruct watermark region DFT
   - Skip connections: Preserve fine details

### Network Architecture

```
Input: Grayscale image (H×W×1)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  ENCODER                                                      │
├──────────────────────────────────────────────────────────────┤
│  Conv2d(1→32, 3×3) + ReLU + BatchNorm                        │
│  Conv2d(32→32, 3×3) + ReLU + BatchNorm                       │
│  MaxPool2d(2×2)                                    → skip1    │
│       │                                                       │
│  Conv2d(32→64, 3×3) + ReLU + BatchNorm                       │
│  Conv2d(64→64, 3×3) + ReLU + BatchNorm                       │
│  MaxPool2d(2×2)                                    → skip2    │
│       │                                                       │
│  Conv2d(64→128, 3×3) + ReLU + BatchNorm                      │
│  Conv2d(128→128, 3×3) + ReLU + BatchNorm                     │
│  MaxPool2d(2×2)                                    → skip3    │
│       │                                                       │
│  Conv2d(128→256, 3×3) + ReLU + BatchNorm                     │
│  Conv2d(256→256, 3×3) + ReLU + BatchNorm                     │
│       │                                                       │
└───────┼──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  BOTTLENECK                                                   │
├──────────────────────────────────────────────────────────────┤
│  Conv2d(256→512, 3×3) + ReLU + BatchNorm                     │
│  Conv2d(512→256, 3×3) + ReLU + BatchNorm                     │
└───────┼──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  DECODER (Frequency Branch)                                   │
├──────────────────────────────────────────────────────────────┤
│  Upsample(2×) + Concat(skip3)                                │
│  Conv2d(256+128→128, 3×3) + ReLU + BatchNorm                 │
│  Conv2d(128→128, 3×3) + ReLU + BatchNorm                     │
│       │                                                       │
│  Upsample(2×) + Concat(skip2)                                │
│  Conv2d(128+64→64, 3×3) + ReLU + BatchNorm                   │
│  Conv2d(64→64, 3×3) + ReLU + BatchNorm                       │
│       │                                                       │
│  Upsample(2×) + Concat(skip1)                                │
│  Conv2d(64+32→32, 3×3) + ReLU + BatchNorm                    │
│  Conv2d(32→32, 3×3) + ReLU + BatchNorm                       │
│       │                                                       │
│  AdaptiveAvgPool2d(p×p)   ← resize to prime size             │
│  Conv2d(32→2, 1×1)        ← output real + imaginary          │
│       │                                                       │
└───────┼──────────────────────────────────────────────────────┘
        │
        ▼
Output: DFT watermark region (p×p×2)
```

### Parameters

| Layer Group | Parameters |
|-------------|------------|
| Encoder | ~500K |
| Bottleneck | ~1.2M |
| Decoder | ~500K |
| **Total** | **~2.2M** |

Small enough to train on 1,000 images without overfitting.

## Loss Function

### Primary: Watermark Region MSE

```python
def watermark_loss(pred_dft, target_dft, p):
    # Only care about watermark region [1:p+1, 1:p+1]
    pred_wm = pred_dft[:, :, 1:p+1, 1:p+1]
    target_wm = target_dft[:, :, 1:p+1, 1:p+1]
    return F.mse_loss(pred_wm, target_wm)
```

### Secondary: Detection Confidence Loss (optional)

```python
def detection_loss(pred_dft, message, p):
    # Run Legendre correlation on predicted DFT
    # Maximize correlation peak for correct shifts
    confidence = compute_peak_to_rms(pred_dft, message, p)
    return -confidence  # Maximize
```

### Combined Loss

```python
loss = watermark_loss + 0.1 * detection_loss
```

## Training Procedure

### Data Generation (on-the-fly)

```python
def generate_sample(image, p):
    # 1. Generate random wallet address
    wallet = generate_random_wallet()  # "0x" + 40 hex chars

    # 2. Embed watermark
    watermarked = embed_legendre(image, wallet, strength=2.0)

    # 3. Compute target DFT (before attack)
    target_dft = compute_dft(watermarked)[:p, :p]

    # 4. Apply random JPEG attack
    quality = random.randint(50, 95)
    attacked = jpeg_compress(watermarked, quality)

    return attacked, target_dft, wallet
```

### Training Loop

```python
for epoch in range(100):
    for image in dataset:
        attacked, target_dft, wallet = generate_sample(image, p)

        pred_dft = model(attacked)
        loss = watermark_loss(pred_dft, target_dft)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Epochs | 100 |
| Image size | 512×512 (resized) |
| JPEG quality range | 50-95 |
| Watermark strength | 2.0 |

## Inference Pipeline

```python
def detect_with_neural(attacked_image, model, p):
    # 1. Run neural network to recover DFT
    pred_dft = model(attacked_image)

    # 2. Run standard Legendre detection on recovered DFT
    result = detect_from_dft(pred_dft, p, threshold=6.0)

    return result.message  # Wallet address
```

## Data Augmentation

During training, apply additional augmentations after JPEG:

| Augmentation | Probability | Range |
|--------------|-------------|-------|
| JPEG compression | 100% | Q50-Q95 |
| Gaussian noise | 30% | σ=0.01-0.05 |
| Brightness shift | 30% | ±10% |
| Contrast shift | 30% | ±10% |
| Gaussian blur | 20% | σ=0.5-1.5 |
| Resize (down+up) | 20% | 0.7-0.9× |

## File Structure

```
neural/
├── DESIGN.md           # This file
├── model.py            # Network architecture
├── dataset.py          # Data loading + augmentation
├── train.py            # Training script
├── inference.py        # Detection with neural enhancement
├── utils.py            # DFT helpers, Legendre interface
└── weights/            # Saved model checkpoints
    └── frnet_v1.pth
```

## Integration with Rust

Two options:

### Option A: Python subprocess
```rust
// Call Python script for neural detection
let output = Command::new("python3")
    .args(["neural/inference.py", image_path])
    .output()?;
```

### Option B: ONNX Runtime
```rust
// Export PyTorch model to ONNX, load in Rust
let model = ort::Session::new("neural/weights/frnet.onnx")?;
let output = model.run(input_tensor)?;
```

Option B is faster but more complex. Start with Option A.

## Success Metrics

| Metric | Target |
|--------|--------|
| Detection rate on JPEG Q70 | >90% |
| Detection rate on JPEG Q50 | >70% |
| Bit accuracy on detected messages | >99% |
| Inference time | <500ms per image |
| False positive rate | <1% |

## Next Steps

1. [ ] Implement `model.py` with FRNet architecture
2. [ ] Implement `dataset.py` with on-the-fly generation
3. [ ] Implement `train.py` training loop
4. [ ] Train on 1,000 artwork images
5. [ ] Evaluate on held-out test set
6. [ ] Integrate with Rust detection CLI
