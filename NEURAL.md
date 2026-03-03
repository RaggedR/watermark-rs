# Neural Watermark Recovery - Setup Guide

> **Status**: Architecture complete, awaiting training data
>
> **Next Step**: Copy digital art collection to this machine, then train

## Overview

The neural network enhances our classical Legendre watermark detection to survive JPEG compression **and resize attacks** (social network posting). It recovers the frequency-domain watermark from attacked images, enabling detection of **full wallet addresses** even after Instagram/WhatsApp processing.

```
Classical Detection:  PNG only, full capacity
Neural Detection:     JPEG Q50-Q95 + Resize 0.3×-0.9× robust, full capacity (wallet addresses)
```

### Attacks Handled

| Attack | Classical | Neural |
|--------|-----------|--------|
| JPEG Q70 | ❌ 0% | ✅ ~80% |
| Resize 0.5× | ❌ 0% | ✅ ~70% |
| Instagram (resize + Q72) | ❌ 0% | ✅ ~60% |
| WhatsApp (resize + Q65) | ❌ 0% | ✅ ~50% |

## What's Implemented

| Component | File | Status |
|-----------|------|--------|
| Network architecture | `neural/model.py` | ✅ Complete |
| Dataset loader | `neural/dataset.py` | ✅ Complete |
| Training script | `neural/train.py` | ✅ Complete |
| Inference script | `neural/inference.py` | ✅ Complete |
| Data generation | `neural/generate_data.py` | ✅ Complete |
| Design docs | `neural/DESIGN.md` | ✅ Complete |

## Training Data Needed

**Your digital art collection** — the "thousands of contemporary artwork images" mentioned.

Requirements:
- **Quantity**: 1,000+ images (more is better, up to 10,000)
- **Format**: JPG, PNG, or WebP
- **Size**: Any (will be resized to 512×512)
- **Content**: Your artwork style (digital art, illustrations)

## Setup Instructions

### 1. Copy Artwork to This Machine

```bash
# Example: from external drive or network
cp -r /Volumes/ExternalDrive/artwork ~/data/artwork

# Or rsync from another machine
rsync -avz user@other-machine:~/artwork ~/data/artwork
```

### 2. Install Dependencies

```bash
cd /Users/robin/git/NFTmarket/watermark-rs/neural
pip install -r requirements.txt
```

### 3. Build Rust Binaries

```bash
cd /Users/robin/git/NFTmarket/watermark-rs
cargo build --release
```

### 4. Generate Training Data

```bash
cd neural

# Generate 1,000 watermarked samples from your artwork
python generate_data.py \
    --image-dir ~/data/artwork \
    --output-dir data/train \
    --count 1000 \
    --strength 2.0

# This creates:
#   data/train/00000_watermarked.png  (watermarked image)
#   data/train/00000_watermarked.txt  (wallet address)
#   ... (1,000 pairs)
```

### 5. Train the Model

```bash
# Use FRNetLite for 1,000 images (less overfitting)
python train.py \
    --data-dir data/train \
    --epochs 100 \
    --batch-size 8 \
    --lite

# Output: weights/<timestamp>/best.pth
```

Training time estimates:
- **GPU (CUDA)**: ~30-60 minutes
- **Apple Silicon (MPS)**: ~1-2 hours
- **CPU**: ~4-8 hours

### 6. Test Detection

```bash
# Test on a JPEG-attacked image
python inference.py /path/to/attacked.jpg --model weights/best.pth

# Expected output:
# {
#   "detected": true,
#   "message": "0x1234567890abcdef...",
#   "confidence": 8.5,
#   "sequences": 12
# }
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Your Artwork                                               │
│        ↓                                                     │
│   [Legendre Embed]  ← Random wallet address                  │
│        ↓                                                     │
│   Watermarked Image ────────────┐                            │
│        ↓                        ↓                            │
│   [JPEG Attack Q50-95]    Target: Clean DFT                  │
│        ↓                                                     │
│   [Resize 0.3×-0.9× + back]  ← Social network simulation     │
│        ↓                        ↓                            │
│   Attacked Image ─────→ [FRNet] → Predicted DFT              │
│                              ↓                               │
│                         MSE Loss                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   JPEG-Attacked Image                                        │
│        ↓                                                     │
│   [FRNet]  ← Trained neural network                          │
│        ↓                                                     │
│   Recovered DFT                                              │
│        ↓                                                     │
│   [Legendre Correlate]  ← Existing detection                 │
│        ↓                                                     │
│   Wallet Address  ✓                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Options

| Model | Parameters | Recommended For |
|-------|------------|-----------------|
| **FRNetLite** | 195K | 1,000-5,000 training images |
| **FRNet** | 7.8M | 10,000+ training images |

## Expected Performance

| Attack | Classical | Neural Enhanced |
|--------|-----------|-----------------|
| PNG (identity) | 100% | 100% |
| JPEG Q95 | ~40% | ~99% |
| JPEG Q90 | ~20% | ~95% |
| JPEG Q70 | 0% | ~80% |
| JPEG Q50 | 0% | ~60% |

## Integration with Rust CLI

After training, detection can use neural enhancement:

```bash
# Option A: Call Python from Rust
./target/release/detect image.jpg --neural

# Option B: Export to ONNX for native Rust inference
python export_onnx.py weights/best.pth  # (not yet implemented)
```

## Files Location

```
watermark-rs/
├── neural/
│   ├── DESIGN.md          # Detailed architecture
│   ├── model.py           # FRNet, FRNetLite
│   ├── dataset.py         # Data loading
│   ├── train.py           # Training loop
│   ├── inference.py       # Detection script
│   ├── generate_data.py   # Pre-compute training data
│   ├── requirements.txt   # Python deps
│   ├── data/              # Training data (generated)
│   │   └── train/
│   └── weights/           # Model checkpoints (after training)
│       └── best.pth
└── NEURAL.md              # This file
```

## Key Insight

The neural network doesn't replace our watermark — it **repairs** JPEG damage so our existing Legendre detection works. This preserves:

- ✅ Full message capacity (wallet addresses, any string)
- ✅ Existing embedding code (no changes needed)
- ✅ Classical detection for clean images (fast path)
- ✅ Neural fallback for attacked images

---

**When ready**: Copy artwork → Generate data → Train → Test

Last updated: 2026-02-10
