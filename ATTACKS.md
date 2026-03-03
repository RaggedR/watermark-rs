# Watermark Attack Library Documentation

This document catalogs attacks used to test and train watermark robustness. Understanding these attacks is essential for building robust watermarking systems.

## Overview

| Category | Standard Attacks | Google SynthID | Our Library |
|----------|-----------------|----------------|-------------|
| Geometric | 6 | 4 | 6 |
| Signal Processing | 8 | 7 | 4 |
| Color/Tone | 7 | 6 | 3 |
| Real-World | 4 | 3 | 0 |
| Custom/Novel | 0 | ? | 1 |
| **Total** | **25** | **20+** | **14** |

---

## Standard Attacks (Academic Benchmarks)

These attacks appear in virtually every watermarking paper and form the baseline for robustness evaluation.

### Geometric Attacks

| Attack | Parameters | Description |
|--------|------------|-------------|
| **Rotation** | ±0.5° to ±45° | Global image rotation |
| **Scaling** | 0.5x to 2.0x | Uniform resize |
| **Cropping** | 1% to 50% | Remove edges, resize back |
| **Shearing** | 0.01 to 0.2 | Horizontal/vertical skew |
| **Flipping** | H/V | Mirror image |
| **Aspect Ratio** | 0.8 to 1.2 | Non-uniform scaling |

### Signal Processing Attacks

| Attack | Parameters | Description |
|--------|------------|-------------|
| **JPEG Compression** | Q=10 to Q=95 | Lossy compression artifacts |
| **Gaussian Noise** | σ=0.01 to σ=0.1 | Additive white noise |
| **Salt & Pepper** | p=0.01 to p=0.1 | Impulse noise |
| **Gaussian Blur** | k=3 to k=11 | Low-pass filtering |
| **Median Filter** | k=3 to k=7 | Non-linear smoothing |
| **Sharpening** | α=0.5 to α=2.0 | High-pass enhancement |
| **Histogram Equalization** | — | Contrast normalization |
| **Quantization** | 4-bit to 7-bit | Color depth reduction |

### Color/Tone Attacks

| Attack | Parameters | Description |
|--------|------------|-------------|
| **Brightness** | ±20% | Exposure adjustment |
| **Contrast** | 0.7x to 1.5x | Dynamic range change |
| **Saturation** | 0x to 2.0x | Color intensity |
| **Hue Shift** | ±30° | Color rotation |
| **Gamma Correction** | γ=0.5 to γ=2.0 | Tone curve |
| **Grayscale** | — | Remove color |
| **Color Space Conv.** | RGB↔YUV↔LAB | Channel transformation |

### Real-World Attacks

| Attack | Parameters | Description |
|--------|------------|-------------|
| **Print-Scan** | 150-600 DPI | Physical reproduction |
| **Print-Camera** | Various phones | Photo of printout |
| **Screenshot** | Various displays | Screen capture |
| **Social Media** | IG/TW/FB | Platform compression |

---

## Google SynthID Attacks

Based on the [SynthID-Image paper](https://arxiv.org/html/2510.09263v1) (Google DeepMind, 2024).

### Documented Attacks

#### Geometric
- Rotation (including small angles)
- Flips (left-right and up-down)
- Cropping
- Resizing

#### Signal Processing
- JPEG compression
- Quantization
- File format conversion
- Gaussian noise
- Impulse noise
- Shot noise
- Speckle noise

#### Color/Tone
- Brightness adjustment
- Contrast adjustment
- Exposure shifts
- Saturation variation
- Hue changes
- Grayscale conversion

#### Filtering
- Gaussian blur
- Sharpening
- Denoising
- Zoom

#### Real-World / Social Media
- Instagram-like filters
- Emoji overlays
- Light text overlays

### Key Insight: Combined Transforms

Google emphasizes testing **combinations** of attacks:

> "Combinations of transforms can be effective ways to fool networks"

```python
# Google's approach: chain multiple attacks
def combined_attack(image):
    for _ in range(random.randint(1, 4)):
        attack = random.choice(attack_library)
        image = attack(image)
    return image
```

### Suspected Proprietary Attacks (Not Published)

| Attack | Evidence | Purpose |
|--------|----------|---------|
| Elastic mesh warping | Survives ring attack | Local distortion robustness |
| Exact social media pipelines | Partnership with platforms | Real-world robustness |
| Adversarial removal networks | Standard practice | Arms race defense |
| Print-camera simulation | Mentioned but not detailed | Physical robustness |

---

## Custom Attacks (Our Additions)

These attacks are not standard benchmarks but represent real threats.

### Ring Attack

**Category:** Local Geometric (spatially-varying)

**Description:** Divides image into concentric rings and rotates alternate rings in opposite directions.

**Parameters:**
- `num_rings`: Number of concentric rings (5-30)
- `max_angle`: Maximum rotation angle (0.1° - 5.0°)

**Why It's Dangerous:**
- Not in standard benchmarks
- Cannot be corrected by global rotation
- Destroys frequency-domain watermarks at ±0.06°
- Pre-trained models (StegaStamp, HiDDeN) fail against it

**Structure:**
```
    ┌─────────────────┐
    │  ↻ +θ          │   Ring 1: Clockwise
    │ ┌───────────┐  │
    │ │  ↺ -θ    │  │   Ring 2: Counter-clockwise
    │ │ ┌─────┐  │  │
    │ │ │ ↻+θ │  │  │   Ring 3: Clockwise
    │ │ └─────┘  │  │
    │ └───────────┘  │   ... alternating
    └─────────────────┘
```

**Implementation:** See `attacks/attack_library.py::RingAttack`

---

## Attack Taxonomy

```
ATTACKS
├── GLOBAL GEOMETRIC (uniform transform)
│   ├── Rotation
│   ├── Scale
│   ├── Translation
│   ├── Shear
│   └── Flip
│
├── LOCAL GEOMETRIC (spatially-varying)
│   ├── Perspective
│   ├── Lens distortion
│   ├── Elastic warping
│   ├── Wave distortion
│   └── Ring attack ★
│
├── SIGNAL PROCESSING (pixel-wise)
│   ├── Compression (JPEG, WebP)
│   ├── Noise (Gaussian, S&P, Speckle)
│   ├── Filtering (Blur, Sharpen, Median)
│   └── Quantization
│
├── COLOR/TONE
│   ├── Brightness/Contrast
│   ├── Saturation/Hue
│   ├── Gamma
│   └── Color space conversion
│
└── REAL-WORLD
    ├── Print-scan
    ├── Print-camera
    ├── Screenshot
    └── Social media compression
```

---

## Generalization Rules

Neural networks generalize **within** categories but NOT **across** them:

| Trained On | Generalizes To | Does NOT Generalize To |
|------------|----------------|------------------------|
| ±30° rotation | ±45° rotation | Shear, perspective |
| Gaussian blur σ=2 | Gaussian blur σ=4 | Motion blur |
| JPEG Q=50 | JPEG Q=30 | WebP compression |
| Elastic warping | Similar warping | **Ring attack** |

**Key Insight:** The ring attack has a unique alternating concentric structure that generic local distortions don't cover. It must be explicitly included in training.

---

## Robustness Comparison

### Our Rust Implementation (Traditional)

| Attack | Threshold | Status |
|--------|-----------|--------|
| Rotation (no pilot) | ±0.05° | ⚠️ Fragile |
| Rotation (with pilot) | ±5.0° | ✅ Robust |
| Scaling up | ≤110% | ✅ Robust |
| Scaling down | ≥99% | ⚠️ Fragile |
| Cropping | ≤10% | ✅ Robust |
| Shearing | ≥1% | ❌ Fails |
| Ring attack | ≥±0.06° | ❌ Fails |
| JPEG | Q≥70 | ✅ Robust |

### Neural Watermarking (Trained)

| Attack | If Trained | If Not Trained |
|--------|------------|----------------|
| Any standard attack | ✅ Robust | ❌ Fails |
| Ring attack | ✅ Robust | ❌ Fails |
| Novel attack X | ❌ Unknown | ❌ Fails |

---

## Implementation

### Python Attack Library

Location: `attacks/attack_library.py`

```python
from attack_library import AttackLayer

# Composite attack layer for training
attack_layer = AttackLayer(
    p_geometric=0.5,  # 50% chance of geometric attack
    p_signal=0.5,     # 50% chance of signal attack
    p_color=0.3       # 30% chance of color attack
)

# Apply random attacks during training
attacked_image = attack_layer(watermarked_image)
```

### Adding New Attacks

1. Create a new `nn.Module` class
2. Implement differentiable forward pass (for training)
3. Add to appropriate attack list in `AttackLayer`
4. Test with `python attacks/attack_library.py`

Example:
```python
class MyNovelAttack(nn.Module):
    def __init__(self, param: float = 1.0):
        super().__init__()
        self.param = param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement attack (must be differentiable!)
        return attacked_x
```

---

## References

1. [SynthID-Image: Image watermarking at internet scale](https://arxiv.org/html/2510.09263v1) - Google DeepMind, 2024
2. [HiDDeN: Hiding Data with Deep Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.pdf) - ECCV 2018
3. [StegaStamp: Invisible Hyperlinks in Physical Photographs](https://arxiv.org/abs/1904.05343) - CVPR 2020
4. [Probing SynthID](https://www.sri.inf.ethz.ch/blog/probingsynthid) - ETH Zurich SRI Lab

---

## Changelog

- **2024-02**: Initial documentation
- **2024-02**: Added ring attack to custom attacks
- **2024-02**: Added Google SynthID analysis
