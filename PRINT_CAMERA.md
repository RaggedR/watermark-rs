# Print-Camera Attack Analysis

> **Status**: Testing complete, halftoning is the critical vulnerability
>
> **Key Finding**: Frequency-domain watermarks do not survive print-camera pipeline

## Summary

| Attack Type | Survivable? | Root Cause |
|-------------|-------------|------------|
| High-DPI scanner | ~67% | Minimal halftoning, no perspective |
| Photo of print | 0% | Halftoning + perspective + lighting |
| T-shirt photo | 0% | Fabric texture + halftoning |
| Mug photo | 0% | Cylindrical warp + halftoning |

## Test Results (Strength 8.0)

```
Scenario                Detection Rate    Message Accuracy
────────────────────────────────────────────────────────────
Paper (Scanner)         66.7%             66.7%
Paper (Photo)           0.0%              0.0%
T-Shirt (Flat)          0.0%              0.0%
T-Shirt (Worn)          0.0%              0.0%
Mug (Curved)            0.0%              0.0%
Poster (Large)          0.0%              0.0%

OVERALL: 11.1% detection
```

## Why Print-Camera Fails

### 1. Halftoning Destroys High Frequencies

Printing converts continuous tones to discrete dots:

```
Original:     Smooth gradients, all frequencies present
                      ↓
Halftoned:    Binary dots, high frequencies = noise
                      ↓
DFT impact:   Watermark frequencies overwhelmed by halftone pattern
```

| Print DPI | Halftone Pattern | Watermark Survival |
|-----------|------------------|-------------------|
| 300+ | Fine dots (barely visible) | Possible |
| 150 | Visible dots | Destroyed |
| 100 | Coarse dots | Completely destroyed |

### 2. Geometric Distortion Breaks Correlation

Our Legendre watermark relies on exact frequency positions:

```
Embedded:  DFT[i, L(i)] for i in 1..p

After perspective:
  DFT'[i, j] ≠ DFT[i, L(i)]  -- positions shifted non-uniformly

Correlation:
  Σ DFT'[i, j] × L(i) → near zero (no alignment)
```

### 3. Substrate Texture Adds Noise

| Substrate | Texture Type | Impact |
|-----------|--------------|--------|
| Paper | Fine grain | Adds high-frequency noise |
| Canvas | Coarse weave | Periodic interference patterns |
| Fabric | Thread structure | Strong periodic noise |
| Ceramic | Glaze variations | Random noise + reflection |

## Attack Pipeline Breakdown

```
Original Image
     ↓
[RGB→CMYK→RGB]     ← Color gamut loss
     ↓
[Halftoning]        ← Most destructive (frequency destruction)
     ↓
[Ink Bleeding]      ← Blur/spread
     ↓
[Substrate Texture] ← Added noise pattern
     ↓
[Perspective]       ← Geometric misalignment
     ↓
[Lighting]          ← Uneven exposure
     ↓
[Camera Noise]      ← Sensor noise
     ↓
[JPEG Compression]  ← Final degradation
```

## Comparison: Social vs Print-Camera

| Attack Type | Best Case | Worst Case |
|-------------|-----------|------------|
| **Social Network** | 100% (no resize) | 0% (with resize) |
| **Print-Camera** | 67% (scanner) | 0% (any photo) |

Print-camera is harder because:
1. Halftoning is more destructive than JPEG
2. Physical-to-digital introduces geometric error
3. Multiple compounding distortions

## Theoretical Solutions

### Solution 1: Accept Limitations (Recommended)

For NFTmarket's use case:
- Original PNG on IPFS → Always has watermark
- Print merchandise → Watermark lost, but:
  - Physical item has value beyond digital
  - Original on-chain proves ownership
  - Can detect if someone digitizes and claims as own

### Solution 2: Spatial-Domain Watermark

Instead of frequency domain, embed directly in pixel values:

```
Frequency domain: Fragile to halftoning
Spatial domain:   More robust to printing
                  But: Visible, less capacity
```

Examples: Visible watermark, QR code overlay

### Solution 3: Hybrid Approach

Embed two watermarks:
1. High-capacity Legendre (digital-to-digital)
2. Robust low-capacity mark (survives print)

Trade-off: Reduced capacity for primary watermark.

### Solution 4: Multi-Scale Embedding

Embed watermark at multiple frequencies:

```
f_high: Full capacity, digital-only
f_mid:  Reduced capacity, survives mild print
f_low:  Minimal capacity, survives harsh print
```

### Solution 5: Template Matching

Instead of blind detection, use reference:

```
1. Store original embedding pattern
2. For detection: warp attacked image to match original
3. Correlate after alignment
```

Requires database lookup, not truly blind.

## Recommendations for NFTmarket

### Primary (Digital) Watermark

Continue using Legendre frequency-domain watermark for:
- IPFS storage (PNG, lossless)
- High-quality digital distribution
- Proof of ownership disputes

### Physical Merchandise

Accept that watermark won't survive, but:
1. **Visible branding**: Add small logo/signature overlay
2. **Metadata embedding**: Store ownership info in EXIF
3. **Perceptual hashing**: Database lookup for known images
4. **Legal protection**: License explicitly covers physical reproduction

## Test Command

```bash
python3 print_camera_attack.py ~/path/to/images --sample 10 \
    --scenarios paper_scan paper_photo tshirt_worn mug \
    --strength 8.0
```

## Files

| File | Purpose |
|------|---------|
| `print_camera_attack.py` | Attack simulation and testing |
| `PRINT_CAMERA.md` | This file - findings and analysis |

---

Last updated: 2026-02-10
