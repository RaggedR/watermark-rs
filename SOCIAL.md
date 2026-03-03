# Social Network Attack Analysis

> **Status**: Testing complete, resize is the critical vulnerability
>
> **Key Finding**: JPEG compression is survivable at high strength, but resize destroys watermarks

## Summary

| Attack Type | Survivable? | Solution |
|-------------|-------------|----------|
| JPEG compression | ✅ At strength 8+ | Use higher embedding strength |
| Instagram filters | ✅ Minimal impact | Color transforms don't affect grayscale |
| Resize/downsample | ❌ Complete loss | Multi-scale embedding or neural recovery |

## Platform Configurations

| Platform | Max Dimension | JPEG Quality | Resize Risk |
|----------|---------------|--------------|-------------|
| **Discord** | 4000px | Q90 | Low |
| **Twitter/X** | 4096px | Q85 | Low |
| **Telegram** | 2560px | Q82 | Medium |
| **Facebook** | 2048px | Q80 | Medium |
| **Instagram** | 1080px | Q72 | High |
| **TikTok** | 1080px | Q75 | High |
| **WhatsApp** | 1600px | Q65 | High |

## Test Results

### Strength 2.0 (Default - Invisible)

```
Platform        JPEG Quality    Detection Rate
─────────────────────────────────────────────
Discord         Q90             0%
Twitter/X       Q85             0%
Instagram       Q72             0%
WhatsApp        Q65             0%

OVERALL: 0/36 (0.0%) - Complete failure
```

**Conclusion**: Default invisible watermark does not survive any social network.

### Strength 8.0 (Robust - May Be Visible)

```
Platform        JPEG Quality    Detection Rate    Avg Confidence
────────────────────────────────────────────────────────────────
Discord         Q90             100%              26.3
Twitter/X       Q85             100%              22.2
Instagram       Q72             67%               16.3
WhatsApp        Q65             67%               13.9

OVERALL: 10/12 (83.3%)
```

**Critical caveat**: Images that were resized had 0% detection.

```
RESIZE IMPACT:
  With resize:    0% detection   ← Watermark destroyed
  Without resize: 100% detection ← Works fine
```

## The Resize Problem

When an image is resized, the DFT dimensions change, and our Legendre watermark is lost:

```
Original: 1944×2362
    ↓ Prime p = 1931
    ↓ Watermark embedded at frequencies [1..1931, 1..1931]

Resized (Instagram): 889×1080
    ↓ Prime p = 883
    ↓ Frequencies shifted/scaled
    ↓ Correlation fails completely
```

This is a **fundamental limitation** of frequency-domain watermarking with fixed prime.

## Filter Impact

Instagram-style filters have minimal impact because:
1. Our watermark is embedded in grayscale (V channel of HSV)
2. Filters primarily affect color/saturation
3. Brightness/contrast changes are survivable

| Filter | Effect | Impact on Watermark |
|--------|--------|---------------------|
| Clarendon | +Contrast, +Saturation | Minimal |
| Gingham | Desaturated, warm | Minimal |
| Juno | Warm highlights | Minimal |
| Valencia | Warm, faded, vignette | Minimal |
| X-Pro II | High contrast, vignette | Minimal |

## Attack Severity Ranking

```
1. RESIZE          ████████████████████ 100% destruction
2. WhatsApp JPEG   ████████████████     (Q65 + double compression)
3. Instagram JPEG  ██████████████       (Q72)
4. Twitter JPEG    ████████             (Q85)
5. Discord JPEG    ████                 (Q90)
6. Filters         ██                   (color only)
```

## Solutions

### Option 1: Accept Limitations

For NFTmarket's use case (PNG on IPFS):
- Original high-res PNG stored on IPFS → Watermark always survives
- Social media copies degrade → Acceptable for sharing
- Proof of ownership uses original, not social media copy

### Option 2: Multi-Scale Embedding

Embed watermark at multiple resolutions so one survives resize:

```
Original 2048×2048
    ├── Watermark at full scale (p=2039)
    ├── Watermark at 1024×1024 (p=1021) ← survives Instagram
    └── Watermark at 512×512 (p=509)   ← survives aggressive resize
```

Trade-off: Reduces capacity, increases complexity.

### Option 3: Neural Network Recovery

Train neural network to handle resized images:

```
Training augmentations:
  1. JPEG compression (Q50-95)
  2. Resize down (0.3× to 0.9×)    ← Add this
  3. Resize up (back to original)
  4. Detect watermark
```

The network learns to interpolate/reconstruct watermark from resized version.

### Option 4: Perceptual Hashing Fallback

For resized images, fall back to perceptual hash matching:

```
1. Try watermark detection → Fails (resized)
2. Compute perceptual hash (pHash, dHash)
3. Match against database of known watermarked images
4. Return associated wallet address
```

Not a watermark, but achieves the same goal of identifying ownership.

## Recommended Strategy for NFTmarket

```
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Image                                                │
│        ↓                                                     │
│   [1] Classical Legendre Detection (fast)                    │
│        ├── Success → Return wallet address                   │
│        └── Fail ↓                                            │
│                                                              │
│   [2] Neural-Enhanced Detection (JPEG recovery)              │
│        ├── Success → Return wallet address                   │
│        └── Fail ↓                                            │
│                                                              │
│   [3] Perceptual Hash Lookup (resize fallback)               │
│        ├── Match → Return wallet address                     │
│        └── No match → "Not found"                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `social_attack.py` | Social network attack simulator |
| `SOCIAL.md` | This file - findings and analysis |

## Test Command

```bash
# Test all platforms
python social_attack.py ~/path/to/images --sample 10 \
    --platforms instagram twitter whatsapp discord \
    --filters none clarendon valencia \
    --strength 8.0
```

## Key Takeaways

1. **JPEG is survivable** at strength 8+ (but may be visible on flat images)
2. **Resize is not survivable** with current implementation
3. **Filters are not a problem** — color transforms don't affect grayscale embedding
4. **Small images** (<1080px) have best survival on social media
5. **Neural enhancement** should include resize augmentation in training

---

Last updated: 2026-02-10
