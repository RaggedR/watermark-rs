# watermark-rs

> Legendre sequence spread-spectrum watermarking in the DFT domain. Embeds and detects arbitrary ASCII messages (wallet addresses, token IDs) in images.

## Overview

This is a blind watermarking system — detection requires no access to the original image. Messages are encoded as cyclic shifts of Legendre arrays embedded in the low-frequency DFT coefficients.

Originally part of the NFTmarket project (`~/git/NFTmarket/`). NFTmarket now calls this system via API (`WATERMARK_API_URL`).

## Architecture

```
Embedding:
  Message → base-p² digits (shifts) → shifted Legendre arrays → add to DFT → IDFT → watermarked image

Detection:
  Image → DFT → correlate with Legendre sequences → peak-to-RMS → decode shifts → message
```

## Key Files

| Path | Purpose |
|------|---------|
| `src/lib.rs` | Library root, re-exports |
| `src/legendre.rs` | Legendre sequence generation (quadratic residues mod p) |
| `src/embedding.rs` | DFT-domain watermark embedding |
| `src/detection.rs` | Blind detection via Legendre correlation |
| `src/correlation.rs` | 2D correlation engine |
| `src/dft.rs` | FFT-based DFT/IDFT |
| `src/pilot.rs` | Pilot signal for rotation-invariant detection |
| `src/markers.rs` | Corner markers for perspective correction (print-camera) |
| `src/bin/embed.rs` | CLI: embed watermark into image |
| `src/bin/detect.rs` | CLI: detect watermark from image |

## Neural Watermark Recovery

The `neural/` directory contains a U-Net (FRNet) that recovers DFT coefficients from JPEG-attacked images, enabling the classical Legendre correlator to work on compressed images.

**Status**: Architecture complete, awaiting training data.

See `NEURAL.md` for setup instructions.

## Attack Testing

- `test_suite.py` — Full attack test battery (JPEG, rotation, scale, crop, noise, etc.)
- `jpeg_threshold_test.py` — JPEG quality threshold test (updated: accepts `--message` and `--strength`)
- `strength_vs_jpeg_test.py` — Strength vs JPEG quality matrix
- `attacks/attack_library.py` — Differentiable attack layers for training
- `ring_attack.py` — Novel concentric ring rotation attack

See `ATTACKS.md` for full attack documentation.

## Known Issues

- **All robustness tests were run with "TEST123" (7 chars).** Real wallet address messages (~45 chars) require ~18 Legendre sequences vs ~3. JPEG robustness for long messages is untested. Run `jpeg_threshold_test.py --message "0x...:42"` to get real numbers.

## Building

```bash
cargo build --release
# Binaries: target/release/embed, target/release/detect
```

## Related Projects

- **NFTmarket** (`~/git/NFTmarket/`) — Consumes this watermarking system via API
- **CLIP Explorer** (`~/git/images/clip-explorer/`) — Image embedding and aesthetic scoring (shares the same test image set)

## Reference Implementations

- `cpp-reference/` — Nick's original C++ implementation (detection + embedding)
- `cpp-original/` — Original C++ detection binary with CMake build
