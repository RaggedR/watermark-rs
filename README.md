# watermark-rs

Blind spread-spectrum image watermarking in Rust, using Legendre sequences embedded in the DFT domain. Encodes arbitrary ASCII messages (e.g. wallet addresses, token IDs) as cyclic shifts of Legendre arrays in low-frequency DFT coefficients. Detection requires no original image.

Originally extracted from the NFTmarket project; NFTmarket calls this system via API.

## Tech Stack
- Rust (rustfft, image, clap, num-complex)
- Python (attack test suite, JPEG threshold tests, perceptual matching)
- U-Net (FRNet) in `neural/` for recovering DFT coefficients from JPEG-attacked images (architecture complete, awaiting training data)

## Setup
```bash
# Build CLI tools
cargo build --release
# Produces: target/release/embed, target/release/detect

# Embed a watermark
./target/release/embed --input image.png --output watermarked.png --message "hello"

# Detect a watermark
./target/release/detect --input watermarked.png

# Run attack test suite (requires Python)
python test_suite.py
```

## Results

Robustness of the classical (Rust) detector against common attacks
(full details and methodology in [ATTACKS.md](ATTACKS.md)):

| Attack | Threshold | Status |
|--------|-----------|--------|
| JPEG compression | Q ≥ 70 | ✅ Robust |
| Rotation (with pilot signal) | ±5.0° | ✅ Robust |
| Rotation (no pilot) | ±0.05° | ⚠️ Fragile |
| Scaling up | ≤ 110% | ✅ Robust |
| Scaling down | ≥ 99% | ⚠️ Fragile |
| Cropping | ≤ 10% | ✅ Robust |
| Shearing | ≥ 1% | ❌ Fails |
| Ring attack | ≥ ±0.06° | ❌ Fails |

The neural recovery pipeline in `neural/` targets the failing cases
(resize + recompression as applied by social platforms).

## License
MIT
