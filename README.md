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

## License
MIT
