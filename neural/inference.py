#!/usr/bin/env python3
"""
Neural-enhanced watermark detection.

Usage:
    python inference.py <image_path> [--model weights/best.pth]

Outputs JSON with detection results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from model import FRNet, FRNetLite


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    p = config.get("p", 509)
    lite = config.get("lite", False)

    if lite:
        model = FRNetLite(p=p)
    else:
        model = FRNet(p=p)

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model, config


def recover_dft(model, image: Image.Image, device: torch.device) -> np.ndarray:
    """
    Use neural network to recover DFT watermark region.

    Args:
        model: Trained FRNet model
        image: Input image (possibly JPEG-attacked)
        device: Torch device

    Returns:
        Complex DFT array (p, p)
    """
    # Get model's expected size
    p = model.p

    # Resize image
    target_size = p + 3  # Slightly larger than p (matches training)
    img = image.convert('L').resize((512, 512), Image.BICUBIC)

    # Convert to tensor
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Run model
    with torch.no_grad():
        pred = model(img_tensor)  # (1, 2, p, p)

    # Convert back to numpy
    pred = pred.squeeze(0).cpu().numpy()  # (2, p, p)
    real = pred[0]
    imag = pred[1]

    # Undo log normalization
    real = np.sign(real) * (np.exp(np.abs(real)) - 1)
    imag = np.sign(imag) * (np.exp(np.abs(imag)) - 1)

    return real + 1j * imag


def legendre_symbol(a: int, p: int) -> int:
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def generate_legendre_array(p: int, k: int) -> np.ndarray:
    """Generate 2D Legendre sequence array."""
    arr = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            val = (i * i + k * j * j) % p
            arr[i, j] = legendre_symbol(val, p)
    return arr


def correlate_2d(signal: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """2D correlation using FFT."""
    signal_fft = np.fft.fft2(signal)
    pattern_fft = np.fft.fft2(pattern)
    correlation = np.fft.ifft2(signal_fft * np.conj(pattern_fft))
    return np.real(correlation)


def peak_to_rms(correlation: np.ndarray):
    """Calculate peak-to-RMS ratio."""
    max_idx = np.argmax(correlation)
    max_val = correlation.flat[max_idx]

    # RMS excluding peak
    mask = np.ones_like(correlation, dtype=bool)
    mask.flat[max_idx] = False
    rms = np.sqrt(np.mean(correlation[mask] ** 2))

    ratio = max_val / rms if rms > 1e-10 else 0.0
    return ratio, max_idx


def detect_from_dft(
    dft_coeffs: np.ndarray,
    p: int,
    threshold: float = 6.0,
    max_sequences: int = 100,
):
    """
    Run Legendre correlation detection on recovered DFT.

    Args:
        dft_coeffs: Complex DFT watermark region (p, p)
        p: Prime number
        threshold: Detection threshold
        max_sequences: Max Legendre sequences to try

    Returns:
        dict with detection results
    """
    # Use magnitude of DFT
    magnitude = np.abs(dft_coeffs)

    shifts = []
    min_confidence = float('inf')

    for k in range(1, max_sequences + 1):
        pattern = generate_legendre_array(p, k)
        correlation = correlate_2d(magnitude, pattern)

        peak_rms, max_idx = peak_to_rms(correlation)

        if peak_rms < threshold:
            break

        shifts.append(int(max_idx))
        min_confidence = min(min_confidence, peak_rms)

    if not shifts:
        return {
            "detected": False,
            "message": "",
            "confidence": 0.0,
            "sequences": 0,
        }

    # Decode shifts to ASCII
    message = decode_shifts(shifts, p * p)

    return {
        "detected": True,
        "message": message,
        "confidence": min_confidence,
        "sequences": len(shifts),
        "shifts": shifts,
    }


def decode_shifts(shifts: list, p_sq: int) -> str:
    """Convert shifts to ASCII message (matching Rust implementation)."""
    if not shifts:
        return ""

    # Reconstruct big integer from shifts (LSB-first, base p²)
    value = 0
    multiplier = 1
    for shift in shifts:
        value += shift * multiplier
        multiplier *= p_sq

    # Extract 7-bit ASCII characters
    chars = []
    while value > 0:
        c = value & 0x7F
        if 32 <= c < 127:
            chars.append(chr(c))
        value >>= 7

    return ''.join(reversed(chars))


def main():
    parser = argparse.ArgumentParser(description="Neural-enhanced watermark detection")
    parser.add_argument("image_path", type=Path, help="Image to analyze")
    parser.add_argument("--model", type=Path, default=Path("weights/best.pth"), help="Model checkpoint")
    parser.add_argument("--threshold", type=float, default=6.0, help="Detection threshold")
    parser.add_argument("--max-sequences", type=int, default=100, help="Max sequences to try")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check files exist
    if not args.image_path.exists():
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    if not args.model.exists():
        print(json.dumps({"error": f"Model not found: {args.model}"}))
        sys.exit(1)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if args.verbose:
        print(f"Device: {device}", file=sys.stderr)

    # Load model
    model, config = load_model(args.model, device)
    p = config.get("p", 509)

    if args.verbose:
        print(f"Model loaded, p={p}", file=sys.stderr)

    # Load image
    image = Image.open(args.image_path)

    if args.verbose:
        print(f"Image: {image.size}", file=sys.stderr)

    # Recover DFT
    dft_coeffs = recover_dft(model, image, device)

    if args.verbose:
        print(f"DFT recovered: {dft_coeffs.shape}", file=sys.stderr)

    # Detect watermark
    result = detect_from_dft(
        dft_coeffs,
        p,
        threshold=args.threshold,
        max_sequences=args.max_sequences,
    )

    # Output JSON
    print(json.dumps(result, indent=2))

    # Exit code based on detection
    sys.exit(0 if result["detected"] else 1)


if __name__ == "__main__":
    main()
