#!/usr/bin/env python3
"""
Watermark Attack Test Suite

Tests watermark robustness against a battery of attacks.
Runs embed → attack → detect pipeline on a set of images.

Usage:
    python test_suite.py /path/to/images --sample 100 --output results.csv
"""

import argparse
import csv
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Attack definitions: (name, function, parameters)
# Each attack returns a PIL Image

def attack_identity(img: Image.Image, **kwargs) -> Image.Image:
    """No attack - baseline."""
    return img

def attack_jpeg(img: Image.Image, quality: int = 70) -> Image.Image:
    """JPEG compression."""
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def attack_rotation(img: Image.Image, angle: float = 1.0) -> Image.Image:
    """Rotation by angle degrees."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)

def attack_scale(img: Image.Image, factor: float = 0.9) -> Image.Image:
    """Scale by factor, then resize back."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    scaled = img.resize((new_w, new_h), Image.BICUBIC)
    return scaled.resize((w, h), Image.BICUBIC)

def attack_crop(img: Image.Image, ratio: float = 0.9) -> Image.Image:
    """Crop center and resize back."""
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)

def attack_gaussian_noise(img: Image.Image, std: float = 10.0) -> Image.Image:
    """Add Gaussian noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def attack_gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    """Gaussian blur."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def attack_brightness(img: Image.Image, factor: float = 1.2) -> Image.Image:
    """Brightness adjustment."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def attack_contrast(img: Image.Image, factor: float = 1.2) -> Image.Image:
    """Contrast adjustment."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def attack_saturation(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Saturation adjustment."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def attack_sharpening(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Sharpening."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def attack_median_filter(img: Image.Image, size: int = 3) -> Image.Image:
    """Median filter."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.MedianFilter(size=size))

def attack_flip_h(img: Image.Image, **kwargs) -> Image.Image:
    """Horizontal flip."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def attack_grayscale(img: Image.Image, **kwargs) -> Image.Image:
    """Convert to grayscale and back to RGB."""
    gray = img.convert('L')
    return gray.convert('RGB')

def attack_posterize(img: Image.Image, bits: int = 4) -> Image.Image:
    """Reduce color depth."""
    from PIL import ImageOps
    return ImageOps.posterize(img, bits)


# Attack configurations: (name, function, param_name, param_values)
ATTACKS = [
    # Baseline
    ("identity", attack_identity, None, [None]),

    # Compression
    ("jpeg_q90", attack_jpeg, "quality", [90]),
    ("jpeg_q70", attack_jpeg, "quality", [70]),
    ("jpeg_q50", attack_jpeg, "quality", [50]),

    # Geometric
    ("rotation_0.5", attack_rotation, "angle", [0.5]),
    ("rotation_1", attack_rotation, "angle", [1.0]),
    ("rotation_2", attack_rotation, "angle", [2.0]),
    ("rotation_5", attack_rotation, "angle", [5.0]),
    ("scale_95", attack_scale, "factor", [0.95]),
    ("scale_90", attack_scale, "factor", [0.90]),
    ("scale_110", attack_scale, "factor", [1.10]),
    ("crop_95", attack_crop, "ratio", [0.95]),
    ("crop_90", attack_crop, "ratio", [0.90]),
    ("flip_h", attack_flip_h, None, [None]),

    # Noise
    ("noise_5", attack_gaussian_noise, "std", [5.0]),
    ("noise_10", attack_gaussian_noise, "std", [10.0]),
    ("noise_20", attack_gaussian_noise, "std", [20.0]),

    # Filtering
    ("blur_1", attack_gaussian_blur, "radius", [1.0]),
    ("blur_2", attack_gaussian_blur, "radius", [2.0]),
    ("median_3", attack_median_filter, "size", [3]),
    ("sharpen_2", attack_sharpening, "factor", [2.0]),

    # Color
    ("brightness_120", attack_brightness, "factor", [1.2]),
    ("brightness_80", attack_brightness, "factor", [0.8]),
    ("contrast_120", attack_contrast, "factor", [1.2]),
    ("contrast_80", attack_contrast, "factor", [0.8]),
    ("saturation_130", attack_saturation, "factor", [1.3]),
    ("saturation_70", attack_saturation, "factor", [0.7]),
    ("grayscale", attack_grayscale, None, [None]),
    ("posterize_4", attack_posterize, "bits", [4]),
]


@dataclass
class TestResult:
    image_name: str
    attack_name: str
    detected: bool
    message_correct: bool
    confidence: float
    time_embed_ms: float
    time_detect_ms: float
    error: Optional[str] = None


def run_test(
    image_path: Path,
    attack_name: str,
    attack_fn,
    param_name: Optional[str],
    param_value,
    embed_binary: str,
    detect_binary: str,
    message: str = "TEST123",
    strength: float = 2.0,
) -> TestResult:
    """Run a single embed → attack → detect test."""

    image_name = image_path.name

    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Ensure reasonable size (resize if too large)
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.BICUBIC)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original
            original_path = Path(tmpdir) / "original.png"
            img.save(original_path)

            # Embed watermark
            watermarked_path = Path(tmpdir) / "watermarked.png"
            t0 = time.time()
            result = subprocess.run(
                [embed_binary, str(original_path), str(watermarked_path), message,
                 "--strength", str(strength), "--pilot"],
                capture_output=True,
                text=True,
                timeout=60
            )
            time_embed = (time.time() - t0) * 1000

            if result.returncode != 0:
                return TestResult(
                    image_name=image_name,
                    attack_name=attack_name,
                    detected=False,
                    message_correct=False,
                    confidence=0.0,
                    time_embed_ms=time_embed,
                    time_detect_ms=0.0,
                    error=f"Embed failed: {result.stderr[:200]}"
                )

            # Load watermarked image and apply attack
            wm_img = Image.open(watermarked_path).convert('RGB')

            if param_name:
                attacked_img = attack_fn(wm_img, **{param_name: param_value})
            else:
                attacked_img = attack_fn(wm_img)

            # Save attacked image
            attacked_path = Path(tmpdir) / "attacked.png"
            attacked_img.save(attacked_path)

            # Detect watermark
            t0 = time.time()
            result = subprocess.run(
                [detect_binary, str(attacked_path), "--pilot"],
                capture_output=True,
                text=True,
                timeout=60
            )
            time_detect = (time.time() - t0) * 1000

            # Parse detection result
            detected = result.returncode == 0
            message_correct = message in result.stdout

            # Extract confidence from output
            confidence = 0.0
            for line in result.stdout.split('\n'):
                if 'confidence' in line.lower():
                    try:
                        confidence = float(line.split(':')[-1].strip())
                    except:
                        pass

            return TestResult(
                image_name=image_name,
                attack_name=attack_name,
                detected=detected,
                message_correct=message_correct,
                confidence=confidence,
                time_embed_ms=time_embed,
                time_detect_ms=time_detect,
            )

    except Exception as e:
        return TestResult(
            image_name=image_name,
            attack_name=attack_name,
            detected=False,
            message_correct=False,
            confidence=0.0,
            time_embed_ms=0.0,
            time_detect_ms=0.0,
            error=str(e)[:200]
        )


def find_images(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Find all images in directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

    images = []
    for ext in extensions:
        images.extend(directory.glob(f'**/*{ext}'))
        images.extend(directory.glob(f'**/*{ext.upper()}'))

    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(description='Watermark Attack Test Suite')
    parser.add_argument('image_dir', type=Path, help='Directory containing test images')
    parser.add_argument('--sample', type=int, default=100, help='Number of images to test')
    parser.add_argument('--output', type=Path, default=Path('results.csv'), help='Output CSV file')
    parser.add_argument('--embed', type=str, default='cargo run --release --bin embed --', help='Embed command')
    parser.add_argument('--detect', type=str, default='cargo run --release --bin detect --', help='Detect command')
    parser.add_argument('--message', type=str, default='TEST123', help='Message to embed')
    parser.add_argument('--strength', type=float, default=2.0, help='Watermark strength')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--attacks', type=str, nargs='*', help='Specific attacks to run (default: all)')
    args = parser.parse_args()

    # Find images
    print(f"Scanning {args.image_dir} for images...")
    all_images = find_images(args.image_dir)
    print(f"Found {len(all_images)} images")

    if len(all_images) == 0:
        print("No images found!")
        return

    # Sample images
    if len(all_images) > args.sample:
        images = random.sample(all_images, args.sample)
        print(f"Sampled {args.sample} images for testing")
    else:
        images = all_images
        print(f"Using all {len(images)} images")

    # Build first to get binary path
    print("\nBuilding release binaries...")
    subprocess.run(['cargo', 'build', '--release'], check=True)

    # Get binary paths
    embed_binary = Path('target/release/embed')
    detect_binary = Path('target/release/detect')

    if not embed_binary.exists():
        print(f"Error: {embed_binary} not found")
        return

    # Filter attacks if specified
    attacks = ATTACKS
    if args.attacks:
        attacks = [(n, f, p, v) for n, f, p, v in ATTACKS if n in args.attacks]
        print(f"Running {len(attacks)} attacks: {[a[0] for a in attacks]}")

    # Calculate total tests
    total_tests = len(images) * len(attacks)
    print(f"\nRunning {total_tests} tests ({len(images)} images × {len(attacks)} attacks)")
    print("=" * 60)

    # Run tests
    results: List[TestResult] = []
    completed = 0

    start_time = time.time()

    for img_path in images:
        for attack_name, attack_fn, param_name, param_values in attacks:
            for param_value in param_values:
                result = run_test(
                    img_path,
                    attack_name,
                    attack_fn,
                    param_name,
                    param_value,
                    str(embed_binary),
                    str(detect_binary),
                    args.message,
                    args.strength,
                )
                results.append(result)
                completed += 1

                # Progress
                status = "✓" if result.detected else "✗"
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_tests - completed)
                print(f"[{completed}/{total_tests}] {status} {img_path.name[:20]:20} | {attack_name:20} | conf={result.confidence:.2f} | ETA: {eta:.0f}s")

    # Write results
    print(f"\nWriting results to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'attack', 'detected', 'message_correct', 'confidence',
                         'time_embed_ms', 'time_detect_ms', 'error'])
        for r in results:
            writer.writerow([r.image_name, r.attack_name, r.detected, r.message_correct,
                             r.confidence, r.time_embed_ms, r.time_detect_ms, r.error or ''])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Per-attack summary
    attack_stats = {}
    for r in results:
        if r.attack_name not in attack_stats:
            attack_stats[r.attack_name] = {'total': 0, 'detected': 0, 'correct': 0}
        attack_stats[r.attack_name]['total'] += 1
        if r.detected:
            attack_stats[r.attack_name]['detected'] += 1
        if r.message_correct:
            attack_stats[r.attack_name]['correct'] += 1

    print(f"\n{'Attack':<25} {'Detection Rate':>15} {'Correct Rate':>15}")
    print("-" * 60)
    for attack, stats in sorted(attack_stats.items()):
        det_rate = stats['detected'] / stats['total'] * 100
        cor_rate = stats['correct'] / stats['total'] * 100
        status = "✓" if det_rate >= 90 else "⚠" if det_rate >= 50 else "✗"
        print(f"{status} {attack:<23} {det_rate:>14.1f}% {cor_rate:>14.1f}%")

    # Overall
    total_detected = sum(1 for r in results if r.detected)
    total_correct = sum(1 for r in results if r.message_correct)
    print("-" * 60)
    print(f"{'OVERALL':<25} {total_detected/len(results)*100:>14.1f}% {total_correct/len(results)*100:>14.1f}%")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == '__main__':
    main()
