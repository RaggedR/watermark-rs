#!/usr/bin/env python3
"""
JPEG Quality Threshold Test

Tests watermark survival across different JPEG quality levels.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from PIL import Image
import io

# JPEG quality levels to test
QUALITY_LEVELS = [100, 98, 95, 92, 90, 88, 85, 82, 80, 75, 70, 60, 50]

def test_jpeg_quality(image_path: Path, embed_binary: Path, detect_binary: Path, message: str = "TEST123", strength: float = 2.0):
    """Test watermark survival at different JPEG quality levels."""

    results = []

    # Load and prepare image
    img = Image.open(image_path).convert('RGB')

    # Resize if too large
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save original
        original_path = tmpdir / "original.png"
        img.save(original_path)

        # Embed watermark
        watermarked_path = tmpdir / "watermarked.png"
        result = subprocess.run(
            [str(embed_binary), str(original_path), str(watermarked_path), message,
             "--strength", str(strength), "--pilot"],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"Embed failed: {result.stderr}")
            return []

        # Load watermarked image
        wm_img = Image.open(watermarked_path).convert('RGB')

        # Test each quality level
        for quality in QUALITY_LEVELS:
            # Apply JPEG compression
            buffer = io.BytesIO()
            wm_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            jpeg_img = Image.open(buffer).convert('RGB')

            # Save for detection
            attacked_path = tmpdir / f"jpeg_q{quality}.png"
            jpeg_img.save(attacked_path)

            # Detect
            result = subprocess.run(
                [str(detect_binary), str(attacked_path), "--pilot"],
                capture_output=True, text=True, timeout=60
            )

            detected = result.returncode == 0
            message_correct = message in result.stdout

            # Extract confidence
            confidence = 0.0
            for line in result.stdout.split('\n'):
                if 'confidence' in line.lower():
                    try:
                        confidence = float(line.split(':')[-1].strip())
                    except:
                        pass

            results.append({
                'quality': quality,
                'detected': detected,
                'message_correct': message_correct,
                'confidence': confidence
            })

            status = "✓" if detected else "✗"
            msg_status = "✓" if message_correct else "✗"
            print(f"  Q{quality:3d}: {status} detected, {msg_status} message, conf={confidence:.2f}")

    return results


def estimate_sequences(message: str, p: int) -> int:
    """Estimate how many Legendre sequences are needed for this message."""
    import math
    bits = len(message) * 7
    bits_per_shift = math.log2(p * p)
    return max(1, math.ceil(bits / bits_per_shift))


def largest_prime_for(width: int, height: int) -> int:
    """Find largest prime p such that p <= min(width, height) - 2."""
    max_size = min(width, height) - 2
    for n in range(max_size, 1, -1):
        if all(n % i != 0 for i in range(2, int(n**0.5) + 1)):
            return n
    return 2


def main():
    import argparse
    parser = argparse.ArgumentParser(description='JPEG Quality Threshold Test')
    parser.add_argument('image_dir', type=Path, help='Directory with test images')
    parser.add_argument('--sample', type=int, default=5, help='Number of images to test')
    parser.add_argument('--message', type=str, default='TEST123', help='Message to embed (use a wallet address for realistic testing)')
    parser.add_argument('--strength', type=float, default=2.0, help='Watermark strength')
    args = parser.parse_args()

    # Build
    print("Building release binaries...")
    subprocess.run(['cargo', 'build', '--release'], check=True, cwd=Path(__file__).parent)

    embed_binary = Path(__file__).parent / 'target/release/embed'
    detect_binary = Path(__file__).parent / 'target/release/detect'

    # Find images
    images = sorted(
        list(args.image_dir.glob('*.jpg')) + list(args.image_dir.glob('*.jpeg'))
        + list(args.image_dir.glob('*.png')) + list(args.image_dir.glob('*.webp'))
        + list(args.image_dir.glob('*.JPG')) + list(args.image_dir.glob('*.JPEG'))
        + list(args.image_dir.glob('*.PNG'))
    )
    images = list(set(images))[:args.sample]

    # Report message info
    print(f"\nMessage: \"{args.message}\" ({len(args.message)} chars, {len(args.message) * 7} bits)")
    print(f"Strength: {args.strength}")
    print(f"\nTesting {len(images)} images across {len(QUALITY_LEVELS)} JPEG quality levels")
    print("=" * 70)

    all_results = {q: {'detected': 0, 'correct': 0, 'total': 0, 'conf_sum': 0} for q in QUALITY_LEVELS}

    for img_path in images:
        # Report image dimensions and estimated sequences
        img_info = Image.open(img_path)
        w, h = img_info.size
        # Apply same resize logic as test function
        max_dim = 1024
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            w, h = int(w * ratio), int(h * ratio)
        p = largest_prime_for(w, h)
        n_seq = estimate_sequences(args.message, p)
        print(f"\n{img_path.name} ({w}x{h}, p={p}, ~{n_seq} sequences needed):")

        results = test_jpeg_quality(img_path, embed_binary, detect_binary,
                                    message=args.message, strength=args.strength)

        for r in results:
            q = r['quality']
            all_results[q]['total'] += 1
            if r['detected']:
                all_results[q]['detected'] += 1
            if r['message_correct']:
                all_results[q]['correct'] += 1
            all_results[q]['conf_sum'] += r['confidence']

    # Summary
    print("\n" + "=" * 70)
    print(f"JPEG QUALITY THRESHOLD SUMMARY")
    print(f"Message: \"{args.message}\" ({len(args.message)} chars, ~{len(args.message) * 7} bits)")
    print(f"Strength: {args.strength}")
    print("=" * 70)
    print(f"{'Quality':<10} {'Detection':>12} {'Message':>12} {'Avg Conf':>12}")
    print("-" * 70)

    for q in QUALITY_LEVELS:
        stats = all_results[q]
        if stats['total'] > 0:
            det_rate = stats['detected'] / stats['total'] * 100
            msg_rate = stats['correct'] / stats['total'] * 100
            avg_conf = stats['conf_sum'] / stats['total']

            status = "✓" if det_rate >= 80 else "⚠" if det_rate >= 50 else "✗"
            print(f"{status} Q{q:<7} {det_rate:>11.0f}% {msg_rate:>11.0f}% {avg_conf:>11.2f}")

    print("-" * 70)
    print("\nThreshold Analysis:")

    # Find the threshold
    for q in QUALITY_LEVELS:
        stats = all_results[q]
        if stats['total'] > 0:
            det_rate = stats['detected'] / stats['total'] * 100
            if det_rate >= 80:
                print(f"  ✓ Watermark survives at Q{q}+ (≥80% detection)")
            elif det_rate < 50:
                print(f"  ✗ Watermark fails below Q{q} (<50% detection)")
                break


if __name__ == '__main__':
    main()
