#!/usr/bin/env python3
"""
Test watermark strength vs JPEG survival trade-off.
"""

import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import io

STRENGTHS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
JPEG_QUALITIES = [98, 95, 92, 90, 85, 80]

def test_strength_vs_jpeg(image_path: Path, embed_binary: Path, detect_binary: Path, message: str = "TEST123"):
    """Test watermark strength vs JPEG quality."""

    img = Image.open(image_path).convert('RGB')

    # Resize if too large
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)

    results = {}

    for strength in STRENGTHS:
        results[strength] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save original
            original_path = tmpdir / "original.png"
            img.save(original_path)

            # Embed with this strength
            watermarked_path = tmpdir / "watermarked.png"
            result = subprocess.run(
                [str(embed_binary), str(original_path), str(watermarked_path), message,
                 "--strength", str(strength), "--pilot"],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                continue

            wm_img = Image.open(watermarked_path).convert('RGB')

            for quality in JPEG_QUALITIES:
                # Apply JPEG
                buffer = io.BytesIO()
                wm_img.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                jpeg_img = Image.open(buffer).convert('RGB')

                # Save and detect
                attacked_path = tmpdir / f"jpeg_q{quality}.png"
                jpeg_img.save(attacked_path)

                result = subprocess.run(
                    [str(detect_binary), str(attacked_path), "--pilot"],
                    capture_output=True, text=True, timeout=60
                )

                detected = result.returncode == 0
                message_correct = message in result.stdout

                confidence = 0.0
                for line in result.stdout.split('\n'):
                    if 'confidence' in line.lower():
                        try:
                            confidence = float(line.split(':')[-1].strip())
                        except:
                            pass

                results[strength][quality] = {
                    'detected': detected,
                    'correct': message_correct,
                    'confidence': confidence
                }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('--sample', type=int, default=3)
    args = parser.parse_args()

    print("Building release binaries...")
    subprocess.run(['cargo', 'build', '--release'], check=True, cwd=Path(__file__).parent)

    embed_binary = Path(__file__).parent / 'target/release/embed'
    detect_binary = Path(__file__).parent / 'target/release/detect'

    images = list(args.image_dir.glob('*.jpg')) + list(args.image_dir.glob('*.png'))
    images = images[:args.sample]

    print(f"\nTesting {len(images)} images")
    print(f"Strengths: {STRENGTHS}")
    print(f"JPEG qualities: {JPEG_QUALITIES}")
    print("=" * 80)

    # Aggregate results
    agg = {s: {q: {'det': 0, 'cor': 0, 'n': 0, 'conf': 0} for q in JPEG_QUALITIES} for s in STRENGTHS}

    for img_path in images:
        print(f"\n{img_path.name}")
        results = test_strength_vs_jpeg(img_path, embed_binary, detect_binary)

        for strength, qs in results.items():
            for quality, r in qs.items():
                agg[strength][quality]['n'] += 1
                if r['detected']:
                    agg[strength][quality]['det'] += 1
                if r['correct']:
                    agg[strength][quality]['cor'] += 1
                agg[strength][quality]['conf'] += r['confidence']

    # Print matrix
    print("\n" + "=" * 80)
    print("STRENGTH vs JPEG QUALITY MATRIX")
    print("Detection rates (%) - higher is better")
    print("=" * 80)

    # Header
    header = f"{'Strength':>10} |"
    for q in JPEG_QUALITIES:
        header += f" Q{q:>4} |"
    print(header)
    print("-" * len(header))

    # Rows
    for strength in STRENGTHS:
        row = f"{strength:>10} |"
        for quality in JPEG_QUALITIES:
            stats = agg[strength][quality]
            if stats['n'] > 0:
                rate = stats['det'] / stats['n'] * 100
                if rate >= 80:
                    row += f" {rate:>4.0f}%|"
                elif rate >= 50:
                    row += f" {rate:>4.0f}⚠|"
                else:
                    row += f" {rate:>4.0f}✗|"
            else:
                row += f"    - |"
        print(row)

    print("\n" + "=" * 80)
    print("AVERAGE CONFIDENCE")
    print("=" * 80)

    header = f"{'Strength':>10} |"
    for q in JPEG_QUALITIES:
        header += f" Q{q:>4} |"
    print(header)
    print("-" * len(header))

    for strength in STRENGTHS:
        row = f"{strength:>10} |"
        for quality in JPEG_QUALITIES:
            stats = agg[strength][quality]
            if stats['n'] > 0:
                conf = stats['conf'] / stats['n']
                row += f" {conf:>5.1f}|"
            else:
                row += f"    - |"
        print(row)


if __name__ == '__main__':
    main()
