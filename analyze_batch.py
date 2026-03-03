#!/usr/bin/env python3
"""
Batch analyze images for watermark strength recommendations.
"""

import subprocess
import json
from pathlib import Path
import sys

def analyze_image(binary: Path, image_path: Path) -> dict:
    """Analyze a single image and return parsed JSON."""
    result = subprocess.run(
        [str(binary), str(image_path), "--json"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_batch.py <image_dir>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    binary = Path(__file__).parent / "target/release/analyze"

    if not binary.exists():
        print("Error: Run 'cargo build --release' first")
        sys.exit(1)

    # Find images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    images = sorted(images)[:20]  # Limit to 20 for display

    print(f"Analyzing {len(images)} images from {image_dir}\n")
    print("=" * 100)
    print(f"{'Image':<20} {'Texture':<12} {'Flat %':<10} {'Max Invis':<10} {'Recommend':<10} {'JPEG Q90':<12} {'JPEG Q80':<12}")
    print("=" * 100)

    results = []
    for img in images:
        data = analyze_image(binary, img)
        if data and 'strength' in data:
            s = data['strength']
            texture = s['texture_score']
            flat = s['flat_region_percent']
            max_inv = s['max_invisible']
            rec = s['recommended']
            q90 = s['jpeg_q90_strength']
            q80 = s['jpeg_q80_strength']

            # Can survive JPEG?
            q90_ok = "✓" if q90 <= max_inv else "✗"
            q80_ok = "✓" if q80 <= max_inv else "✗"

            print(f"{img.name:<20} {texture:<12.3f} {flat:<10.1f} {max_inv:<10.1f} {rec:<10.1f} {q90:<6.1f} {q90_ok:<5} {q80:<6.1f} {q80_ok}")

            results.append({
                'name': img.name,
                'texture': texture,
                'flat': flat,
                'max_invisible': max_inv,
                'recommended': rec,
                'can_q90': q90 <= max_inv,
                'can_q80': q80 <= max_inv,
            })

    print("=" * 100)

    # Summary stats
    if results:
        avg_texture = sum(r['texture'] for r in results) / len(results)
        avg_flat = sum(r['flat'] for r in results) / len(results)
        avg_max = sum(r['max_invisible'] for r in results) / len(results)
        can_q90 = sum(1 for r in results if r['can_q90'])
        can_q80 = sum(1 for r in results if r['can_q80'])

        print(f"\nSUMMARY ({len(results)} images)")
        print(f"  Average texture score: {avg_texture:.3f}")
        print(f"  Average flat regions:  {avg_flat:.1f}%")
        print(f"  Average max invisible: {avg_max:.1f}")
        print(f"  Can survive JPEG Q90:  {can_q90}/{len(results)} ({can_q90/len(results)*100:.0f}%)")
        print(f"  Can survive JPEG Q80:  {can_q80}/{len(results)} ({can_q80/len(results)*100:.0f}%)")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        if avg_texture < 0.05:
            print("  ⚠️  These images are mostly flat (digital art/illustrations)")
            print("     → Use strength 2.0 and store as PNG only")
            print("     → JPEG-robust watermarking would require visible artifacts")
        elif avg_texture < 0.15:
            print("  ○ These images have moderate texture")
            print("     → Can use strength ~3-4 for some JPEG robustness")
        else:
            print("  ✓ These images are highly textured (photos)")
            print("     → Can use strength 5-8 for full JPEG robustness")


if __name__ == '__main__':
    main()
