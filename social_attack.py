#!/usr/bin/env python3
"""
Social Network Attack Simulator

Simulates how different social platforms process uploaded images.
Tests watermark survival against real-world sharing scenarios.

Usage:
    python social_attack.py ~/Desktop/photos/nfts --sample 5
"""

import argparse
import io
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# =============================================================================
# PLATFORM CONFIGURATIONS
# =============================================================================

@dataclass
class PlatformConfig:
    """Configuration for a social media platform's image processing."""
    name: str
    max_dimension: int          # Max width or height
    jpeg_quality: int           # Typical JPEG quality
    strip_metadata: bool = True
    force_srgb: bool = True
    additional_compression: bool = False  # Some platforms recompress multiple times


PLATFORMS = {
    "instagram": PlatformConfig(
        name="Instagram",
        max_dimension=1080,
        jpeg_quality=72,
    ),
    "twitter": PlatformConfig(
        name="Twitter/X",
        max_dimension=4096,
        jpeg_quality=85,
    ),
    "facebook": PlatformConfig(
        name="Facebook",
        max_dimension=2048,
        jpeg_quality=80,
    ),
    "whatsapp": PlatformConfig(
        name="WhatsApp",
        max_dimension=1600,
        jpeg_quality=65,
        additional_compression=True,
    ),
    "tiktok": PlatformConfig(
        name="TikTok",
        max_dimension=1080,
        jpeg_quality=75,
    ),
    "discord": PlatformConfig(
        name="Discord",
        max_dimension=4000,
        jpeg_quality=90,  # Discord is more lenient
    ),
    "telegram": PlatformConfig(
        name="Telegram",
        max_dimension=2560,
        jpeg_quality=82,
    ),
}


# =============================================================================
# INSTAGRAM-STYLE FILTERS
# =============================================================================

def filter_clarendon(img: Image.Image) -> Image.Image:
    """Clarendon: Boost contrast and saturation, cool shadows."""
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Color(img).enhance(1.2)
    # Cool tint in shadows (approximate)
    arr = np.array(img, dtype=np.float32)
    # Boost blue in dark areas
    mask = (arr.mean(axis=2) < 100)[:, :, np.newaxis]
    arr[:, :, 2] = np.clip(arr[:, :, 2] + mask[:, :, 0] * 15, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def filter_gingham(img: Image.Image) -> Image.Image:
    """Gingham: Desaturated, slight warm tint, faded."""
    img = ImageEnhance.Color(img).enhance(0.7)  # Desaturate
    img = ImageEnhance.Contrast(img).enhance(0.9)  # Slight fade
    # Warm tint
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.05, 0, 255)  # Boost red
    return Image.fromarray(arr.astype(np.uint8))


def filter_juno(img: Image.Image) -> Image.Image:
    """Juno: Warm highlights, cool shadows, boosted saturation."""
    img = ImageEnhance.Color(img).enhance(1.3)
    arr = np.array(img, dtype=np.float32)
    # Warm highlights
    bright_mask = (arr.mean(axis=2) > 150)[:, :, np.newaxis]
    arr[:, :, 0] = np.clip(arr[:, :, 0] + bright_mask[:, :, 0] * 10, 0, 255)
    # Cool shadows
    dark_mask = (arr.mean(axis=2) < 80)[:, :, np.newaxis]
    arr[:, :, 2] = np.clip(arr[:, :, 2] + dark_mask[:, :, 0] * 10, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def filter_lark(img: Image.Image) -> Image.Image:
    """Lark: Desaturated reds, boosted blues and greens."""
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = arr[:, :, 0] * 0.85  # Reduce red
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)  # Boost green
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.1, 0, 255)  # Boost blue
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img


def filter_valencia(img: Image.Image) -> Image.Image:
    """Valencia: Warm, faded, slight vignette."""
    img = ImageEnhance.Color(img).enhance(0.9)
    img = ImageEnhance.Contrast(img).enhance(0.85)
    # Warm tint
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08, 0, 255)  # Boost red
    arr[:, :, 2] = arr[:, :, 2] * 0.95  # Reduce blue
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Simple vignette
    img = apply_vignette(img, strength=0.2)
    return img


def filter_xpro2(img: Image.Image) -> Image.Image:
    """X-Pro II: High contrast, warm vignette, golden tones."""
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Color(img).enhance(1.1)
    # Golden/sepia tint
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)
    arr[:, :, 2] = arr[:, :, 2] * 0.9
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = apply_vignette(img, strength=0.35)
    return img


def filter_nashville(img: Image.Image) -> Image.Image:
    """Nashville: Pink tint, high exposure, soft."""
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(0.9)
    arr = np.array(img, dtype=np.float32)
    # Pink/magenta tint
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.05, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def filter_rise(img: Image.Image) -> Image.Image:
    """Rise: Warm, soft, glowing."""
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(0.95)
    arr = np.array(img, dtype=np.float32)
    # Warm glow
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.03, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def filter_hudson(img: Image.Image) -> Image.Image:
    """Hudson: Cold, bluish, vintage."""
    img = ImageEnhance.Contrast(img).enhance(1.1)
    arr = np.array(img, dtype=np.float32)
    # Cool blue tint
    arr[:, :, 0] = arr[:, :, 0] * 0.92
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = apply_vignette(img, strength=0.25)
    return img


def filter_none(img: Image.Image) -> Image.Image:
    """No filter - just platform processing."""
    return img


def apply_vignette(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply a vignette effect."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create radial gradient
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = np.sqrt(center_x ** 2 + center_y ** 2)

    # Normalize and create vignette mask
    vignette = 1 - (radius / max_radius) * strength
    vignette = np.clip(vignette, 1 - strength, 1)

    # Apply to all channels
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * vignette

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


FILTERS = {
    "none": ("No Filter", filter_none),
    "clarendon": ("Clarendon", filter_clarendon),
    "gingham": ("Gingham", filter_gingham),
    "juno": ("Juno", filter_juno),
    "lark": ("Lark", filter_lark),
    "valencia": ("Valencia", filter_valencia),
    "xpro2": ("X-Pro II", filter_xpro2),
    "nashville": ("Nashville", filter_nashville),
    "rise": ("Rise", filter_rise),
    "hudson": ("Hudson", filter_hudson),
}


# =============================================================================
# SOCIAL NETWORK SIMULATION
# =============================================================================

def simulate_platform(img: Image.Image, platform: PlatformConfig, filter_fn: Optional[Callable] = None) -> Image.Image:
    """
    Simulate a social media platform's image processing pipeline.

    Args:
        img: Input image
        platform: Platform configuration
        filter_fn: Optional Instagram-style filter to apply

    Returns:
        Processed image as the platform would display it
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 1. Resize if exceeds max dimension
    w, h = img.size
    if max(w, h) > platform.max_dimension:
        scale = platform.max_dimension / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 2. Apply filter if specified
    if filter_fn:
        img = filter_fn(img)

    # 3. JPEG compression
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=platform.jpeg_quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB')

    # 4. Additional compression (some platforms compress multiple times)
    if platform.additional_compression:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=platform.jpeg_quality - 5)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')

    return img


# =============================================================================
# TESTING
# =============================================================================

def test_watermark_survival(
    image_path: Path,
    embed_binary: Path,
    detect_binary: Path,
    platform: PlatformConfig,
    filter_name: str = "none",
    message: str = "0x1234567890abcdef1234567890abcdef12345678",
    strength: float = 2.0,
) -> dict:
    """
    Test if watermark survives a social network pipeline.

    Returns dict with detection results.
    """
    filter_display, filter_fn = FILTERS[filter_name]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Load original
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        # Save for embedding
        input_path = tmpdir / "input.png"
        img.save(input_path)

        # Embed watermark
        watermarked_path = tmpdir / "watermarked.png"
        result = subprocess.run(
            [str(embed_binary), str(input_path), str(watermarked_path), message,
             "--strength", str(strength), "--pilot"],
            capture_output=True, timeout=60
        )

        if result.returncode != 0:
            return {"error": "Embedding failed"}

        # Load watermarked
        watermarked = Image.open(watermarked_path).convert('RGB')

        # Apply social network simulation
        attacked = simulate_platform(watermarked, platform, filter_fn)
        attacked_size = attacked.size

        # Save for detection
        attacked_path = tmpdir / "attacked.png"
        attacked.save(attacked_path)

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

        return {
            "platform": platform.name,
            "filter": filter_display,
            "original_size": original_size,
            "final_size": attacked_size,
            "resized": original_size != attacked_size,
            "jpeg_quality": platform.jpeg_quality,
            "detected": detected,
            "message_correct": message_correct,
            "confidence": confidence,
        }


def main():
    parser = argparse.ArgumentParser(description="Social Network Attack Simulator")
    parser.add_argument("image_dir", type=Path, help="Directory with test images")
    parser.add_argument("--sample", type=int, default=5, help="Number of images to test")
    parser.add_argument("--platforms", nargs="*", default=list(PLATFORMS.keys()), help="Platforms to test")
    parser.add_argument("--filters", nargs="*", default=["none", "clarendon", "valencia", "xpro2"], help="Filters to test")
    parser.add_argument("--strength", type=float, default=2.0, help="Watermark strength")
    args = parser.parse_args()

    # Find binaries
    script_dir = Path(__file__).parent
    embed_binary = script_dir / "target/release/embed"
    detect_binary = script_dir / "target/release/detect"

    if not embed_binary.exists():
        print(f"Error: {embed_binary} not found. Run 'cargo build --release'")
        return

    # Find images
    images = list(args.image_dir.glob("*.jpg")) + list(args.image_dir.glob("*.png"))
    images = images[:args.sample]

    if not images:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Testing {len(images)} images")
    print(f"Platforms: {', '.join(args.platforms)}")
    print(f"Filters: {', '.join(args.filters)}")
    print(f"Strength: {args.strength}")
    print("=" * 90)

    # Results table
    results = []

    for img_path in images:
        print(f"\n{img_path.name}:")

        for platform_key in args.platforms:
            platform = PLATFORMS[platform_key]

            for filter_name in args.filters:
                result = test_watermark_survival(
                    img_path, embed_binary, detect_binary,
                    platform, filter_name, strength=args.strength
                )
                results.append(result)

                if "error" in result:
                    status = "ERR"
                elif result["detected"] and result["message_correct"]:
                    status = "✓✓"
                elif result["detected"]:
                    status = "✓✗"
                else:
                    status = "✗✗"

                resize_note = f" (→{result['final_size'][0]}×{result['final_size'][1]})" if result.get("resized") else ""
                print(f"  {status} {platform.name:12} + {result['filter']:12} Q{platform.jpeg_quality}{resize_note:20} conf={result.get('confidence', 0):.2f}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY BY PLATFORM")
    print("=" * 90)

    for platform_key in args.platforms:
        platform = PLATFORMS[platform_key]
        platform_results = [r for r in results if r.get("platform") == platform.name]

        detected = sum(1 for r in platform_results if r.get("detected"))
        correct = sum(1 for r in platform_results if r.get("message_correct"))
        total = len(platform_results)

        if total > 0:
            det_rate = detected / total * 100
            cor_rate = correct / total * 100
            status = "✓" if det_rate >= 80 else "⚠" if det_rate >= 50 else "✗"
            print(f"{status} {platform.name:15} Detection: {det_rate:5.1f}%  Message: {cor_rate:5.1f}%  (n={total})")

    print("\n" + "=" * 90)
    print("SUMMARY BY FILTER")
    print("=" * 90)

    for filter_name in args.filters:
        filter_display = FILTERS[filter_name][0]
        filter_results = [r for r in results if r.get("filter") == filter_display]

        detected = sum(1 for r in filter_results if r.get("detected"))
        correct = sum(1 for r in filter_results if r.get("message_correct"))
        total = len(filter_results)

        if total > 0:
            det_rate = detected / total * 100
            cor_rate = correct / total * 100
            status = "✓" if det_rate >= 80 else "⚠" if det_rate >= 50 else "✗"
            print(f"{status} {filter_display:15} Detection: {det_rate:5.1f}%  Message: {cor_rate:5.1f}%  (n={total})")

    # Overall
    detected = sum(1 for r in results if r.get("detected"))
    correct = sum(1 for r in results if r.get("message_correct"))
    total = len(results)

    print("\n" + "=" * 90)
    print(f"OVERALL: Detection {detected}/{total} ({detected/total*100:.1f}%)  "
          f"Message {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 90)

    # Resize impact
    resized = [r for r in results if r.get("resized")]
    not_resized = [r for r in results if not r.get("resized")]

    if resized and not_resized:
        print("\nRESIZE IMPACT:")
        res_det = sum(1 for r in resized if r.get("detected")) / len(resized) * 100
        nores_det = sum(1 for r in not_resized if r.get("detected")) / len(not_resized) * 100
        print(f"  With resize:    {res_det:.1f}% detection")
        print(f"  Without resize: {nores_det:.1f}% detection")


if __name__ == "__main__":
    main()
