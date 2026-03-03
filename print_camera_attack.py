#!/usr/bin/env python3
"""
Print-Camera Attack Simulator

Simulates the pipeline when an image is printed on merchandise
(paper, t-shirt, mug) and then photographed.

Usage:
    python print_camera_attack.py ~/Desktop/photos/nfts --sample 3
"""

import argparse
import io
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import random
import math

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage


# =============================================================================
# PRINT SIMULATION
# =============================================================================

def simulate_halftoning(img: Image.Image, dpi: int = 150) -> Image.Image:
    """
    Simulate halftone printing (dot pattern).
    Lower DPI = more visible dots = more destruction.
    """
    arr = np.array(img.convert('L'), dtype=np.float32)
    h, w = arr.shape

    # Create halftone pattern
    # Simulate screen angle (typically 45° for grayscale)
    cell_size = max(2, 300 // dpi)  # Larger cells = lower quality print

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            block = arr[y:y+cell_size, x:x+cell_size]
            if block.size > 0:
                avg = np.mean(block)
                # Create dot based on average intensity
                dot_radius = (1 - avg/255) * cell_size / 2
                cy, cx = cell_size // 2, cell_size // 2
                for dy in range(block.shape[0]):
                    for dx in range(block.shape[1]):
                        dist = math.sqrt((dy - cy)**2 + (dx - cx)**2)
                        if dist < dot_radius:
                            arr[y+dy, x+dx] = 0
                        else:
                            arr[y+dy, x+dx] = 255

    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def simulate_cmyk_conversion(img: Image.Image) -> Image.Image:
    """Simulate RGB to CMYK and back (lossy color conversion)."""
    arr = np.array(img, dtype=np.float32) / 255.0

    # RGB to CMY
    c = 1 - arr[:, :, 0]
    m = 1 - arr[:, :, 1]
    y = 1 - arr[:, :, 2]

    # Generate K (black)
    k = np.minimum(np.minimum(c, m), y)

    # Adjust CMY based on K
    c = (c - k) / (1 - k + 1e-10)
    m = (m - k) / (1 - k + 1e-10)
    y = (y - k) / (1 - k + 1e-10)

    # Add some noise to simulate ink mixing
    c += np.random.normal(0, 0.02, c.shape)
    m += np.random.normal(0, 0.02, m.shape)
    y += np.random.normal(0, 0.02, y.shape)
    k += np.random.normal(0, 0.01, k.shape)

    # CMYK back to RGB
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    result = np.stack([r, g, b], axis=2)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def simulate_ink_bleeding(img: Image.Image, amount: float = 1.0) -> Image.Image:
    """Simulate ink spreading/bleeding on substrate."""
    arr = np.array(img, dtype=np.float32)

    # Gaussian blur to simulate bleeding
    sigma = amount * 0.5
    for c in range(3):
        arr[:, :, c] = ndimage.gaussian_filter(arr[:, :, c], sigma=sigma)

    return Image.fromarray(arr.astype(np.uint8))


def add_paper_texture(img: Image.Image, intensity: float = 0.1) -> Image.Image:
    """Add paper/canvas texture noise."""
    arr = np.array(img, dtype=np.float32)

    # Create texture pattern
    h, w = arr.shape[:2]
    texture = np.random.normal(0, intensity * 30, (h, w))

    # Apply texture to all channels
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + texture, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def add_fabric_texture(img: Image.Image, weave_size: int = 4) -> Image.Image:
    """Simulate fabric weave pattern (t-shirt)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create weave pattern
    y_coords, x_coords = np.ogrid[:h, :w]
    weave = np.sin(2 * np.pi * x_coords / weave_size) * np.sin(2 * np.pi * y_coords / weave_size)
    weave = weave * 15  # Intensity

    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + weave, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


# =============================================================================
# CAMERA SIMULATION
# =============================================================================

def apply_perspective(img: Image.Image, strength: float = 0.1) -> Image.Image:
    """Apply random perspective distortion (camera at angle)."""
    w, h = img.size

    # Random perspective coefficients
    # (a, b, c, d, e, f, g, h) for transform
    dx = w * strength * random.uniform(-1, 1)
    dy = h * strength * random.uniform(-1, 1)

    # Define source and destination points
    coeffs = find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],  # Source
        [
            (dx * random.uniform(0, 1), dy * random.uniform(0, 1)),
            (w - dx * random.uniform(0, 1), dy * random.uniform(0, 1)),
            (w - dx * random.uniform(0, 1), h - dy * random.uniform(0, 1)),
            (dx * random.uniform(0, 1), h - dy * random.uniform(0, 1)),
        ]  # Destination
    )

    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def find_perspective_coeffs(source, target):
    """Calculate perspective transform coefficients."""
    matrix = []
    for s, t in zip(source, target):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([s for s, t in zip(source, target) for s in s]).reshape(8)
    # Use lstsq for numerical stability (handles near-singular matrices)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return res


def apply_cylindrical_warp(img: Image.Image, curvature: float = 0.3) -> Image.Image:
    """Apply cylindrical warping (mug simulation)."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Create mapping
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Cylindrical distortion
    center_x = w / 2
    normalized_x = (xx - center_x) / center_x
    distortion = curvature * np.sin(np.pi * normalized_x)

    # Apply vertical displacement
    new_y = yy + distortion * h * 0.1
    new_y = np.clip(new_y, 0, h - 1).astype(int)

    # Map pixels
    result = arr[new_y, xx]

    return Image.fromarray(result)


def simulate_lighting(img: Image.Image, variation: float = 0.3) -> Image.Image:
    """Simulate uneven lighting (spotlight, shadows)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create gradient lighting
    y_coords, x_coords = np.ogrid[:h, :w]
    center_x = w * random.uniform(0.3, 0.7)
    center_y = h * random.uniform(0.3, 0.7)

    # Radial falloff
    dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(w**2 + h**2) / 2
    lighting = 1 - variation * (dist / max_dist)

    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * lighting, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def simulate_camera_noise(img: Image.Image, iso: int = 400) -> Image.Image:
    """Simulate camera sensor noise based on ISO."""
    arr = np.array(img, dtype=np.float32)

    # Noise increases with ISO
    noise_std = iso / 100.0
    noise = np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def simulate_motion_blur(img: Image.Image, amount: int = 2) -> Image.Image:
    """Simulate slight motion blur."""
    kernel_size = amount * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = ndimage.convolve(arr[:, :, c], kernel)

    return Image.fromarray(arr.astype(np.uint8))


# =============================================================================
# PRINT-CAMERA SCENARIOS
# =============================================================================

@dataclass
class PrintScenario:
    name: str
    substrate: str  # paper, fabric, ceramic
    halftone_dpi: int
    has_texture: bool
    perspective_strength: float
    is_curved: bool
    lighting_variation: float
    camera_iso: int
    jpeg_quality: int


SCENARIOS = {
    "paper_scan": PrintScenario(
        name="Paper (Scanner)",
        substrate="paper",
        halftone_dpi=300,
        has_texture=False,
        perspective_strength=0.0,
        is_curved=False,
        lighting_variation=0.05,
        camera_iso=100,
        jpeg_quality=95,
    ),
    "paper_photo": PrintScenario(
        name="Paper (Photo)",
        substrate="paper",
        halftone_dpi=150,
        has_texture=True,
        perspective_strength=0.1,
        is_curved=False,
        lighting_variation=0.2,
        camera_iso=400,
        jpeg_quality=85,
    ),
    "tshirt_flat": PrintScenario(
        name="T-Shirt (Flat)",
        substrate="fabric",
        halftone_dpi=100,
        has_texture=True,
        perspective_strength=0.1,
        is_curved=False,
        lighting_variation=0.25,
        camera_iso=400,
        jpeg_quality=80,
    ),
    "tshirt_worn": PrintScenario(
        name="T-Shirt (Worn)",
        substrate="fabric",
        halftone_dpi=100,
        has_texture=True,
        perspective_strength=0.2,
        is_curved=False,
        lighting_variation=0.35,
        camera_iso=800,
        jpeg_quality=75,
    ),
    "mug": PrintScenario(
        name="Mug (Curved)",
        substrate="ceramic",
        halftone_dpi=150,
        has_texture=False,
        perspective_strength=0.05,
        is_curved=True,
        lighting_variation=0.3,
        camera_iso=400,
        jpeg_quality=80,
    ),
    "poster": PrintScenario(
        name="Poster (Large)",
        substrate="paper",
        halftone_dpi=200,
        has_texture=True,
        perspective_strength=0.15,
        is_curved=False,
        lighting_variation=0.2,
        camera_iso=200,
        jpeg_quality=85,
    ),
}


def simulate_print_camera(img: Image.Image, scenario: PrintScenario) -> Image.Image:
    """Apply full print-camera pipeline for a given scenario."""

    # 1. Color conversion (print)
    img = simulate_cmyk_conversion(img)

    # 2. Halftoning
    if scenario.halftone_dpi < 300:  # Only visible at lower DPI
        img = simulate_halftoning(img, scenario.halftone_dpi)

    # 3. Ink bleeding
    img = simulate_ink_bleeding(img, amount=1.5)

    # 4. Substrate texture
    if scenario.has_texture:
        if scenario.substrate == "fabric":
            img = add_fabric_texture(img, weave_size=4)
        else:
            img = add_paper_texture(img, intensity=0.1)

    # 5. Geometric distortion
    if scenario.is_curved:
        img = apply_cylindrical_warp(img, curvature=0.3)
    if scenario.perspective_strength > 0:
        img = apply_perspective(img, scenario.perspective_strength)

    # 6. Lighting
    if scenario.lighting_variation > 0:
        img = simulate_lighting(img, scenario.lighting_variation)

    # 7. Camera noise
    img = simulate_camera_noise(img, scenario.camera_iso)

    # 8. Slight motion blur (optional)
    if scenario.camera_iso >= 800:
        img = simulate_motion_blur(img, amount=1)

    # 9. JPEG compression (camera save)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=scenario.jpeg_quality)
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
    scenario: PrintScenario,
    message: str = "0x1234567890abcdef1234567890abcdef12345678",
    strength: float = 8.0,
    use_markers: bool = True,
) -> dict:
    """Test if watermark survives print-camera pipeline."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Load original
        img = Image.open(image_path).convert('RGB')

        # Save for embedding
        input_path = tmpdir / "input.png"
        img.save(input_path)

        # Embed watermark (with markers for print-camera)
        watermarked_path = tmpdir / "watermarked.png"
        embed_args = [
            str(embed_binary), str(input_path), str(watermarked_path), message,
            "--strength", str(strength), "--pilot"
        ]

        result = subprocess.run(embed_args, capture_output=True, timeout=60)
        if result.returncode != 0:
            return {"error": "Embedding failed"}

        # Load watermarked
        watermarked = Image.open(watermarked_path).convert('RGB')

        # Apply print-camera simulation
        attacked = simulate_print_camera(watermarked, scenario)

        # Save for detection
        attacked_path = tmpdir / "attacked.png"
        attacked.save(attacked_path)

        # Detect
        detect_args = [str(detect_binary), str(attacked_path), "--pilot"]

        result = subprocess.run(detect_args, capture_output=True, text=True, timeout=60)

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
            "scenario": scenario.name,
            "substrate": scenario.substrate,
            "halftone_dpi": scenario.halftone_dpi,
            "detected": detected,
            "message_correct": message_correct,
            "confidence": confidence,
        }


def main():
    parser = argparse.ArgumentParser(description="Print-Camera Attack Simulator")
    parser.add_argument("image_dir", type=Path, help="Directory with test images")
    parser.add_argument("--sample", type=int, default=3, help="Number of images to test")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()), help="Scenarios to test")
    parser.add_argument("--strength", type=float, default=8.0, help="Watermark strength")
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
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Strength: {args.strength}")
    print("=" * 80)

    results = []

    for img_path in images:
        print(f"\n{img_path.name}:")

        for scenario_key in args.scenarios:
            scenario = SCENARIOS[scenario_key]
            result = test_watermark_survival(
                img_path, embed_binary, detect_binary,
                scenario, strength=args.strength
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

            print(f"  {status} {scenario.name:20} DPI={scenario.halftone_dpi:3}  conf={result.get('confidence', 0):.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY SCENARIO")
    print("=" * 80)

    for scenario_key in args.scenarios:
        scenario = SCENARIOS[scenario_key]
        scenario_results = [r for r in results if r.get("scenario") == scenario.name]

        detected = sum(1 for r in scenario_results if r.get("detected"))
        correct = sum(1 for r in scenario_results if r.get("message_correct"))
        total = len(scenario_results)

        if total > 0:
            det_rate = detected / total * 100
            cor_rate = correct / total * 100
            status = "✓" if det_rate >= 50 else "⚠" if det_rate >= 20 else "✗"
            print(f"{status} {scenario.name:20} Detection: {det_rate:5.1f}%  Message: {cor_rate:5.1f}%")

    # Overall
    detected = sum(1 for r in results if r.get("detected"))
    correct = sum(1 for r in results if r.get("message_correct"))
    total = len(results)

    print("\n" + "=" * 80)
    print(f"OVERALL: Detection {detected}/{total} ({detected/total*100:.1f}%)  "
          f"Message {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
