#!/usr/bin/env python3
"""
Generate pre-computed training data.

Usage:
    python generate_data.py --image-dir /path/to/images --output-dir data/train --count 1000
"""

import argparse
import random
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image


def generate_random_wallet() -> str:
    """Generate a random Ethereum wallet address."""
    return "0x" + "".join(random.choices("0123456789abcdef", k=40))


def process_image(args):
    """Process a single image."""
    image_path, output_dir, embed_binary, image_size, strength, idx = args

    try:
        # Load and resize
        img = Image.open(image_path).convert('L')
        img = img.resize((image_size, image_size), Image.BICUBIC)

        # Generate wallet
        wallet = generate_random_wallet()

        # Save input
        input_path = output_dir / f"{idx:05d}_input.png"
        output_path = output_dir / f"{idx:05d}_watermarked.png"
        wallet_path = output_dir / f"{idx:05d}_watermarked.txt"

        img.save(input_path)

        # Embed watermark
        result = subprocess.run(
            [
                str(embed_binary),
                str(input_path),
                str(output_path),
                wallet,
                "--strength", str(strength),
                "--pilot",
            ],
            capture_output=True,
            timeout=60,
        )

        if result.returncode != 0:
            return None

        # Save wallet
        wallet_path.write_text(wallet)

        # Remove input (we only need watermarked)
        input_path.unlink()

        return idx

    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate pre-computed training data")
    parser.add_argument("--image-dir", type=Path, required=True, help="Source images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    parser.add_argument("--strength", type=float, default=2.0, help="Watermark strength")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Find embed binary
    embed_binary = Path(__file__).parent.parent / "target/release/embed"
    if not embed_binary.exists():
        print(f"Error: {embed_binary} not found. Run 'cargo build --release' first.")
        return

    # Find source images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_paths.extend(args.image_dir.glob(ext))
        image_paths.extend(args.image_dir.glob(ext.upper()))
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Found {len(image_paths)} source images")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples (cycle through images if count > len(images))
    tasks = []
    for i in range(args.count):
        image_path = image_paths[i % len(image_paths)]
        tasks.append((
            image_path,
            args.output_dir,
            embed_binary,
            args.image_size,
            args.strength,
            i,
        ))

    # Process in parallel
    print(f"Generating {args.count} samples...")
    successful = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                successful += 1

    print(f"\nGenerated {successful}/{args.count} samples")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
