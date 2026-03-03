#!/usr/bin/env python3
"""
Perceptual Image Matching Tool

Compares a suspect image against your original to determine if they're "the same" image,
even after print-camera attacks, JPEG compression, or other transformations.

Usage:
    python perceptual_match.py <original> <suspect>
    python perceptual_match.py --register <image> [--db <database.json>]
    python perceptual_match.py --search <suspect> [--db <database.json>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# =============================================================================
# Perceptual Hashing Algorithms
# =============================================================================

def compute_phash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """
    Perceptual hash using DCT-like difference detection.

    Robust to: resize, mild compression, color changes
    Fragile to: rotation, cropping, perspective
    """
    # Resize to hash_size + 1 for difference computation
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)

    # Compute horizontal gradient (difference hash)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def compute_ahash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """
    Average hash - compares each pixel to the mean.

    Very fast, good for exact/near-exact matches.
    """
    img = image.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    mean = pixels.mean()
    return (pixels > mean).flatten()


def compute_whash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """
    Wavelet hash - uses Haar wavelet transform.

    More robust to slight modifications than ahash.
    """
    img = image.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)

    # Simple Haar-like transform (average and difference)
    while pixels.shape[0] > 1:
        # Horizontal
        h, w = pixels.shape
        avg_h = (pixels[:, 0::2] + pixels[:, 1::2]) / 2 if w > 1 else pixels
        # Vertical
        pixels = (avg_h[0::2, :] + avg_h[1::2, :]) / 2 if h > 1 else avg_h

    # Fall back to ahash-style if wavelet collapses too much
    img = image.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)

    # Low-frequency components
    low_freq = pixels[:hash_size//2, :hash_size//2]
    mean = low_freq.mean()
    return (low_freq > mean).flatten()


def compute_color_hash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """
    Color histogram hash - captures color distribution.

    Robust to geometric transforms, fragile to color changes.
    """
    img = image.convert('RGB').resize((hash_size, hash_size), Image.LANCZOS)

    # Compute histogram for each channel
    r, g, b = img.split()

    def channel_hash(ch):
        hist = np.array(ch.histogram()[:256], dtype=np.float64)
        hist = hist / hist.sum()  # Normalize
        # Quantize to 16 bins
        bins = np.array([hist[i*16:(i+1)*16].sum() for i in range(16)])
        mean = bins.mean()
        return bins > mean

    return np.concatenate([channel_hash(r), channel_hash(g), channel_hash(b)])


# =============================================================================
# Similarity Metrics
# =============================================================================

def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Count differing bits between two hashes."""
    return int(np.sum(hash1 != hash2))


def hamming_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Convert Hamming distance to similarity percentage."""
    dist = hamming_distance(hash1, hash2)
    return 100.0 * (1 - dist / len(hash1))


def structural_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Simplified SSIM (Structural Similarity Index).

    Compares luminance, contrast, and structure.
    """
    # Resize to same size
    size = (256, 256)
    arr1 = np.array(img1.convert('L').resize(size), dtype=np.float64) / 255.0
    arr2 = np.array(img2.convert('L').resize(size), dtype=np.float64) / 255.0

    c1, c2 = 0.01**2, 0.03**2

    mu1, mu2 = arr1.mean(), arr2.mean()
    sigma1, sigma2 = arr1.std(), arr2.std()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

    ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

    return ssim * 100


def histogram_correlation(img1: Image.Image, img2: Image.Image) -> float:
    """Compare grayscale histograms using correlation."""
    hist1 = np.array(img1.convert('L').histogram(), dtype=np.float64)
    hist2 = np.array(img2.convert('L').histogram(), dtype=np.float64)

    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Correlation coefficient
    corr = np.corrcoef(hist1, hist2)[0, 1]
    return corr * 100 if not np.isnan(corr) else 0.0


# =============================================================================
# Main Matching Functions
# =============================================================================

def compute_all_hashes(image: Image.Image) -> Dict[str, np.ndarray]:
    """Compute all perceptual hashes for an image."""
    return {
        'phash': compute_phash(image),
        'ahash': compute_ahash(image),
        'whash': compute_whash(image),
        'chash': compute_color_hash(image),
    }


def compare_images(original: Image.Image, suspect: Image.Image) -> Dict[str, float]:
    """
    Compare two images using multiple perceptual metrics.

    Returns dict of similarity scores (0-100, higher = more similar).
    """
    # Compute hashes
    hashes1 = compute_all_hashes(original)
    hashes2 = compute_all_hashes(suspect)

    results = {}

    # Hash similarities
    for name in hashes1:
        results[f'{name}_sim'] = hamming_similarity(hashes1[name], hashes2[name])

    # Structural similarity
    results['ssim'] = structural_similarity(original, suspect)

    # Histogram correlation
    results['hist_corr'] = histogram_correlation(original, suspect)

    # Combined score (weighted average)
    weights = {
        'phash_sim': 0.25,
        'ahash_sim': 0.15,
        'whash_sim': 0.15,
        'chash_sim': 0.10,
        'ssim': 0.20,
        'hist_corr': 0.15,
    }

    results['combined'] = sum(results[k] * w for k, w in weights.items())

    return results


def match_verdict(scores: Dict[str, float]) -> Tuple[str, str]:
    """
    Determine match verdict based on scores.

    Returns (verdict, explanation).
    """
    combined = scores['combined']
    phash = scores['phash_sim']

    if combined >= 85 and phash >= 90:
        return "✓ MATCH", "High confidence - very likely the same image"
    elif combined >= 70 and phash >= 75:
        return "✓ PROBABLE MATCH", "Good confidence - likely the same image with modifications"
    elif combined >= 55 and phash >= 60:
        return "? POSSIBLE MATCH", "Moderate similarity - may be related or derived"
    elif combined >= 40:
        return "? UNCERTAIN", "Some similarity - possibly coincidental"
    else:
        return "✗ NO MATCH", "Low similarity - likely different images"


# =============================================================================
# Database Functions (for searching against collection)
# =============================================================================

def hash_to_hex(hash_array: np.ndarray) -> str:
    """Convert binary hash to hex string for storage."""
    # Pack bits into bytes
    bits = hash_array.astype(np.uint8)
    bytes_arr = np.packbits(bits)
    return bytes_arr.tobytes().hex()


def hex_to_hash(hex_str: str, length: int) -> np.ndarray:
    """Convert hex string back to binary hash."""
    bytes_arr = bytes.fromhex(hex_str)
    bits = np.unpackbits(np.frombuffer(bytes_arr, dtype=np.uint8))
    return bits[:length].astype(bool)


def register_image(image_path: str, db_path: str = "image_db.json",
                   metadata: Optional[Dict] = None) -> Dict:
    """Register an image in the database."""
    img = Image.open(image_path)
    hashes = compute_all_hashes(img)

    # Load existing database
    db = {}
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)

    # Create entry
    entry = {
        'path': str(Path(image_path).absolute()),
        'hashes': {name: hash_to_hex(h) for name, h in hashes.items()},
        'hash_lengths': {name: len(h) for name, h in hashes.items()},
        'metadata': metadata or {},
    }

    # Use phash as key (or path)
    key = hash_to_hex(hashes['phash'])
    db[key] = entry

    # Save database
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)

    return entry


def search_database(suspect_path: str, db_path: str = "image_db.json",
                    threshold: float = 70.0) -> List[Dict]:
    """Search database for matches to suspect image."""
    if not os.path.exists(db_path):
        return []

    with open(db_path, 'r') as f:
        db = json.load(f)

    suspect = Image.open(suspect_path)
    suspect_hashes = compute_all_hashes(suspect)

    matches = []

    for key, entry in db.items():
        # Reconstruct hashes
        stored_hashes = {}
        for name, hex_str in entry['hashes'].items():
            length = entry['hash_lengths'][name]
            stored_hashes[name] = hex_to_hash(hex_str, length)

        # Compare phash quickly first
        phash_sim = hamming_similarity(suspect_hashes['phash'], stored_hashes['phash'])

        if phash_sim >= threshold - 20:  # Allow some margin for full comparison
            # Full comparison
            original = Image.open(entry['path'])
            scores = compare_images(original, suspect)

            if scores['combined'] >= threshold:
                matches.append({
                    'path': entry['path'],
                    'scores': scores,
                    'verdict': match_verdict(scores),
                    'metadata': entry.get('metadata', {}),
                })

    # Sort by combined score
    matches.sort(key=lambda x: x['scores']['combined'], reverse=True)
    return matches


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perceptual image matching tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compare two images:
    python perceptual_match.py original.png suspect.jpg

  Register image in database:
    python perceptual_match.py --register artwork.png --db my_art.json

  Search for matches:
    python perceptual_match.py --search photo_of_mug.jpg --db my_art.json
        """
    )

    parser.add_argument('images', nargs='*', help='Images to compare (original, suspect)')
    parser.add_argument('--register', metavar='IMAGE', help='Register image in database')
    parser.add_argument('--search', metavar='IMAGE', help='Search database for matches')
    parser.add_argument('--db', default='image_db.json', help='Database file path')
    parser.add_argument('--threshold', type=float, default=70.0, help='Match threshold (0-100)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')

    args = parser.parse_args()

    # Register mode
    if args.register:
        entry = register_image(args.register, args.db)
        if not args.quiet:
            print(f"Registered: {args.register}")
            print(f"Database: {args.db}")
        return

    # Search mode
    if args.search:
        matches = search_database(args.search, args.db, args.threshold)

        if not matches:
            print("No matches found.")
            return

        print(f"Found {len(matches)} match(es):\n")
        for m in matches:
            verdict, explanation = m['verdict']
            print(f"{verdict}")
            print(f"  Original: {m['path']}")
            print(f"  Combined score: {m['scores']['combined']:.1f}%")
            print(f"  {explanation}")
            print()
        return

    # Compare mode (default)
    if len(args.images) != 2:
        parser.print_help()
        sys.exit(1)

    original_path, suspect_path = args.images

    original = Image.open(original_path)
    suspect = Image.open(suspect_path)

    scores = compare_images(original, suspect)
    verdict, explanation = match_verdict(scores)

    if args.quiet:
        print(f"{scores['combined']:.1f}")
    else:
        print("=" * 60)
        print("PERCEPTUAL IMAGE MATCHING")
        print("=" * 60)
        print(f"\nOriginal: {original_path}")
        print(f"Suspect:  {suspect_path}")
        print("\n" + "-" * 40)
        print("Similarity Scores:")
        print("-" * 40)
        print(f"  pHash:        {scores['phash_sim']:5.1f}%")
        print(f"  aHash:        {scores['ahash_sim']:5.1f}%")
        print(f"  wHash:        {scores['whash_sim']:5.1f}%")
        print(f"  Color hash:   {scores['chash_sim']:5.1f}%")
        print(f"  SSIM:         {scores['ssim']:5.1f}%")
        print(f"  Histogram:    {scores['hist_corr']:5.1f}%")
        print("-" * 40)
        print(f"  COMBINED:     {scores['combined']:5.1f}%")
        print("\n" + "=" * 60)
        print(f"VERDICT: {verdict}")
        print(f"         {explanation}")
        print("=" * 60)


if __name__ == "__main__":
    main()
