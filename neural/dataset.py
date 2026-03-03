"""
Dataset for watermark recovery training.

Generates training samples on-the-fly:
1. Load original image
2. Embed watermark using Rust binary
3. Compute target DFT
4. Apply JPEG attack
5. Return (attacked_image, target_dft)
"""

import os
import io
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def generate_random_wallet() -> str:
    """Generate a random Ethereum wallet address."""
    return "0x" + "".join(random.choices("0123456789abcdef", k=40))


def compute_dft(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D DFT of grayscale image.

    Args:
        image: Grayscale image (H, W) with values in [0, 1]

    Returns:
        Complex DFT coefficients (H, W)
    """
    return np.fft.fft2(image)


def jpeg_attack(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression attack."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('L')


class WatermarkDataset(Dataset):
    """
    Dataset that generates watermarked samples on-the-fly.

    For each sample:
    1. Load image and resize to target size
    2. Embed watermark with random wallet address
    3. Compute DFT of watermarked image (target)
    4. Apply JPEG attack to watermarked image (input)
    """

    def __init__(
        self,
        image_dir: Path,
        embed_binary: Path,
        image_size: int = 512,
        p: int = 509,  # Prime for 512x512 images (512-2=510, largest prime <= 510)
        jpeg_quality_range: Tuple[int, int] = (50, 95),
        strength: float = 2.0,
        augment: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        self.image_dir = Path(image_dir)
        self.embed_binary = Path(embed_binary)
        self.image_size = image_size
        self.p = p
        self.jpeg_quality_range = jpeg_quality_range
        self.strength = strength
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.image_dir.glob(ext))
            self.image_paths.extend(self.image_dir.glob(ext.upper()))
        self.image_paths = sorted(set(self.image_paths))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_paths)} images")

        # Create cache dir if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        # Each image can generate many samples with different wallets/JPEG qualities
        return len(self.image_paths) * 10  # 10 variations per image

    def __getitem__(self, idx):
        # Map idx to image
        image_idx = idx % len(self.image_paths)
        image_path = self.image_paths[image_idx]

        # Load and resize image
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        # Generate random wallet
        wallet = generate_random_wallet()

        # Embed watermark using Rust binary
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.png"
            output_path = tmpdir / "watermarked.png"

            img.save(input_path)

            result = subprocess.run(
                [
                    str(self.embed_binary),
                    str(input_path),
                    str(output_path),
                    wallet,
                    "--strength", str(self.strength),
                    "--pilot",
                ],
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                # Fallback: return zeros if embedding fails
                print(f"Warning: embedding failed for {image_path}")
                return self._get_fallback()

            # Load watermarked image
            watermarked = Image.open(output_path).convert('L')

        # Compute target DFT (from clean watermarked image)
        wm_array = np.array(watermarked, dtype=np.float32) / 255.0
        target_dft = compute_dft(wm_array)

        # Extract watermark region [0:p, 0:p]
        target_wm = target_dft[:self.p, :self.p]
        target_real = np.real(target_wm).astype(np.float32)
        target_imag = np.imag(target_wm).astype(np.float32)

        # Normalize targets (important for training stability)
        # Use log-magnitude normalization for better gradient flow
        target_real = np.sign(target_real) * np.log1p(np.abs(target_real))
        target_imag = np.sign(target_imag) * np.log1p(np.abs(target_imag))

        # Apply JPEG attack
        quality = random.randint(*self.jpeg_quality_range)
        attacked = jpeg_attack(watermarked, quality)

        # Additional augmentations
        if self.augment:
            attacked = self._augment(attacked)

        # Convert to tensors
        attacked_tensor = TF.to_tensor(attacked)  # (1, H, W)
        target_tensor = torch.stack([
            torch.from_numpy(target_real),
            torch.from_numpy(target_imag),
        ])  # (2, p, p)

        return attacked_tensor, target_tensor, wallet

    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations simulating social network processing."""
        original_size = img.size

        # RESIZE: Simulates Instagram/WhatsApp/etc downscaling
        # This is the most destructive attack - train to recover from it
        if random.random() < 0.5:
            scale = random.uniform(0.3, 0.9)
            w, h = img.size
            # Downsample
            small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # Upsample back to original (simulates viewing at original resolution)
            img = small.resize(original_size, Image.LANCZOS)

        # Gaussian noise (camera/sensor noise)
        if random.random() < 0.3:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(2, 10), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Brightness (different lighting/screens)
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            img = TF.adjust_brightness(img, factor)

        # Contrast (display settings, filters)
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            img = TF.adjust_contrast(img, factor)

        # Gaussian blur (slight defocus, compression artifacts)
        if random.random() < 0.2:
            img = TF.gaussian_blur(img, kernel_size=3)

        # Sharpening (some platforms sharpen after resize)
        if random.random() < 0.2:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SHARPEN)

        return img

    def _get_fallback(self):
        """Return zero tensors as fallback."""
        attacked = torch.zeros(1, self.image_size, self.image_size)
        target = torch.zeros(2, self.p, self.p)
        return attacked, target, ""


class PrecomputedDataset(Dataset):
    """
    Dataset using pre-computed watermarked images.

    Faster than on-the-fly generation. Use generate_dataset.py to create.
    """

    def __init__(
        self,
        data_dir: Path,
        p: int = 509,
        jpeg_quality_range: Tuple[int, int] = (50, 95),
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.p = p
        self.jpeg_quality_range = jpeg_quality_range
        self.augment = augment

        # Find all watermarked images
        self.samples = []
        for wm_path in sorted(self.data_dir.glob("*.png")):
            # Each .png has a corresponding .txt with wallet address
            wallet_path = wm_path.with_suffix('.txt')
            if wallet_path.exists():
                self.samples.append((wm_path, wallet_path))

        print(f"Found {len(self.samples)} pre-computed samples")

    def __len__(self):
        return len(self.samples) * 5  # 5 JPEG variations per image

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        wm_path, wallet_path = self.samples[sample_idx]

        # Load watermarked image
        watermarked = Image.open(wm_path).convert('L')
        wallet = wallet_path.read_text().strip()

        # Compute target DFT
        wm_array = np.array(watermarked, dtype=np.float32) / 255.0
        target_dft = compute_dft(wm_array)
        target_wm = target_dft[:self.p, :self.p]
        target_real = np.sign(np.real(target_wm)) * np.log1p(np.abs(np.real(target_wm)))
        target_imag = np.sign(np.imag(target_wm)) * np.log1p(np.abs(np.imag(target_wm)))

        # Apply JPEG attack
        quality = random.randint(*self.jpeg_quality_range)
        attacked = jpeg_attack(watermarked, quality)

        if self.augment:
            attacked = self._augment(attacked)

        attacked_tensor = TF.to_tensor(attacked)
        target_tensor = torch.stack([
            torch.from_numpy(target_real.astype(np.float32)),
            torch.from_numpy(target_imag.astype(np.float32)),
        ])

        return attacked_tensor, target_tensor, wallet

    def _augment(self, img):
        """Apply random augmentations (same as WatermarkDataset)."""
        original_size = img.size

        # Resize attack (social network downscaling)
        if random.random() < 0.5:
            scale = random.uniform(0.3, 0.9)
            w, h = img.size
            small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img = small.resize(original_size, Image.LANCZOS)

        # Noise
        if random.random() < 0.3:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(2, 10), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Brightness
        if random.random() < 0.3:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))

        # Contrast
        if random.random() < 0.3:
            img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

        # Blur
        if random.random() < 0.2:
            img = TF.gaussian_blur(img, kernel_size=3)

        # Sharpen
        if random.random() < 0.2:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SHARPEN)

        return img


def test_dataset():
    """Quick test of dataset."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <image_dir>")
        return

    image_dir = Path(sys.argv[1])
    embed_binary = Path(__file__).parent.parent / "target/release/embed"

    if not embed_binary.exists():
        print(f"Error: {embed_binary} not found. Run 'cargo build --release' first.")
        return

    dataset = WatermarkDataset(
        image_dir=image_dir,
        embed_binary=embed_binary,
        image_size=512,
        p=509,
    )

    print(f"\nDataset size: {len(dataset)}")
    print("Loading sample...")

    attacked, target, wallet = dataset[0]
    print(f"  Attacked shape: {attacked.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Wallet: {wallet}")
    print("  ✓ Dataset OK")


if __name__ == "__main__":
    test_dataset()
