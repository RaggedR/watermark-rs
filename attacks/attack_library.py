"""
Comprehensive Attack Library for Watermark Robustness Training

This library provides differentiable (where possible) implementations of
common image attacks. Use these for:
1. Training neural watermarking networks
2. Testing traditional watermark robustness
3. Augmenting datasets for robust model training

All attacks accept tensors in [0, 1] range, shape (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


# =============================================================================
# GEOMETRIC ATTACKS
# =============================================================================

class RandomRotation(nn.Module):
    """Random rotation with bilinear interpolation (differentiable)."""

    def __init__(self, max_angle: float = 30.0):
        super().__init__()
        self.max_angle = max_angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = (torch.rand(1) * 2 - 1) * self.max_angle
        angle_rad = angle * math.pi / 180

        # Create rotation matrix
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(theta.expand(x.size(0), -1, -1), x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class RandomScale(nn.Module):
    """Random scaling (zoom in/out)."""

    def __init__(self, min_scale: float = 0.7, max_scale: float = 1.3):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale

        theta = torch.tensor([
            [1/scale, 0, 0],
            [0, 1/scale, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(theta.expand(x.size(0), -1, -1), x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class RandomCrop(nn.Module):
    """Random crop and resize back to original size."""

    def __init__(self, min_ratio: float = 0.7):
        super().__init__()
        self.min_ratio = min_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ratio = torch.rand(1) * (1 - self.min_ratio) + self.min_ratio
        _, _, h, w = x.shape

        new_h, new_w = int(h * ratio), int(w * ratio)
        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        cropped = x[:, :, top:top+new_h, left:left+new_w]
        return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)


class RandomPerspective(nn.Module):
    """Random perspective transform (differentiable via grid_sample)."""

    def __init__(self, distortion: float = 0.2):
        super().__init__()
        self.distortion = distortion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Random perspective distortion
        d = self.distortion
        pts1 = torch.tensor([[0, 0], [W, 0], [W, H], [0, H]], dtype=torch.float32)
        pts2 = pts1 + torch.randn(4, 2) * d * min(H, W)

        # Compute homography (simplified - use kornia for full implementation)
        # For now, approximate with affine
        theta = torch.tensor([
            [1 + torch.randn(1)*d*0.1, torch.randn(1)*d*0.1, torch.randn(1)*d*0.1],
            [torch.randn(1)*d*0.1, 1 + torch.randn(1)*d*0.1, torch.randn(1)*d*0.1]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(theta.expand(B, -1, -1), x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class RingAttack(nn.Module):
    """Concentric ring rotation attack (our nemesis!)."""

    def __init__(self, num_rings: int = 15, max_angle: float = 1.0):
        super().__init__()
        self.num_rings = num_rings
        self.max_angle = max_angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Create coordinate grid
        y, x_coord = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        cx, cy = W / 2, H / 2
        dx, dy = x_coord - cx, y - cy
        r = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)

        # Determine ring for each pixel
        max_r = math.sqrt(cx**2 + cy**2)
        ring_idx = (r / max_r * self.num_rings).long().clamp(0, self.num_rings - 1)

        # Alternate rotation direction
        angle_rad = self.max_angle * math.pi / 180
        rotation = torch.where(
            ring_idx % 2 == 0,
            torch.full_like(r, angle_rad),
            torch.full_like(r, -angle_rad)
        )

        # Apply rotation per ring
        new_theta = theta + rotation
        new_x = cx + r * torch.cos(new_theta)
        new_y = cy + r * torch.sin(new_theta)

        # Normalize to [-1, 1] for grid_sample
        grid_x = (new_x / (W - 1)) * 2 - 1
        grid_y = (new_y / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


# =============================================================================
# SIGNAL PROCESSING ATTACKS
# =============================================================================

class GaussianNoise(nn.Module):
    """Additive Gaussian noise."""

    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.05)):
        super().__init__()
        self.std_range = std_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.rand(1) * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0, 1)


class GaussianBlur(nn.Module):
    """Gaussian blur with random kernel size."""

    def __init__(self, kernel_range: Tuple[int, int] = (3, 7)):
        super().__init__()
        self.kernel_range = kernel_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(self.kernel_range[0], self.kernel_range[1] + 1, (1,)).item()
        if k % 2 == 0:
            k += 1

        # Create Gaussian kernel
        sigma = k / 3
        ax = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        kernel_1d = torch.exp(-ax**2 / (2 * sigma**2))
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()

        # Apply separable convolution
        kernel = kernel_2d.expand(x.size(1), 1, k, k)
        padding = k // 2

        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


class DifferentiableJPEG(nn.Module):
    """
    Differentiable JPEG approximation.
    Uses DCT + quantization with straight-through estimator.
    """

    def __init__(self, quality_range: Tuple[int, int] = (50, 95)):
        super().__init__()
        self.quality_range = quality_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quality = torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item()

        # Simplified: add quantization noise proportional to quality
        noise_level = (100 - quality) / 100 * 0.1
        noise = torch.randn_like(x) * noise_level

        # Straight-through estimator: add noise in forward, zero grad in backward
        if self.training:
            return x + noise - noise.detach() + (noise * 0.1).detach()
        else:
            return (x + noise).clamp(0, 1)


class MotionBlur(nn.Module):
    """Motion blur in random direction."""

    def __init__(self, kernel_range: Tuple[int, int] = (5, 15)):
        super().__init__()
        self.kernel_range = kernel_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(self.kernel_range[0], self.kernel_range[1] + 1, (1,)).item()
        angle = torch.rand(1) * math.pi

        # Create motion blur kernel
        kernel = torch.zeros(k, k, dtype=x.dtype, device=x.device)
        center = k // 2

        for i in range(k):
            offset = i - center
            px = center + int(offset * math.cos(angle))
            py = center + int(offset * math.sin(angle))
            if 0 <= px < k and 0 <= py < k:
                kernel[py, px] = 1

        kernel = kernel / kernel.sum()
        kernel = kernel.expand(x.size(1), 1, k, k)

        return F.conv2d(x, kernel, padding=k//2, groups=x.size(1))


# =============================================================================
# COLOR/TONE ATTACKS
# =============================================================================

class BrightnessContrast(nn.Module):
    """Random brightness and contrast adjustment."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = (torch.rand(1) * 2 - 1) * self.brightness
        c = 1 + (torch.rand(1) * 2 - 1) * self.contrast

        return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


class HueSaturation(nn.Module):
    """Random hue and saturation adjustment."""

    def __init__(self, hue: float = 0.1, saturation: float = 0.3):
        super().__init__()
        self.hue = hue
        self.saturation = saturation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 3:
            return x  # Only for RGB

        # Convert to HSV (simplified)
        h_shift = (torch.rand(1) * 2 - 1) * self.hue
        s_mult = 1 + (torch.rand(1) * 2 - 1) * self.saturation

        # Approximate hue shift by rotating RGB channels
        if torch.rand(1) > 0.5:
            x = torch.roll(x, 1, dims=1)

        # Saturation: blend with grayscale
        gray = x.mean(dim=1, keepdim=True)
        x = gray + (x - gray) * s_mult

        return x.clamp(0, 1)


class GammaCorrection(nn.Module):
    """Random gamma correction."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5)):
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = torch.rand(1) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        return x.pow(gamma)


# =============================================================================
# COMPOSITE ATTACK LAYER
# =============================================================================

class AttackLayer(nn.Module):
    """
    Composite attack layer for training.
    Randomly applies multiple attacks in sequence.
    """

    def __init__(self,
                 p_geometric: float = 0.5,
                 p_signal: float = 0.5,
                 p_color: float = 0.3):
        super().__init__()

        self.geometric_attacks = nn.ModuleList([
            RandomRotation(max_angle=30),
            RandomScale(min_scale=0.7, max_scale=1.3),
            RandomCrop(min_ratio=0.7),
            RandomPerspective(distortion=0.1),
            RingAttack(num_rings=15, max_angle=0.5),
        ])

        self.signal_attacks = nn.ModuleList([
            GaussianNoise(std_range=(0.01, 0.05)),
            GaussianBlur(kernel_range=(3, 7)),
            DifferentiableJPEG(quality_range=(50, 95)),
            MotionBlur(kernel_range=(5, 11)),
        ])

        self.color_attacks = nn.ModuleList([
            BrightnessContrast(brightness=0.2, contrast=0.2),
            HueSaturation(hue=0.1, saturation=0.3),
            GammaCorrection(gamma_range=(0.7, 1.5)),
        ])

        self.p_geometric = p_geometric
        self.p_signal = p_signal
        self.p_color = p_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply random geometric attack
        if torch.rand(1) < self.p_geometric:
            attack = self.geometric_attacks[torch.randint(len(self.geometric_attacks), (1,))]
            x = attack(x)

        # Apply random signal processing attack
        if torch.rand(1) < self.p_signal:
            attack = self.signal_attacks[torch.randint(len(self.signal_attacks), (1,))]
            x = attack(x)

        # Apply random color attack
        if torch.rand(1) < self.p_color:
            attack = self.color_attacks[torch.randint(len(self.color_attacks), (1,))]
            x = attack(x)

        return x


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing attack library...")

    # Create dummy image
    x = torch.rand(1, 3, 256, 256)

    attacks = [
        ("Rotation", RandomRotation(30)),
        ("Scale", RandomScale(0.7, 1.3)),
        ("Crop", RandomCrop(0.7)),
        ("Perspective", RandomPerspective(0.1)),
        ("Ring Attack", RingAttack(15, 1.0)),
        ("Gaussian Noise", GaussianNoise()),
        ("Gaussian Blur", GaussianBlur()),
        ("JPEG", DifferentiableJPEG()),
        ("Motion Blur", MotionBlur()),
        ("Brightness/Contrast", BrightnessContrast()),
        ("Hue/Saturation", HueSaturation()),
        ("Gamma", GammaCorrection()),
        ("Composite", AttackLayer()),
    ]

    for name, attack in attacks:
        try:
            y = attack(x)
            print(f"  ✓ {name}: {x.shape} -> {y.shape}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nAll attacks tested!")
