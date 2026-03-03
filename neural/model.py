"""
Frequency Recovery Network (FRNet)

A U-Net style network that recovers DFT watermark coefficients
from JPEG-attacked images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: ConvBlock + MaxPool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        conv_out = self.conv(x)
        pooled = self.pool(conv_out)
        return pooled, conv_out  # Return skip connection


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concat + ConvBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FRNet(nn.Module):
    """
    Frequency Recovery Network.

    Takes a grayscale image and outputs the recovered DFT watermark region.

    Args:
        p: Prime number defining watermark region size (p×p)
        base_channels: Base number of channels (default: 32)
    """

    def __init__(self, p=593, base_channels=32):
        super().__init__()
        self.p = p

        # Encoder
        self.down1 = DownBlock(1, base_channels)           # 1 → 32
        self.down2 = DownBlock(base_channels, base_channels * 2)    # 32 → 64
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)  # 64 → 128
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)  # 128 → 256

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)  # 256 → 512

        # Decoder
        self.up1 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)  # 512+256 → 256
        self.up2 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)   # 256+128 → 128
        self.up3 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)   # 128+64 → 64
        self.up4 = UpBlock(base_channels * 2 + base_channels, base_channels)           # 64+32 → 32

        # Output: resize to p×p and output 2 channels (real + imaginary)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((p, p))
        self.output_conv = nn.Conv2d(base_channels, 2, 1)

    def forward(self, x):
        """
        Args:
            x: Grayscale image tensor (B, 1, H, W)

        Returns:
            DFT watermark region (B, 2, p, p) - channels are [real, imaginary]
        """
        # Encoder with skip connections
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Output
        x = self.adaptive_pool(x)
        x = self.output_conv(x)

        return x

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FRNetLite(nn.Module):
    """
    Lightweight version of FRNet for faster training/inference.
    ~500K parameters instead of ~2.2M.
    """

    def __init__(self, p=593, base_channels=16):
        super().__init__()
        self.p = p

        # Encoder (fewer layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((p, p))
        self.output_conv = nn.Conv2d(base_channels, 2, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.adaptive_pool(x)
        x = self.output_conv(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Quick test of model architecture."""
    print("Testing FRNet...")
    model = FRNet(p=593, base_channels=32)
    print(f"  Parameters: {model.count_parameters():,}")

    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (2, 2, 593, 593), f"Expected (2, 2, 593, 593), got {y.shape}"
    print("  ✓ FRNet OK\n")

    print("Testing FRNetLite...")
    model_lite = FRNetLite(p=593, base_channels=16)
    print(f"  Parameters: {model_lite.count_parameters():,}")

    y_lite = model_lite(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_lite.shape}")
    assert y_lite.shape == (2, 2, 593, 593), f"Expected (2, 2, 593, 593), got {y_lite.shape}"
    print("  ✓ FRNetLite OK\n")


if __name__ == "__main__":
    test_model()
