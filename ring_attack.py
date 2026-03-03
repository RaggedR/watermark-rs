#!/usr/bin/env python3
"""
Concentric Ring Rotation Attack

This attack divides an image into concentric rings and rotates
alternate rings in opposite directions to break watermark detection.
"""

import numpy as np
from PIL import Image
import sys
from scipy import ndimage

def create_ring_mask(shape, center, inner_radius, outer_radius):
    """Create a boolean mask for a ring region."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (distance >= inner_radius) & (distance < outer_radius)

def rotate_ring(image_array, mask, angle, center):
    """Rotate only the pixels within the mask by the given angle."""
    # Create output array
    result = image_array.copy()

    # Get bounding box of the mask to limit computation
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Extract the region and rotate it
    region = image_array[rmin:rmax+1, cmin:cmax+1].copy()
    region_mask = mask[rmin:rmax+1, cmin:cmax+1]

    # Rotate around the center point (adjusted for region)
    local_center = (center[0] - rmin, center[1] - cmin)

    # For each channel if color image
    if len(region.shape) == 3:
        rotated_region = np.zeros_like(region)
        for c in range(region.shape[2]):
            rotated_region[:,:,c] = ndimage.rotate(
                region[:,:,c], angle, reshape=False,
                order=1, mode='nearest'
            )
    else:
        rotated_region = ndimage.rotate(
            region, angle, reshape=False,
            order=1, mode='nearest'
        )

    # Only apply rotation where mask is True
    if len(image_array.shape) == 3:
        for c in range(image_array.shape[2]):
            result[rmin:rmax+1, cmin:cmax+1, c] = np.where(
                region_mask,
                rotated_region[:,:,c],
                result[rmin:rmax+1, cmin:cmax+1, c]
            )
    else:
        result[rmin:rmax+1, cmin:cmax+1] = np.where(
            region_mask,
            rotated_region,
            result[rmin:rmax+1, cmin:cmax+1]
        )

    return result

def ring_attack(image_path, output_path, num_rings=15, max_angle=1.5, verbose=True):
    """
    Apply concentric ring rotation attack.

    Args:
        image_path: Input image path
        output_path: Output image path
        num_rings: Number of concentric rings
        max_angle: Maximum rotation angle in degrees (alternates ±)
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)

    if verbose:
        print(f"Image size: {img_array.shape}")
        print(f"Number of rings: {num_rings}")
        print(f"Max rotation angle: ±{max_angle}°")
        print()

    height, width = img_array.shape[:2]
    center = (height // 2, width // 2)

    # Maximum radius (corner distance)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    ring_width = max_radius / num_rings

    result = img_array.copy()

    if verbose:
        print("Applying ring rotations:")

    for i in range(num_rings):
        inner_r = i * ring_width
        outer_r = (i + 1) * ring_width

        # Alternate direction: even rings clockwise, odd counter-clockwise
        angle = max_angle if (i % 2 == 0) else -max_angle

        # Create mask for this ring
        mask = create_ring_mask(img_array.shape[:2], center, inner_r, outer_r)

        # Apply rotation to this ring
        result = rotate_ring(result, mask, angle, center)

        if verbose:
            direction = "↻ CW " if angle > 0 else "↺ CCW"
            print(f"  Ring {i+1:2d}: r=[{inner_r:6.1f}, {outer_r:6.1f}] {direction} {abs(angle):.1f}°")

    # Save result
    result_img = Image.fromarray(result.astype(np.uint8))
    result_img.save(output_path)

    if verbose:
        print()
        print(f"Output saved: {output_path}")

    return result

def main():
    if len(sys.argv) < 3:
        print("Usage: python ring_attack.py <input> <output> [num_rings] [max_angle]")
        print()
        print("Arguments:")
        print("  input      Input image path")
        print("  output     Output image path")
        print("  num_rings  Number of concentric rings (default: 15)")
        print("  max_angle  Maximum rotation angle in degrees (default: 1.5)")
        print()
        print("Example:")
        print("  python ring_attack.py watermarked.png attacked.png 20 2.0")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    num_rings = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    max_angle = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5

    print("=" * 50)
    print("    CONCENTRIC RING ROTATION ATTACK")
    print("=" * 50)
    print()

    ring_attack(input_path, output_path, num_rings, max_angle)

    print()
    print("Attack complete!")

if __name__ == "__main__":
    main()
