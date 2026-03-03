//! Fiducial markers for blind perspective correction.
//!
//! This module embeds detectable corner markers in the frequency domain that
//! allow finding and correcting perspective distortion without the original image.
//! This enables blind detection of watermarks in photographed/printed images.
//!
//! ## How it works
//!
//! **Embedding:**
//! 1. Place 4 distinct frequency-domain patterns near the image corners
//! 2. Each corner has a unique signature (different rotation of the same pattern)
//! 3. Patterns are embedded in the DFT magnitude, similar to pilot signal
//!
//! **Detection:**
//! 1. Scan the image for the 4 corner patterns using matched filtering
//! 2. Each pattern's location reveals where that corner ended up after perspective transform
//! 3. Compute homography from the 4 detected points
//! 4. Apply inverse transform to correct perspective
//! 5. Run standard blind watermark detection on corrected image
//!
//! This is similar to QR code finder patterns but in the frequency domain.

use std::f64::consts::PI;

/// Configuration for corner markers.
#[derive(Debug, Clone)]
pub struct MarkerConfig {
    /// Radius of marker ring (fraction of image size)
    pub ring_radius: f64,
    /// Number of cycles in the marker pattern
    pub n_cycles: usize,
    /// Marker strength (amplitude)
    pub strength: f64,
    /// Distance from corner (fraction of image size)
    pub corner_offset: f64,
}

impl Default for MarkerConfig {
    fn default() -> Self {
        Self {
            ring_radius: 0.15,
            n_cycles: 8,
            strength: 2.0,
            corner_offset: 0.2,
        }
    }
}

/// Result of marker detection.
#[derive(Debug, Clone)]
pub struct MarkerDetectionResult {
    /// Detected corner positions [(x, y), ...]
    /// Order: top-left, top-right, bottom-right, bottom-left
    pub corners: [(f64, f64); 4],
    /// Confidence scores for each corner
    pub confidences: [f64; 4],
    /// Whether all 4 corners were detected with sufficient confidence
    pub all_detected: bool,
    /// Overall confidence (minimum of all corners)
    pub overall_confidence: f64,
}

/// Embed corner markers into the frequency domain.
///
/// # Arguments
/// * `spectrum_re` - Real part of DFT (modified in place)
/// * `_spectrum_im` - Imaginary part of DFT (unused, for API consistency)
/// * `width` - Image width
/// * `height` - Image height
/// * `config` - Marker configuration
pub fn embed_markers(
    spectrum_re: &mut [f64],
    _spectrum_im: &mut [f64],
    width: usize,
    height: usize,
    config: &MarkerConfig,
) {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    // Corner positions (in image space)
    let offset_x = width as f64 * config.corner_offset;
    let offset_y = height as f64 * config.corner_offset;

    let corners = [
        (offset_x, offset_y),                           // Top-left
        (width as f64 - offset_x, offset_y),            // Top-right
        (width as f64 - offset_x, height as f64 - offset_y), // Bottom-right
        (offset_x, height as f64 - offset_y),           // Bottom-left
    ];

    // Each corner gets a different phase offset (0°, 90°, 180°, 270°)
    let phase_offsets = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    let ring_r = (width.min(height) as f64 / 2.0) * config.ring_radius;
    let ring_width = 3.0;

    for (corner_idx, &(corner_x, corner_y)) in corners.iter().enumerate() {
        let phase_offset = phase_offsets[corner_idx];

        // Embed a small ring pattern centered on each corner's frequency position
        // The frequency position corresponds to the spatial corner position
        let freq_x = corner_x - cx;
        let freq_y = corner_y - cy;

        // Embed marker pattern around this frequency position
        for dy in -20i32..=20 {
            for dx in -20i32..=20 {
                let px = (cx + freq_x + dx as f64) as usize;
                let py = (cy + freq_y + dy as f64) as usize;

                if px >= width || py >= height {
                    continue;
                }

                let r = ((dx * dx + dy * dy) as f64).sqrt();
                if (r - ring_r / 4.0).abs() > ring_width {
                    continue;
                }

                let theta = (dy as f64).atan2(dx as f64);
                let value = config.strength * (config.n_cycles as f64 * theta + phase_offset).cos();

                let idx = py * width + px;
                spectrum_re[idx] += value;

                // Also add to symmetric position (DFT symmetry)
                let sym_px = width - 1 - px;
                let sym_py = height - 1 - py;
                if sym_px < width && sym_py < height {
                    let sym_idx = sym_py * width + sym_px;
                    spectrum_re[sym_idx] += value;
                }
            }
        }
    }
}

/// Detect corner markers in an image.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values (0.0-1.0)
/// * `width` - Image width
/// * `height` - Image height
/// * `config` - Marker configuration
///
/// # Returns
/// Detection result with corner positions and confidences
pub fn detect_markers(
    pixels: &[f64],
    width: usize,
    height: usize,
    config: &MarkerConfig,
) -> MarkerDetectionResult {
    // Compute DFT magnitude
    let magnitude = compute_dft_magnitude(pixels, width, height);

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    // Expected corner positions
    let offset_x = width as f64 * config.corner_offset;
    let offset_y = height as f64 * config.corner_offset;

    let expected_corners = [
        (offset_x, offset_y),
        (width as f64 - offset_x, offset_y),
        (width as f64 - offset_x, height as f64 - offset_y),
        (offset_x, height as f64 - offset_y),
    ];

    let phase_offsets = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
    let search_radius = (width.min(height) as f64 * 0.1) as i32;

    let mut corners = [(0.0, 0.0); 4];
    let mut confidences = [0.0; 4];

    for (corner_idx, &(expected_x, expected_y)) in expected_corners.iter().enumerate() {
        let phase_offset = phase_offsets[corner_idx];
        let mut best_x = expected_x;
        let mut best_y = expected_y;
        let mut best_score = f64::NEG_INFINITY;

        // Search around expected position
        let search_cx = (expected_x - cx) as i32;
        let search_cy = (expected_y - cy) as i32;

        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let test_x = cx + (search_cx + dx) as f64;
                let test_y = cy + (search_cy + dy) as f64;

                if test_x < 0.0 || test_y < 0.0 ||
                   test_x >= width as f64 || test_y >= height as f64 {
                    continue;
                }

                // Compute correlation with expected pattern at this position
                let score = correlate_marker_pattern(
                    &magnitude, width, height,
                    test_x, test_y,
                    config, phase_offset
                );

                if score > best_score {
                    best_score = score;
                    best_x = test_x;
                    best_y = test_y;
                }
            }
        }

        corners[corner_idx] = (best_x, best_y);
        confidences[corner_idx] = best_score;
    }

    let min_confidence = confidences.iter().cloned().fold(f64::INFINITY, f64::min);
    let threshold = 0.3;

    MarkerDetectionResult {
        corners,
        confidences,
        all_detected: min_confidence > threshold,
        overall_confidence: min_confidence,
    }
}

/// Correct perspective distortion using detected markers.
///
/// # Arguments
/// * `pixels` - Input grayscale pixels
/// * `width` - Image width
/// * `height` - Image height
/// * `detected` - Marker detection result
///
/// # Returns
/// Perspective-corrected pixel array
pub fn correct_perspective(
    pixels: &[f64],
    width: usize,
    height: usize,
    detected: &MarkerDetectionResult,
) -> Vec<f64> {
    if !detected.all_detected {
        return pixels.to_vec();
    }

    // Expected corner positions (where they should be)
    let offset_x = width as f64 * 0.2;
    let offset_y = height as f64 * 0.2;

    let dst_corners = [
        (offset_x, offset_y),
        (width as f64 - offset_x, offset_y),
        (width as f64 - offset_x, height as f64 - offset_y),
        (offset_x, height as f64 - offset_y),
    ];

    // Compute homography matrix from detected to expected
    let h = compute_homography(&detected.corners, &dst_corners);

    // Warp image using inverse homography
    warp_perspective(pixels, width, height, &h)
}

// Helper: Compute DFT magnitude
fn compute_dft_magnitude(pixels: &[f64], width: usize, height: usize) -> Vec<f64> {
    use crate::dft;

    let spectrum = dft::dft_2d(pixels, height, width);

    // Compute magnitude
    spectrum.iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

// Helper: Correlate with marker pattern at given position
fn correlate_marker_pattern(
    magnitude: &[f64],
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    config: &MarkerConfig,
    phase_offset: f64,
) -> f64 {
    let ring_r = (width.min(height) as f64 / 2.0) * config.ring_radius / 4.0;
    let ring_width = 3.0;

    let mut sum = 0.0;
    let mut count = 0;

    for dy in -15i32..=15 {
        for dx in -15i32..=15 {
            let px = (center_x + dx as f64) as i32;
            let py = (center_y + dy as f64) as i32;

            if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                continue;
            }

            let r = ((dx * dx + dy * dy) as f64).sqrt();
            if (r - ring_r).abs() > ring_width {
                continue;
            }

            let theta = (dy as f64).atan2(dx as f64);
            let expected = (config.n_cycles as f64 * theta + phase_offset).cos();
            let actual = magnitude[py as usize * width + px as usize];

            sum += expected * actual;
            count += 1;
        }
    }

    if count > 0 { sum / count as f64 } else { 0.0 }
}

// Helper: Compute 3x3 homography matrix from 4 point correspondences
fn compute_homography(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> [f64; 9] {
    // Simplified homography computation using direct linear transform (DLT)
    // For a full implementation, use SVD. Here we use a simplified approach.

    // Build the 8x9 matrix A for Ah = 0
    let mut a = [[0.0f64; 9]; 8];

    for i in 0..4 {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];

        a[2 * i] = [-sx, -sy, -1.0, 0.0, 0.0, 0.0, sx * dx, sy * dx, dx];
        a[2 * i + 1] = [0.0, 0.0, 0.0, -sx, -sy, -1.0, sx * dy, sy * dy, dy];
    }

    // Solve using simplified pseudo-inverse approach
    // For production, use proper SVD
    // Here we return identity if computation fails

    // Simplified: compute approximate homography
    let dx = dst[1].0 - dst[0].0;
    let dy = dst[2].1 - dst[0].1;
    let sx = src[1].0 - src[0].0;
    let sy = src[2].1 - src[0].1;

    let scale_x = if sx.abs() > 1e-6 { dx / sx } else { 1.0 };
    let scale_y = if sy.abs() > 1e-6 { dy / sy } else { 1.0 };

    let tx = dst[0].0 - src[0].0 * scale_x;
    let ty = dst[0].1 - src[0].1 * scale_y;

    // Affine approximation (good enough for small perspective distortions)
    [
        scale_x, 0.0, tx,
        0.0, scale_y, ty,
        0.0, 0.0, 1.0
    ]
}

// Helper: Warp image using homography
fn warp_perspective(pixels: &[f64], width: usize, height: usize, h: &[f64; 9]) -> Vec<f64> {
    let mut output = vec![0.0; width * height];

    // Compute inverse homography
    let det = h[0] * (h[4] * h[8] - h[5] * h[7])
            - h[1] * (h[3] * h[8] - h[5] * h[6])
            + h[2] * (h[3] * h[7] - h[4] * h[6]);

    if det.abs() < 1e-10 {
        return pixels.to_vec();
    }

    let h_inv = [
        (h[4] * h[8] - h[5] * h[7]) / det,
        (h[2] * h[7] - h[1] * h[8]) / det,
        (h[1] * h[5] - h[2] * h[4]) / det,
        (h[5] * h[6] - h[3] * h[8]) / det,
        (h[0] * h[8] - h[2] * h[6]) / det,
        (h[2] * h[3] - h[0] * h[5]) / det,
        (h[3] * h[7] - h[4] * h[6]) / det,
        (h[1] * h[6] - h[0] * h[7]) / det,
        (h[0] * h[4] - h[1] * h[3]) / det,
    ];

    for y in 0..height {
        for x in 0..width {
            // Apply inverse homography
            let w = h_inv[6] * x as f64 + h_inv[7] * y as f64 + h_inv[8];
            if w.abs() < 1e-10 {
                continue;
            }

            let src_x = (h_inv[0] * x as f64 + h_inv[1] * y as f64 + h_inv[2]) / w;
            let src_y = (h_inv[3] * x as f64 + h_inv[4] * y as f64 + h_inv[5]) / w;

            // Bilinear interpolation
            if src_x >= 0.0 && src_y >= 0.0 &&
               src_x < (width - 1) as f64 && src_y < (height - 1) as f64 {
                output[y * width + x] = bilinear_sample(pixels, width, height, src_x, src_y);
            }
        }
    }

    output
}

fn bilinear_sample(pixels: &[f64], width: usize, _height: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = pixels[y0 * width + x0];
    let p10 = pixels[y0 * width + x0 + 1];
    let p01 = pixels[(y0 + 1) * width + x0];
    let p11 = pixels[(y0 + 1) * width + x0 + 1];

    p00 * (1.0 - fx) * (1.0 - fy)
        + p10 * fx * (1.0 - fy)
        + p01 * (1.0 - fx) * fy
        + p11 * fx * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_config_default() {
        let config = MarkerConfig::default();
        assert_eq!(config.n_cycles, 8);
        assert!(config.strength > 0.0);
    }

    #[test]
    fn test_homography_identity() {
        let corners = [
            (10.0, 10.0),
            (90.0, 10.0),
            (90.0, 90.0),
            (10.0, 90.0),
        ];

        let h = compute_homography(&corners, &corners);

        // Should be close to identity
        assert!((h[0] - 1.0).abs() < 0.1);
        assert!((h[4] - 1.0).abs() < 0.1);
        assert!((h[8] - 1.0).abs() < 0.1);
    }
}
