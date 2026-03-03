//! Pilot signal for rotation-invariant watermark detection.
//!
//! A pilot signal is a sinusoidal pattern embedded on a circular ring in the DFT domain.
//! Because the ring is rotationally symmetric, its shape survives image rotation —
//! only the phase of the sinusoid shifts. By detecting this phase shift, we can
//! determine the rotation angle and correct for it before main watermark detection.
//!
//! ## Algorithm
//!
//! **Embedding**: Add `strength * cos(n_cycles * theta)` to DFT coefficients on a
//! circular ring at radius `r_pilot`.
//!
//! **Detection**: Sample the DFT magnitude along the ring, cross-correlate with
//! `cos(n_cycles * theta)`, and find the peak. The peak position divided by `n_cycles`
//! gives the rotation angle.

use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::f64::consts::PI;
use std::sync::Arc;

/// Configuration for the pilot signal.
#[derive(Debug, Clone)]
pub struct PilotConfig {
    /// Number of sinusoidal cycles around the ring (default: 4)
    pub n_cycles: usize,
    /// Pilot signal amplitude (default: 1.5)
    pub strength: f64,
    /// Ring radius as fraction of prime p (default: 0.33)
    pub ring_radius_fraction: f64,
    /// Ring thickness in pixels (default: 3.0)
    pub ring_width: f64,
}

impl Default for PilotConfig {
    fn default() -> Self {
        Self {
            n_cycles: 4,
            strength: 1.5,
            ring_radius_fraction: 0.33,
            ring_width: 3.0,
        }
    }
}

/// Result of pilot signal detection.
#[derive(Debug, Clone)]
pub struct PilotDetectionResult {
    /// Detected rotation angle in degrees
    pub rotation_angle: f64,
    /// Confidence metric (peak-to-sidelobe ratio)
    pub confidence: f64,
    /// Whether a valid pilot signal was detected
    pub detected: bool,
}

/// Embed pilot signal into DFT spectrum.
///
/// Adds a sinusoidal pattern on a circular ring in the frequency domain.
/// The pattern is: `strength * cos(n_cycles * theta)` where theta is the angle
/// from the horizontal axis.
///
/// # Arguments
/// * `spectrum` - Mutable reference to the 2D DFT spectrum (row-major)
/// * `width` - Image width
/// * `height` - Image height
/// * `p` - Prime used for watermark embedding (determines ring radius)
/// * `config` - Pilot configuration parameters
pub fn embed_pilot(
    spectrum: &mut [Complex64],
    width: usize,
    height: usize,
    p: usize,
    config: &PilotConfig,
) {
    let r_pilot = (p as f64 * config.ring_radius_fraction).round();
    let half_width = config.ring_width / 2.0;

    // DFT center is at (0, 0), but low frequencies are at corners
    // We work in the low-frequency region [0..p+2, 0..p+2]
    // The "center" for our ring is at (1, 1) to avoid DC at (0, 0)
    let center_y = 1.0;
    let center_x = 1.0;

    // Add pilot signal to pixels near the ring
    for i in 0..=p + 2 {
        for j in 0..=p + 2 {
            if i >= height || j >= width {
                continue;
            }

            let dy = i as f64 - center_y;
            let dx = j as f64 - center_x;
            let r = (dx * dx + dy * dy).sqrt();

            // Check if this pixel is on the ring
            if (r - r_pilot).abs() <= half_width {
                let theta = dy.atan2(dx);
                let value = config.strength * (config.n_cycles as f64 * theta).cos();

                let idx = i * width + j;
                spectrum[idx] = Complex64::new(
                    spectrum[idx].re + value,
                    spectrum[idx].im,
                );
            }
        }
    }
}

/// Detect rotation angle from pilot signal.
///
/// This function uses a multi-angle search approach: it tries detecting the
/// watermark at various rotation angles and returns the angle that gives the
/// best detection confidence. This is more robust than trying to detect the
/// pilot signal directly from the DFT, which can be challenging for small rotations.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `p` - Prime used for watermark
/// * `config` - Pilot configuration parameters
///
/// # Returns
/// Detection result with rotation angle (if detected)
pub fn detect_pilot(
    pixels: &[f64],
    width: usize,
    height: usize,
    p: usize,
    config: &PilotConfig,
) -> PilotDetectionResult {
    // Compute DFT magnitude
    let spectrum = dft_2d(pixels, height, width);

    // Calculate ring parameters - use the same coordinate system as embedding
    let r_pilot = (p as f64 * config.ring_radius_fraction).round();
    let center_y = 1.0;
    let center_x = 1.0;

    // Sample magnitude values around the ring (360 samples for 1-degree resolution)
    let n_samples = 360;
    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let theta = 2.0 * PI * (i as f64) / (n_samples as f64);
        let x = center_x + r_pilot * theta.cos();
        let y = center_y + r_pilot * theta.sin();

        // Bilinear interpolation of magnitude
        let mag = bilinear_interpolate_magnitude(&spectrum, width, height, x, y);
        samples.push(mag);
    }

    // Cross-correlate with reference pattern cos(n_cycles * theta)
    // We're looking for the phase shift that maximizes correlation
    let mut best_correlation = f64::NEG_INFINITY;
    let mut correlations = Vec::with_capacity(n_samples);

    for phase_idx in 0..n_samples {
        let phase = 2.0 * PI * (phase_idx as f64) / (n_samples as f64);
        let mut correlation = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let theta = 2.0 * PI * (i as f64) / (n_samples as f64);
            let reference = (config.n_cycles as f64 * (theta - phase)).cos();
            correlation += sample * reference;
        }

        correlations.push(correlation);

        if correlation > best_correlation {
            best_correlation = correlation;
        }
    }

    // Parabolic interpolation for sub-sample precision
    let best_idx = correlations.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let refined_phase = parabolic_interpolate(&correlations, best_idx, n_samples);

    // Convert phase to rotation angle (divide by n_cycles)
    let rotation_rad = refined_phase / config.n_cycles as f64;
    let rotation_deg = rotation_rad * 180.0 / PI;

    // Calculate confidence (peak-to-mean ratio)
    let mean_corr: f64 = correlations.iter().map(|&c| c.abs()).sum::<f64>() / correlations.len() as f64;
    let confidence = if mean_corr > 1e-10 {
        best_correlation / mean_corr
    } else {
        0.0
    };

    // Threshold for valid detection
    let detected = confidence > 1.5;

    PilotDetectionResult {
        rotation_angle: rotation_deg,
        confidence,
        detected,
    }
}

/// Detect rotation by brute-force angle search.
///
/// Tries multiple rotation angles and finds the one that maximizes watermark
/// detection confidence. This is a more robust approach when the pilot signal
/// is weak or when dealing with small rotations.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `angle_range` - Range of angles to search (in degrees, e.g., 5.0 means -5 to +5)
/// * `angle_step` - Step size for angle search (in degrees)
///
/// # Returns
/// The best rotation angle found (in degrees)
pub fn detect_rotation_bruteforce(
    pixels: &[f64],
    width: usize,
    height: usize,
    angle_range: f64,
    angle_step: f64,
) -> (f64, f64) {
    use crate::{extract_mark, generate_array, Correlator2D, largest_prime_for, peak_to_rms};

    let p = largest_prime_for(width, height);
    let correlator = Correlator2D::new(p, p);
    let wm_array = generate_array(p, 1); // Use first Legendre sequence as test

    let mut best_angle = 0.0;
    let mut best_confidence = 0.0;

    // Search in the angle range
    let n_steps = ((2.0 * angle_range) / angle_step).ceil() as i32;

    for i in 0..=n_steps {
        let angle = -angle_range + (i as f64) * angle_step;

        // Rotate image
        let rotated = if angle.abs() < 0.001 {
            pixels.to_vec()
        } else {
            rotate_image(pixels, width, height, angle)
        };

        // Extract and correlate
        let extracted = extract_mark(&rotated, height, width, p);
        let correlation = correlator.correlate(&extracted, &wm_array);
        let (confidence, _) = peak_to_rms(&correlation);

        if confidence > best_confidence {
            best_confidence = confidence;
            best_angle = angle;
        }
    }

    (best_angle, best_confidence)
}

/// Detect rotation by searching around a specific angle.
///
/// Performs a fine-grained search around a given center angle.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `center_angle` - Center of search range (in degrees)
/// * `range` - Search range around center (in degrees)
/// * `step` - Step size for angle search (in degrees)
///
/// # Returns
/// The best rotation angle found (in degrees) and its confidence
pub fn detect_rotation_bruteforce_around(
    pixels: &[f64],
    width: usize,
    height: usize,
    center_angle: f64,
    range: f64,
    step: f64,
) -> (f64, f64) {
    use crate::{extract_mark, generate_array, Correlator2D, largest_prime_for, peak_to_rms};

    let p = largest_prime_for(width, height);
    let correlator = Correlator2D::new(p, p);
    let wm_array = generate_array(p, 1);

    let mut best_angle = center_angle;
    let mut best_confidence = 0.0;

    let n_steps = ((2.0 * range) / step).ceil() as i32;

    for i in 0..=n_steps {
        let angle = center_angle - range + (i as f64) * step;

        let rotated = if angle.abs() < 0.001 {
            pixels.to_vec()
        } else {
            rotate_image(pixels, width, height, angle)
        };

        let extracted = extract_mark(&rotated, height, width, p);
        let correlation = correlator.correlate(&extracted, &wm_array);
        let (confidence, _) = peak_to_rms(&correlation);

        if confidence > best_confidence {
            best_confidence = confidence;
            best_angle = angle;
        }
    }

    (best_angle, best_confidence)
}

/// Rotate an image by the specified angle using bicubic interpolation.
///
/// Uses bicubic interpolation for higher quality rotation with less artifacts.
///
/// # Arguments
/// * `pixels` - Input grayscale pixels in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `angle_deg` - Rotation angle in degrees (positive = counterclockwise)
///
/// # Returns
/// Rotated pixel values
pub fn rotate_image(
    pixels: &[f64],
    width: usize,
    height: usize,
    angle_deg: f64,
) -> Vec<f64> {
    let angle_rad = angle_deg * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    let mut output = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            // Map output (x, y) to input coordinates via inverse rotation
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;

            // Inverse rotation (rotate by -angle to find source)
            let src_x = cx + dx * cos_a + dy * sin_a;
            let src_y = cy - dx * sin_a + dy * cos_a;

            // Bicubic interpolation for higher quality
            output[y * width + x] = bicubic_interpolate(pixels, width, height, src_x, src_y);
        }
    }

    output
}

/// Bilinear interpolation for sub-pixel sampling.
///
/// Returns 0.0 for coordinates outside the image bounds.
pub fn bilinear_interpolate(
    pixels: &[f64],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> f64 {
    if x < 0.0 || y < 0.0 || x >= width as f64 - 1.0 || y >= height as f64 - 1.0 {
        return 0.0;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = pixels[y0 * width + x0];
    let p10 = pixels[y0 * width + x1];
    let p01 = pixels[y1 * width + x0];
    let p11 = pixels[y1 * width + x1];

    // Bilinear interpolation
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;

    top * (1.0 - fy) + bottom * fy
}

/// Cubic interpolation kernel (Catmull-Rom spline).
fn cubic_kernel(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;

    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Bicubic interpolation for higher quality sub-pixel sampling.
///
/// Uses Catmull-Rom spline for smooth interpolation with less artifacts.
fn bicubic_interpolate(
    pixels: &[f64],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> f64 {
    // Need at least 1 pixel margin for bicubic
    if x < 1.0 || y < 1.0 || x >= (width - 2) as f64 || y >= (height - 2) as f64 {
        // Fall back to bilinear at edges
        return bilinear_interpolate(pixels, width, height, x, y);
    }

    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let kx = cubic_kernel(fx);
    let ky = cubic_kernel(fy);

    let mut result = 0.0;

    for j in 0..4 {
        let py = (y0 + j as isize - 1) as usize;
        let mut row_val = 0.0;

        for i in 0..4 {
            let px = (x0 + i as isize - 1) as usize;
            row_val += pixels[py * width + px] * kx[i];
        }

        result += row_val * ky[j];
    }

    result.clamp(0.0, 1.0)
}

/// Bilinear interpolation of DFT magnitude at sub-pixel coordinates.
fn bilinear_interpolate_magnitude(
    spectrum: &[Complex64],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> f64 {
    if x < 0.0 || y < 0.0 || x >= width as f64 - 1.0 || y >= height as f64 - 1.0 {
        return 0.0;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = spectrum[y0 * width + x0].norm();
    let p10 = spectrum[y0 * width + x1].norm();
    let p01 = spectrum[y1 * width + x0].norm();
    let p11 = spectrum[y1 * width + x1].norm();

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;

    top * (1.0 - fy) + bottom * fy
}

/// Parabolic interpolation for sub-sample peak finding.
///
/// Given correlation values around a peak, refines the peak location
/// using a parabolic fit to the three points around the maximum.
fn parabolic_interpolate(correlations: &[f64], peak_idx: usize, n_samples: usize) -> f64 {
    let prev_idx = if peak_idx == 0 { n_samples - 1 } else { peak_idx - 1 };
    let next_idx = (peak_idx + 1) % n_samples;

    let y0 = correlations[prev_idx];
    let y1 = correlations[peak_idx];
    let y2 = correlations[next_idx];

    // Parabolic interpolation: peak at x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
    let denom = y0 - 2.0 * y1 + y2;

    let offset = if denom.abs() > 1e-10 {
        0.5 * (y0 - y2) / denom
    } else {
        0.0
    };

    // Convert index + offset to phase
    let refined_idx = peak_idx as f64 + offset;
    2.0 * PI * refined_idx / (n_samples as f64)
}

/// Compute 2D DFT (real input, complex output).
fn dft_2d(pixels: &[f64], height: usize, width: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();
    let fft_row: Arc<dyn Fft<f64>> = planner.plan_fft_forward(width);
    let fft_col: Arc<dyn Fft<f64>> = planner.plan_fft_forward(height);

    // Convert to complex
    let mut data: Vec<Complex64> = pixels.iter()
        .map(|&p| Complex64::new(p, 0.0))
        .collect();

    // FFT on rows
    for row in 0..height {
        let start = row * width;
        let row_slice = &mut data[start..start + width];
        fft_row.process(row_slice);
    }

    // Transpose
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for y in 0..height {
        for x in 0..width {
            transposed[x * height + y] = data[y * width + x];
        }
    }

    // FFT on columns (now rows after transpose)
    for col in 0..width {
        let start = col * height;
        let col_slice = &mut transposed[start..start + height];
        fft_col.process(col_slice);
    }

    // Transpose back
    let mut result = vec![Complex64::new(0.0, 0.0); width * height];
    for y in 0..height {
        for x in 0..width {
            result[y * width + x] = transposed[x * height + y];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pilot_config_default() {
        let config = PilotConfig::default();
        assert_eq!(config.n_cycles, 4);
        assert!((config.strength - 1.5).abs() < 1e-10);
        assert!((config.ring_radius_fraction - 0.33).abs() < 1e-10);
        assert!((config.ring_width - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_interpolate() {
        // 2x2 image with values 0, 1, 2, 3
        let pixels = vec![0.0, 1.0, 2.0, 3.0];

        // Corner values
        assert!((bilinear_interpolate(&pixels, 2, 2, 0.0, 0.0) - 0.0).abs() < 1e-10);

        // Center should be average
        let center = bilinear_interpolate(&pixels, 2, 2, 0.5, 0.5);
        assert!((center - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_image_identity() {
        // 3x3 image
        let pixels = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let rotated = rotate_image(&pixels, 3, 3, 0.0);

        // Center pixel should be preserved
        assert!((rotated[4] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_pilot_roundtrip() {
        // Create a simple test image
        let width = 64;
        let height = 64;
        let p = 59; // Prime for 64x64 image

        let pixels: Vec<f64> = (0..width * height)
            .map(|i| ((i % 256) as f64) / 255.0)
            .collect();

        let config = PilotConfig::default();

        // Compute DFT
        let mut spectrum = dft_2d(&pixels, height, width);

        // Embed pilot
        embed_pilot(&mut spectrum, width, height, p, &config);

        // For a proper roundtrip test, we'd need to inverse DFT and re-detect
        // This test just verifies the embedding doesn't crash
        assert_eq!(spectrum.len(), width * height);
    }

    #[test]
    fn test_parabolic_interpolation() {
        // Test with a clear peak at index 5
        let mut correlations = vec![0.0; 10];
        correlations[4] = 0.5;
        correlations[5] = 1.0;
        correlations[6] = 0.5;

        let phase = parabolic_interpolate(&correlations, 5, 10);
        let expected = 2.0 * PI * 5.0 / 10.0;

        // Should be close to the peak index
        assert!((phase - expected).abs() < 0.1);
    }
}
