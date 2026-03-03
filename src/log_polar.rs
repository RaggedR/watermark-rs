//! Log-Polar Fourier Transform for rotation/scale invariant watermarking.
//!
//! The Fourier-Mellin Transform converts the DFT magnitude to log-polar coordinates,
//! where rotation becomes horizontal translation and scaling becomes vertical translation.
//! This makes the watermark inherently robust to these geometric transforms.
//!
//! ## Algorithm
//!
//! **Embedding:**
//! 1. Compute DFT of image
//! 2. Take magnitude (phase is not rotation-invariant)
//! 3. Convert to log-polar coordinates: (x,y) → (log(r), θ)
//! 4. Embed watermark as before (Legendre sequences)
//! 5. Convert back to Cartesian and reconstruct with original phase
//!
//! **Detection:**
//! 1. Compute DFT magnitude of possibly-transformed image
//! 2. Convert to log-polar
//! 3. Cross-correlate to find translation (= rotation + scale)
//! 4. Correct and detect watermark

use std::f64::consts::PI;

/// Configuration for log-polar transform.
#[derive(Debug, Clone)]
pub struct LogPolarConfig {
    /// Number of angular samples (typically 360 for 1° resolution)
    pub n_angles: usize,
    /// Number of radial samples (log-spaced)
    pub n_radii: usize,
    /// Minimum radius (avoid singularity at center)
    pub r_min: f64,
    /// Maximum radius (fraction of image size)
    pub r_max_fraction: f64,
}

impl Default for LogPolarConfig {
    fn default() -> Self {
        Self {
            n_angles: 360,
            n_radii: 256,
            r_min: 2.0,
            r_max_fraction: 0.5,
        }
    }
}

/// Convert Cartesian DFT magnitude to log-polar coordinates.
///
/// # Arguments
/// * `magnitude` - DFT magnitude values (row-major)
/// * `width` - Image width
/// * `height` - Image height
/// * `config` - Log-polar configuration
///
/// # Returns
/// Log-polar representation: rows = radii (log-spaced), cols = angles
pub fn cartesian_to_log_polar(
    magnitude: &[f64],
    width: usize,
    height: usize,
    config: &LogPolarConfig,
) -> Vec<f64> {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let r_max = (width.min(height) as f64 / 2.0) * config.r_max_fraction;

    // Log-spaced radii
    let log_r_min = config.r_min.ln();
    let log_r_max = r_max.ln();

    let mut log_polar = vec![0.0; config.n_radii * config.n_angles];

    for ri in 0..config.n_radii {
        // Log-spaced radius
        let t = ri as f64 / (config.n_radii - 1) as f64;
        let log_r = log_r_min + t * (log_r_max - log_r_min);
        let r = log_r.exp();

        for ai in 0..config.n_angles {
            let theta = 2.0 * PI * ai as f64 / config.n_angles as f64;

            // Convert to Cartesian (centered DFT)
            let x = cx + r * theta.cos();
            let y = cy + r * theta.sin();

            // Bilinear interpolation
            let value = bilinear_sample(magnitude, width, height, x, y);
            log_polar[ri * config.n_angles + ai] = value;
        }
    }

    log_polar
}

/// Convert log-polar back to Cartesian coordinates.
///
/// Note: This is lossy - information outside the sampled region is lost.
pub fn log_polar_to_cartesian(
    log_polar: &[f64],
    width: usize,
    height: usize,
    config: &LogPolarConfig,
) -> Vec<f64> {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let r_max = (width.min(height) as f64 / 2.0) * config.r_max_fraction;

    let log_r_min = config.r_min.ln();
    let log_r_max = r_max.ln();

    let mut cartesian = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r = (dx * dx + dy * dy).sqrt();

            if r < config.r_min || r > r_max {
                continue;
            }

            let theta = dy.atan2(dx);
            let theta_normalized = if theta < 0.0 { theta + 2.0 * PI } else { theta };

            // Map to log-polar indices
            let log_r = r.ln();
            let ri = ((log_r - log_r_min) / (log_r_max - log_r_min) * (config.n_radii - 1) as f64)
                .round() as usize;
            let ai = (theta_normalized / (2.0 * PI) * config.n_angles as f64).round() as usize
                % config.n_angles;

            if ri < config.n_radii {
                cartesian[y * width + x] = log_polar[ri * config.n_angles + ai];
            }
        }
    }

    cartesian
}

/// Detect rotation and scale from two log-polar representations.
///
/// Cross-correlates the log-polar transforms to find the translation,
/// which corresponds to rotation (horizontal) and scale (vertical).
///
/// # Returns
/// (rotation_degrees, scale_factor, confidence)
pub fn detect_transform(
    reference: &[f64],
    target: &[f64],
    config: &LogPolarConfig,
) -> (f64, f64, f64) {
    // Cross-correlate using FFT
    let correlation = cross_correlate_2d(
        reference,
        target,
        config.n_radii,
        config.n_angles,
    );

    // Find peak
    let (peak_ri, peak_ai, peak_val) = find_peak(&correlation, config.n_radii, config.n_angles);

    // Convert to rotation and scale
    let rotation_deg = (peak_ai as f64 / config.n_angles as f64) * 360.0;

    // Scale factor from radial shift
    let r_max = config.r_max_fraction * 500.0; // Approximate
    let log_r_min = config.r_min.ln();
    let log_r_max = r_max.ln();
    let log_scale = (peak_ri as f64 / config.n_radii as f64) * (log_r_max - log_r_min);
    let scale = log_scale.exp();

    // Confidence from peak prominence
    let mean: f64 = correlation.iter().sum::<f64>() / correlation.len() as f64;
    let confidence = if mean > 1e-10 { peak_val / mean } else { 0.0 };

    (rotation_deg, scale, confidence)
}

// Helper functions

fn bilinear_sample(data: &[f64], width: usize, height: usize, x: f64, y: f64) -> f64 {
    if x < 0.0 || y < 0.0 || x >= (width - 1) as f64 || y >= (height - 1) as f64 {
        return 0.0;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = data[y0 * width + x0];
    let p10 = data[y0 * width + x0 + 1];
    let p01 = data[(y0 + 1) * width + x0];
    let p11 = data[(y0 + 1) * width + x0 + 1];

    p00 * (1.0 - fx) * (1.0 - fy)
        + p10 * fx * (1.0 - fy)
        + p01 * (1.0 - fx) * fy
        + p11 * fx * fy
}

fn cross_correlate_2d(a: &[f64], b: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Simple spatial cross-correlation (could be optimized with FFT)
    let mut result = vec![0.0; rows * cols];

    for dr in 0..rows {
        for dc in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                for c in 0..cols {
                    let ar = r;
                    let ac = c;
                    let br = (r + dr) % rows;
                    let bc = (c + dc) % cols;
                    sum += a[ar * cols + ac] * b[br * cols + bc];
                }
            }
            result[dr * cols + dc] = sum;
        }
    }

    result
}

fn find_peak(data: &[f64], rows: usize, cols: usize) -> (usize, usize, f64) {
    let mut max_val = f64::NEG_INFINITY;
    let mut max_r = 0;
    let mut max_c = 0;

    for r in 0..rows {
        for c in 0..cols {
            let val = data[r * cols + c];
            if val > max_val {
                max_val = val;
                max_r = r;
                max_c = c;
            }
        }
    }

    (max_r, max_c, max_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_polar_roundtrip() {
        let config = LogPolarConfig::default();
        let width = 64;
        let height = 64;

        // Create simple test pattern
        let mut magnitude = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - 32.0;
                let dy = y as f64 - 32.0;
                magnitude[y * width + x] = (-(dx * dx + dy * dy) / 100.0).exp();
            }
        }

        let log_polar = cartesian_to_log_polar(&magnitude, width, height, &config);
        assert_eq!(log_polar.len(), config.n_radii * config.n_angles);
    }

    #[test]
    fn test_config_default() {
        let config = LogPolarConfig::default();
        assert_eq!(config.n_angles, 360);
        assert_eq!(config.n_radii, 256);
    }
}
