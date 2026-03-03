//! Image texture and frequency analysis for watermark robustness prediction.
//!
//! These metrics help understand:
//! 1. Which images are better at "hiding" watermarks (texture analysis)
//! 2. Why some degradations are more destructive (frequency analysis)
//! 3. How to improve the watermarking algorithm
//!
//! ## Strength Recommendation
//! Use [`recommend_strength`] to get adaptive watermark strength based on image content.
//! The recommendation uses Noise Visibility Function (NVF) to find regions where
//! watermarks can be hidden without visible artifacts.

use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

/// Watermark strength recommendation based on image analysis.
#[derive(Debug, Clone)]
pub struct StrengthRecommendation {
    /// Minimum strength that will survive identity (no attack)
    pub min_detectable: f64,
    /// Maximum strength before watermark becomes visible
    pub max_invisible: f64,
    /// Recommended strength balancing robustness and invisibility
    pub recommended: f64,
    /// Strength needed to survive JPEG Q90
    pub jpeg_q90_strength: f64,
    /// Strength needed to survive JPEG Q80
    pub jpeg_q80_strength: f64,
    /// Overall texture score (0=flat/visible, 1=textured/hidden)
    pub texture_score: f64,
    /// Percentage of image that is "flat" (watermark visible)
    pub flat_region_percent: f64,
    /// Average NVF (Noise Visibility Function) value
    pub avg_nvf: f64,
    /// Minimum NVF (worst-case visibility region)
    pub min_nvf: f64,
}

/// Comprehensive image analysis metrics.
#[derive(Debug, Clone)]
pub struct ImageMetrics {
    // Basic properties
    pub width: usize,
    pub height: usize,
    pub prime: usize,

    // Intensity statistics
    pub mean_intensity: f64,
    pub std_intensity: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,

    // Texture metrics
    pub local_variance: f64,       // Average variance in 8x8 blocks
    pub entropy: f64,              // Shannon entropy (bits per pixel)
    pub edge_density: f64,         // Proportion of edge pixels (Sobel)

    // Frequency domain metrics
    pub dc_component: f64,         // Magnitude of DC (0,0) coefficient
    pub low_freq_energy: f64,      // Energy in p×p low-frequency region
    pub total_energy: f64,         // Total spectral energy
    pub low_freq_ratio: f64,       // low_freq / total (watermark interference potential)
    pub spectral_flatness: f64,    // How "white" the spectrum is (0=tonal, 1=noise-like)

    // Watermark-specific metrics
    pub watermark_region_energy: f64,  // Energy specifically in [1..p+1, 1..p+1]
    pub watermark_region_snr: f64,     // SNR in watermark region
}

/// Analyze an image and compute all metrics.
pub fn analyze_image(pixels: &[u8], width: usize, height: usize) -> ImageMetrics {
    let n = pixels.len();
    let p = largest_prime_for_analysis(width, height);

    // Convert to f64
    let pixels_f64: Vec<f64> = pixels.iter().map(|&v| v as f64 / 255.0).collect();

    // Basic intensity statistics
    let mean_intensity = pixels_f64.iter().sum::<f64>() / n as f64;
    let variance = pixels_f64.iter().map(|&v| (v - mean_intensity).powi(2)).sum::<f64>() / n as f64;
    let std_intensity = variance.sqrt();
    let min_intensity = pixels_f64.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_intensity = pixels_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Texture metrics
    let local_variance = compute_local_variance(&pixels_f64, width, height);
    let entropy = compute_entropy(pixels);
    let edge_density = compute_edge_density(&pixels_f64, width, height);

    // Frequency domain analysis
    let spectrum = compute_spectrum(&pixels_f64, height, width);
    let (dc_component, low_freq_energy, total_energy, watermark_region_energy) =
        analyze_spectrum(&spectrum, width, height, p);

    let low_freq_ratio = if total_energy > 0.0 { low_freq_energy / total_energy } else { 0.0 };
    let spectral_flatness = compute_spectral_flatness(&spectrum);

    // Estimate SNR in watermark region (before watermarking)
    // This indicates how much "room" there is for the watermark
    let watermark_region_snr = estimate_watermark_snr(&spectrum, width, height, p);

    ImageMetrics {
        width,
        height,
        prime: p,
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        local_variance,
        entropy,
        edge_density,
        dc_component,
        low_freq_energy,
        total_energy,
        low_freq_ratio,
        spectral_flatness,
        watermark_region_energy,
        watermark_region_snr,
    }
}

fn largest_prime_for_analysis(width: usize, height: usize) -> usize {
    let max_size = width.min(height).saturating_sub(2);
    for n in (2..=max_size).rev() {
        if is_prime(n) {
            return n;
        }
    }
    2
}

fn is_prime(n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let sqrt = (n as f64).sqrt() as usize;
    for i in (3..=sqrt).step_by(2) {
        if n % i == 0 { return false; }
    }
    true
}

/// Compute average local variance using 8x8 blocks.
fn compute_local_variance(pixels: &[f64], width: usize, height: usize) -> f64 {
    let block_size = 8;
    let mut total_variance = 0.0;
    let mut block_count = 0;

    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            let mut block_sum = 0.0;
            let mut block_count_inner = 0;

            // Compute block mean
            for y in by..(by + block_size).min(height) {
                for x in bx..(bx + block_size).min(width) {
                    block_sum += pixels[y * width + x];
                    block_count_inner += 1;
                }
            }
            let block_mean = block_sum / block_count_inner as f64;

            // Compute block variance
            let mut block_variance = 0.0;
            for y in by..(by + block_size).min(height) {
                for x in bx..(bx + block_size).min(width) {
                    let diff = pixels[y * width + x] - block_mean;
                    block_variance += diff * diff;
                }
            }
            block_variance /= block_count_inner as f64;

            total_variance += block_variance;
            block_count += 1;
        }
    }

    total_variance / block_count as f64
}

/// Compute Shannon entropy (bits per pixel).
fn compute_entropy(pixels: &[u8]) -> f64 {
    let mut histogram = [0u64; 256];
    for &p in pixels {
        histogram[p as usize] += 1;
    }

    let n = pixels.len() as f64;
    let mut entropy = 0.0;

    for &count in &histogram {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Compute edge density using Sobel operator.
fn compute_edge_density(pixels: &[f64], width: usize, height: usize) -> f64 {
    let mut edge_count = 0;
    let threshold = 0.1; // Edge threshold

    for y in 1..height-1 {
        for x in 1..width-1 {
            // Sobel kernels
            let gx =
                -pixels[(y-1)*width + (x-1)] + pixels[(y-1)*width + (x+1)]
                -2.0*pixels[y*width + (x-1)] + 2.0*pixels[y*width + (x+1)]
                -pixels[(y+1)*width + (x-1)] + pixels[(y+1)*width + (x+1)];

            let gy =
                -pixels[(y-1)*width + (x-1)] - 2.0*pixels[(y-1)*width + x] - pixels[(y-1)*width + (x+1)]
                +pixels[(y+1)*width + (x-1)] + 2.0*pixels[(y+1)*width + x] + pixels[(y+1)*width + (x+1)];

            let magnitude = (gx*gx + gy*gy).sqrt();
            if magnitude > threshold {
                edge_count += 1;
            }
        }
    }

    edge_count as f64 / ((width - 2) * (height - 2)) as f64
}

/// Compute 2D DFT magnitude spectrum.
fn compute_spectrum(pixels: &[f64], height: usize, width: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft_row: Arc<dyn Fft<f64>> = planner.plan_fft_forward(width);
    let fft_col: Arc<dyn Fft<f64>> = planner.plan_fft_forward(height);

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

    // FFT on columns
    for col in 0..width {
        let start = col * height;
        let col_slice = &mut transposed[start..start + height];
        fft_col.process(col_slice);
    }

    // Transpose back and compute magnitude
    let mut spectrum = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            spectrum[y * width + x] = transposed[x * height + y].norm();
        }
    }

    spectrum
}

/// Analyze spectrum for frequency distribution.
fn analyze_spectrum(spectrum: &[f64], width: usize, height: usize, p: usize) -> (f64, f64, f64, f64) {
    let dc_component = spectrum[0];

    let mut low_freq_energy = 0.0;
    let mut watermark_region_energy = 0.0;
    let mut total_energy = 0.0;

    for y in 0..height {
        for x in 0..width {
            let mag_sq = spectrum[y * width + x].powi(2);
            total_energy += mag_sq;

            // Low frequency region (p×p around DC)
            if y < p && x < p {
                low_freq_energy += mag_sq;
            }

            // Watermark region [1..p+1, 1..p+1] (avoiding DC)
            if y >= 1 && y <= p && x >= 1 && x <= p {
                watermark_region_energy += mag_sq;
            }
        }
    }

    (dc_component, low_freq_energy.sqrt(), total_energy.sqrt(), watermark_region_energy.sqrt())
}

/// Compute spectral flatness (Wiener entropy).
/// 0 = tonal (energy concentrated), 1 = noise-like (flat spectrum)
fn compute_spectral_flatness(spectrum: &[f64]) -> f64 {
    let n = spectrum.len() as f64;

    // Geometric mean
    let log_sum: f64 = spectrum.iter()
        .filter(|&&v| v > 1e-10)
        .map(|&v| v.ln())
        .sum();
    let geo_mean = (log_sum / n).exp();

    // Arithmetic mean
    let arith_mean = spectrum.iter().sum::<f64>() / n;

    if arith_mean > 1e-10 {
        geo_mean / arith_mean
    } else {
        0.0
    }
}

/// Estimate SNR available for watermark in the target region.
fn estimate_watermark_snr(spectrum: &[f64], width: usize, height: usize, p: usize) -> f64 {
    // Compute mean and std of watermark region (excluding DC)
    let mut values = Vec::new();
    for y in 1..=p.min(height-1) {
        for x in 1..=p.min(width-1) {
            values.push(spectrum[y * width + x]);
        }
    }

    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std = variance.sqrt();

    if std > 1e-10 {
        mean / std
    } else {
        f64::INFINITY
    }
}

/// Recommend watermark strength based on image texture analysis.
///
/// Uses Noise Visibility Function (NVF) to determine how much watermark
/// an image can hide. Textured images can tolerate stronger watermarks,
/// while flat images require weaker watermarks to remain invisible.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values as u8 (0-255)
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
/// A `StrengthRecommendation` with suggested strength values for different use cases.
pub fn recommend_strength(pixels: &[u8], width: usize, height: usize) -> StrengthRecommendation {
    let pixels_f64: Vec<f64> = pixels.iter().map(|&v| v as f64 / 255.0).collect();

    // Compute NVF (Noise Visibility Function) for each block
    let block_size = 8;
    let mut nvf_values = Vec::new();
    let mut flat_blocks = 0;
    let mut total_blocks = 0;

    // Find max variance across all blocks (for NVF normalization)
    let mut variances = Vec::new();
    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            let variance = compute_block_variance(&pixels_f64, width, height, bx, by, block_size);
            variances.push(variance);
        }
    }

    let max_variance = variances.iter().cloned().fold(0.0_f64, f64::max);
    let variance_threshold = max_variance * 0.1; // Blocks with <10% of max variance are "flat"

    // Compute NVF for each block
    for variance in &variances {
        // NVF = σ² / (σ² + σ²_max)
        // NVF ≈ 0 for flat regions (watermark visible)
        // NVF ≈ 1 for textured regions (watermark hidden)
        let nvf = if max_variance > 1e-10 {
            variance / (variance + max_variance)
        } else {
            0.0
        };
        nvf_values.push(nvf);

        if *variance < variance_threshold {
            flat_blocks += 1;
        }
        total_blocks += 1;
    }

    let avg_nvf = if nvf_values.is_empty() {
        0.0
    } else {
        nvf_values.iter().sum::<f64>() / nvf_values.len() as f64
    };

    let min_nvf = nvf_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_nvf = if min_nvf.is_infinite() { 0.0 } else { min_nvf };

    let flat_region_percent = if total_blocks > 0 {
        (flat_blocks as f64 / total_blocks as f64) * 100.0
    } else {
        100.0
    };

    // Texture score: weighted combination of avg NVF and min NVF
    // We care about worst case (min) because that's where artifacts appear
    let texture_score = 0.3 * avg_nvf + 0.7 * min_nvf;

    // Map texture score to strength recommendations
    // Based on empirical testing:
    // - Very flat images (score < 0.1): max strength ~2.0
    // - Mixed images (score 0.1-0.3): max strength ~4.0
    // - Textured images (score > 0.3): max strength ~8.0+

    let max_invisible = if texture_score < 0.05 {
        1.5 + texture_score * 10.0  // 1.5 - 2.0
    } else if texture_score < 0.15 {
        2.0 + (texture_score - 0.05) * 30.0  // 2.0 - 5.0
    } else if texture_score < 0.30 {
        5.0 + (texture_score - 0.15) * 20.0  // 5.0 - 8.0
    } else {
        8.0 + (texture_score - 0.30) * 10.0  // 8.0 - 15.0
    };

    let max_invisible = max_invisible.min(15.0); // Cap at 15

    // Minimum detectable is always ~1.0 (below noise floor at threshold 6.0)
    let min_detectable = 1.0;

    // Recommended: 70% of max invisible (safety margin)
    let recommended = (max_invisible * 0.7).max(2.0);

    // JPEG survival requirements (from empirical testing):
    // Q90: need strength >= 5.0 (confidence ~12)
    // Q80: need strength >= 8.0 (confidence ~14)
    let jpeg_q90_strength = 5.0;
    let jpeg_q80_strength = 8.0;

    StrengthRecommendation {
        min_detectable,
        max_invisible,
        recommended,
        jpeg_q90_strength,
        jpeg_q80_strength,
        texture_score,
        flat_region_percent,
        avg_nvf,
        min_nvf,
    }
}

/// Compute variance of a single block.
fn compute_block_variance(
    pixels: &[f64],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    for y in by..(by + block_size).min(height) {
        for x in bx..(bx + block_size).min(width) {
            sum += pixels[y * width + x];
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f64;
    let mut variance = 0.0;

    for y in by..(by + block_size).min(height) {
        for x in bx..(bx + block_size).min(width) {
            let diff = pixels[y * width + x] - mean;
            variance += diff * diff;
        }
    }

    variance / count as f64
}

/// Recommend strength from f64 pixels (normalized 0-1).
pub fn recommend_strength_f64(pixels: &[f64], width: usize, height: usize) -> StrengthRecommendation {
    let pixels_u8: Vec<u8> = pixels.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8).collect();
    recommend_strength(&pixels_u8, width, height)
}

/// Format strength recommendation as a human-readable string.
pub fn format_recommendation(rec: &StrengthRecommendation) -> String {
    format!(
        "Texture Score: {:.2} ({})
Flat Regions:  {:.1}%
NVF (avg/min): {:.3} / {:.3}

Strength Recommendations:
  Invisible max:  {:.1}
  Recommended:    {:.1}
  JPEG Q90 safe:  {:.1}{}
  JPEG Q80 safe:  {:.1}{}",
        rec.texture_score,
        if rec.texture_score > 0.3 { "highly textured" }
        else if rec.texture_score > 0.15 { "moderately textured" }
        else if rec.texture_score > 0.05 { "some texture" }
        else { "mostly flat" },
        rec.flat_region_percent,
        rec.avg_nvf,
        rec.min_nvf,
        rec.max_invisible,
        rec.recommended,
        rec.jpeg_q90_strength,
        if rec.jpeg_q90_strength <= rec.max_invisible { " ✓" } else { " (may be visible)" },
        rec.jpeg_q80_strength,
        if rec.jpeg_q80_strength <= rec.max_invisible { " ✓" } else { " (may be visible)" },
    )
}

/// Format metrics as CSV header.
pub fn metrics_csv_header() -> String {
    "width,height,prime,mean_intensity,std_intensity,min_intensity,max_intensity,\
     local_variance,entropy,edge_density,dc_component,low_freq_energy,total_energy,\
     low_freq_ratio,spectral_flatness,watermark_region_energy,watermark_region_snr".to_string()
}

/// Format metrics as CSV row.
pub fn metrics_to_csv(m: &ImageMetrics) -> String {
    format!(
        "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        m.width, m.height, m.prime,
        m.mean_intensity, m.std_intensity, m.min_intensity, m.max_intensity,
        m.local_variance, m.entropy, m.edge_density,
        m.dc_component, m.low_freq_energy, m.total_energy,
        m.low_freq_ratio, m.spectral_flatness,
        m.watermark_region_energy, m.watermark_region_snr
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // Uniform image should have low entropy
        let pixels = vec![128u8; 64];
        let entropy = compute_entropy(&pixels);
        assert!(entropy < 0.01, "Uniform image should have ~0 entropy");
    }

    #[test]
    fn test_entropy_random() {
        // Random-ish image should have higher entropy
        let pixels: Vec<u8> = (0..256).cycle().take(1024).map(|v| v as u8).collect();
        let entropy = compute_entropy(&pixels);
        assert!(entropy > 7.0, "Random image should have ~8 bits entropy");
    }

    #[test]
    fn test_edge_density() {
        // Flat image should have zero edges
        let pixels = vec![0.5; 64];
        let density = compute_edge_density(&pixels, 8, 8);
        assert!(density < 0.01, "Flat image should have no edges");
    }

    #[test]
    fn test_analyze_image() {
        let pixels = vec![128u8; 64];
        let metrics = analyze_image(&pixels, 8, 8);
        assert_eq!(metrics.width, 8);
        assert_eq!(metrics.height, 8);
        assert!(metrics.entropy < 0.01);
    }
}
