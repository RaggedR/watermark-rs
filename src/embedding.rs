//! Watermark embedding algorithm.
//!
//! Embeds watermarks in the frequency domain by:
//! 1. Converting message to shift values (base p²)
//! 2. Generating shifted Legendre sequence patterns
//! 3. Adding patterns to low-frequency DFT coefficients
//! 4. Inverse DFT to get watermarked image
//!
//! Optionally embeds a pilot signal for rotation-invariant detection.

use num_bigint::BigUint;
use num_complex::Complex64;
use num_traits::Zero;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

use crate::generate_array;
use crate::pilot::{PilotConfig, embed_pilot};
use crate::markers::{MarkerConfig, embed_markers};

/// Result of watermark embedding.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// The watermarked pixel values (grayscale, 0-255)
    pub pixels: Vec<u8>,
    /// Number of Legendre sequences used
    pub sequences_used: usize,
    /// The prime used for the watermark array
    pub prime: usize,
    /// The shift values encoded
    pub shifts: Vec<usize>,
    /// Whether a pilot signal was embedded
    pub pilot_embedded: bool,
    /// Whether corner markers were embedded
    pub markers_embedded: bool,
}

/// Options for watermark embedding.
#[derive(Debug, Clone)]
pub struct EmbedOptions {
    /// Watermark strength (1-10 typical, higher = more robust but more visible)
    pub strength: f64,
    /// Whether to embed a pilot signal for rotation correction
    pub embed_pilot: bool,
    /// Configuration for the pilot signal
    pub pilot_config: PilotConfig,
    /// Whether to embed corner markers for perspective correction
    pub embed_markers: bool,
    /// Configuration for corner markers
    pub marker_config: MarkerConfig,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            strength: 2.0,
            embed_pilot: true,
            pilot_config: PilotConfig::default(),
            embed_markers: false,
            marker_config: MarkerConfig::default(),
        }
    }
}

impl EmbedOptions {
    /// Create options with specified strength and default pilot settings.
    pub fn with_strength(strength: f64) -> Self {
        Self {
            strength,
            ..Default::default()
        }
    }

    /// Create options without pilot signal.
    pub fn without_pilot(strength: f64) -> Self {
        Self {
            strength,
            embed_pilot: false,
            pilot_config: PilotConfig::default(),
            embed_markers: false,
            marker_config: MarkerConfig::default(),
        }
    }

    /// Create options for print-camera robustness (pilot + markers).
    pub fn for_print_camera(strength: f64) -> Self {
        Self {
            strength,
            embed_pilot: true,
            pilot_config: PilotConfig::default(),
            embed_markers: true,
            marker_config: MarkerConfig::default(),
        }
    }
}

/// Embed a watermark message into an image.
///
/// This is a convenience wrapper around `embed_watermark_with_options` with default options
/// (no pilot signal for backward compatibility).
///
/// # Arguments
/// * `pixels` - Grayscale pixel values (0-255), row-major order
/// * `width` - Image width
/// * `height` - Image height
/// * `message` - ASCII message to embed (e.g., "0xABC123:42")
/// * `strength` - Watermark strength (1-10 typical, higher = more robust but more visible)
///
/// # Returns
/// The watermarked pixel values and embedding metadata.
pub fn embed_watermark(
    pixels: &[u8],
    width: usize,
    height: usize,
    message: &str,
    strength: f64,
) -> EmbedResult {
    let options = EmbedOptions::without_pilot(strength);
    embed_watermark_with_options(pixels, width, height, message, &options)
}

/// Embed a watermark message into an image with configurable options.
///
/// When `embed_pilot` is enabled in options, a sinusoidal pilot signal is also
/// embedded on a circular ring in the DFT domain. This enables rotation-invariant
/// detection by allowing the detection algorithm to determine and correct for
/// any rotation applied to the image.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values (0-255), row-major order
/// * `width` - Image width
/// * `height` - Image height
/// * `message` - ASCII message to embed (e.g., "0xABC123:42")
/// * `options` - Embedding options including strength and pilot configuration
///
/// # Returns
/// The watermarked pixel values and embedding metadata.
pub fn embed_watermark_with_options(
    pixels: &[u8],
    width: usize,
    height: usize,
    message: &str,
    options: &EmbedOptions,
) -> EmbedResult {
    // Find largest prime for this image size
    // C++ uses min(rows, cols) - 2 to avoid DC components
    let p = largest_prime_for_embed(width, height);
    let p_sq = p * p;

    // Convert pixels to f64 for processing
    let luma: Vec<f64> = pixels.iter().map(|&v| v as f64 / 255.0).collect();

    // Convert message to shift values
    let shifts = message_to_shifts(message, p_sq);

    // Compute forward DFT once
    let mut spectrum = dft_2d_real(&luma, height, width);

    // Embed pilot signal first (if enabled)
    if options.embed_pilot {
        embed_pilot(&mut spectrum, width, height, p, &options.pilot_config);
    }

    // Embed corner markers (if enabled)
    if options.embed_markers {
        // Extract real and imaginary parts for marker embedding
        let mut spectrum_re: Vec<f64> = spectrum.iter().map(|c| c.re).collect();
        let mut spectrum_im: Vec<f64> = spectrum.iter().map(|c| c.im).collect();

        embed_markers(&mut spectrum_re, &mut spectrum_im, width, height, &options.marker_config);

        // Recombine into complex spectrum
        for (i, (re, im)) in spectrum_re.into_iter().zip(spectrum_im.into_iter()).enumerate() {
            spectrum[i] = Complex64::new(re, im);
        }
    }

    // Embed each shift as a Legendre sequence
    for (k, &shift) in shifts.iter().enumerate() {
        // Generate Legendre array for sequence k+1
        let wm_array = generate_array(p, k + 1);

        // Scale by strength
        let scaled: Vec<f64> = wm_array.iter().map(|&v| v * options.strength).collect();

        // Shift the array by the message digit
        let shifted = shift_array(&scaled, p, p, shift);

        // Add watermark to low-frequency region (avoiding DC at 0,0)
        for i in 0..p {
            for j in 0..p {
                let pixel_idx = (i + 1) * width + (j + 1);
                let wm_idx = i * p + j;
                spectrum[pixel_idx] = Complex64::new(
                    spectrum[pixel_idx].re + shifted[wm_idx],
                    spectrum[pixel_idx].im,
                );
            }
        }
    }

    // Inverse DFT
    let result_luma = idft_2d_real(&spectrum, height, width);

    // Convert back to u8, clamping to valid range
    let result_pixels: Vec<u8> = result_luma
        .iter()
        .map(|&v| {
            let scaled = v * 255.0;
            if scaled > 255.0 {
                255
            } else if scaled < 0.0 {
                0
            } else {
                scaled.round() as u8
            }
        })
        .collect();

    EmbedResult {
        pixels: result_pixels,
        sequences_used: shifts.len(),
        prime: p,
        shifts,
        pilot_embedded: options.embed_pilot,
        markers_embedded: options.embed_markers,
    }
}

/// Find largest prime p such that p <= min(width, height) - 2.
/// We subtract 2 to leave room for DC components at row/col 0.
fn largest_prime_for_embed(width: usize, height: usize) -> usize {
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

/// Convert ASCII message to shift values (base p²).
///
/// Each character is 7 bits. Pack all bits into a big integer,
/// then convert to base p² digits.
///
/// Uses arbitrary-precision arithmetic to handle messages of any length
/// (matching C++ boost::multiprecision::cpp_int behavior).
fn message_to_shifts(message: &str, p_sq: usize) -> Vec<usize> {
    if message.is_empty() {
        return vec![];
    }

    // Pack characters into a big integer (7 bits each, MSB first)
    let mut value = BigUint::zero();
    for c in message.chars() {
        value <<= 7;
        value |= BigUint::from(c as u8 & 0x7F);
    }

    // Convert to base p² (least significant digit first, like C++)
    let base = BigUint::from(p_sq);
    let mut shifts = Vec::new();
    while !value.is_zero() {
        let remainder = &value % &base;
        // Convert BigUint to usize - safe because remainder < p_sq < usize::MAX
        let shift: usize = remainder.iter_u64_digits().next().unwrap_or(0) as usize;
        shifts.push(shift);
        value /= &base;
    }

    shifts
}

/// Shift a 2D array by a message value.
///
/// Right shift = message % width, down shift = message / width
fn shift_array(array: &[f64], height: usize, width: usize, message: usize) -> Vec<f64> {
    let v_shift = (message / width) % height;
    let h_shift = message % width;

    let mut shifted = vec![0.0; height * width];

    for i in 0..height {
        let k = (i + v_shift) % height;
        for j in 0..width {
            let l = (j + h_shift) % width;
            shifted[k * width + l] = array[i * width + j];
        }
    }

    shifted
}

/// Compute 2D DFT (real input, complex output).
fn dft_2d_real(pixels: &[f64], height: usize, width: usize) -> Vec<Complex64> {
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

/// Compute 2D inverse DFT (complex input, real output).
fn idft_2d_real(spectrum: &[Complex64], height: usize, width: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let ifft_row: Arc<dyn Fft<f64>> = planner.plan_fft_inverse(width);
    let ifft_col: Arc<dyn Fft<f64>> = planner.plan_fft_inverse(height);

    let mut data = spectrum.to_vec();

    // IFFT on rows
    for row in 0..height {
        let start = row * width;
        let row_slice = &mut data[start..start + width];
        ifft_row.process(row_slice);
    }

    // Transpose
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for y in 0..height {
        for x in 0..width {
            transposed[x * height + y] = data[y * width + x];
        }
    }

    // IFFT on columns (now rows after transpose)
    for col in 0..width {
        let start = col * height;
        let col_slice = &mut transposed[start..start + height];
        ifft_col.process(col_slice);
    }

    // Transpose back and take real part with scaling
    let scale = 1.0 / (width * height) as f64;
    let mut result = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            result[y * width + x] = transposed[x * height + y].re * scale;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_to_shifts() {
        let p_sq = 100 * 100; // 10000
        let shifts = message_to_shifts("A", p_sq);
        // 'A' = 65 = 0b1000001
        assert!(!shifts.is_empty());
        // Value should be 65, which is < 10000, so one shift
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0], 65);
    }

    #[test]
    fn test_message_to_shifts_longer() {
        let p_sq = 100 * 100;
        let shifts = message_to_shifts("Hi", p_sq);
        // 'H' = 72, 'i' = 105
        // value = (72 << 7) | 105 = 9216 + 105 = 9321
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0], 9321);
    }

    #[test]
    fn test_shift_array() {
        let arr = vec![1.0, 2.0, 3.0, 4.0];
        let shifted = shift_array(&arr, 2, 2, 1);
        // shift right by 1: [1,2,3,4] -> [2,1,4,3]
        assert_eq!(shifted, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_embed_roundtrip() {
        // Create a simple test image (8x8 gray)
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();

        let result = embed_watermark(&pixels, 8, 8, "X", 2.0);

        assert_eq!(result.pixels.len(), 64);
        assert!(result.sequences_used > 0);
        assert!(result.prime > 0);
    }

    #[test]
    fn test_largest_prime_for_embed() {
        assert_eq!(largest_prime_for_embed(100, 100), 97); // 100 - 2 = 98, largest prime <= 98 is 97
        assert_eq!(largest_prime_for_embed(800, 600), 593); // 600 - 2 = 598, largest prime <= 598 is 593
    }
}
