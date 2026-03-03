//! Blind watermark detection algorithm.
//!
//! Detects watermarks without access to the original image by:
//! 1. Extracting frequency coefficients from the image
//! 2. Correlating with Legendre sequences
//! 3. Finding peak/RMS ratio above threshold
//! 4. Decoding shift values to ASCII message
//!
//! Optionally uses pilot signal for rotation-invariant detection.

use num_bigint::BigUint;
use num_traits::Zero;

use crate::{extract_mark, generate_array, Correlator2D, largest_prime_for};
use crate::pilot::{PilotConfig, detect_pilot, detect_rotation_bruteforce, detect_rotation_bruteforce_around, rotate_image};
use crate::markers::{MarkerConfig, detect_markers, correct_perspective};

/// Result of watermark detection.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub detected: bool,
    pub message: String,
    pub sequences_decoded: usize,
    pub min_confidence: f64,
    pub shifts: Vec<usize>,
}

/// Extended detection result including rotation and perspective correction info.
#[derive(Debug, Clone)]
pub struct ExtendedDetectionResult {
    /// Basic detection result
    pub result: DetectionResult,
    /// Whether rotation correction was applied
    pub rotation_corrected: bool,
    /// Detected rotation angle (if corrected)
    pub rotation_angle: f64,
    /// Pilot signal detection confidence
    pub pilot_confidence: f64,
    /// Whether perspective correction was applied
    pub perspective_corrected: bool,
    /// Marker detection confidence (if perspective corrected)
    pub marker_confidence: f64,
}

/// Options for watermark detection.
#[derive(Debug, Clone)]
pub struct DetectOptions {
    /// Peak/RMS threshold for detection (typically 4-10)
    pub threshold: f64,
    /// Maximum number of Legendre sequences to try
    pub max_sequences: usize,
    /// Whether to use pilot signal for rotation correction
    pub use_pilot: bool,
    /// Configuration for the pilot signal
    pub pilot_config: PilotConfig,
    /// Whether to use markers for perspective correction (print-camera)
    pub use_markers: bool,
    /// Configuration for corner markers
    pub marker_config: MarkerConfig,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            threshold: 6.0,
            max_sequences: 100,
            use_pilot: true,
            pilot_config: PilotConfig::default(),
            use_markers: false,
            marker_config: MarkerConfig::default(),
        }
    }
}

impl DetectOptions {
    /// Create options with specified threshold and default pilot settings.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Create options without pilot signal.
    pub fn without_pilot(threshold: f64, max_sequences: usize) -> Self {
        Self {
            threshold,
            max_sequences,
            use_pilot: false,
            pilot_config: PilotConfig::default(),
            use_markers: false,
            marker_config: MarkerConfig::default(),
        }
    }

    /// Create options for print-camera detection (perspective + rotation correction).
    pub fn for_print_camera(threshold: f64) -> Self {
        Self {
            threshold,
            max_sequences: 100,
            use_pilot: true,
            pilot_config: PilotConfig::default(),
            use_markers: true,
            marker_config: MarkerConfig::default(),
        }
    }
}

/// Detect watermark blindly (without original image).
///
/// # Arguments
/// * `pixels` - Grayscale pixel values in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `threshold` - Peak/RMS threshold for detection (typically 4-10)
/// * `max_sequences` - Maximum Legendre sequences to try
pub fn detect_blind(
    pixels: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
    max_sequences: usize,
) -> DetectionResult {
    let p = largest_prime_for(width, height);

    // Extract watermark from frequency domain
    let extracted = extract_mark(pixels, height, width, p);

    // Correlate with Legendre sequences
    let correlator = Correlator2D::new(p, p);
    let mut shifts = Vec::new();
    let mut min_confidence = f64::MAX;

    for k in 1..=max_sequences {
        let wm_array = generate_array(p, k);
        let correlation = correlator.correlate(&extracted, &wm_array);

        let (peak_rms, max_idx) = peak_to_rms(&correlation);

        if peak_rms < threshold {
            // Below threshold, stop searching
            break;
        }

        // Use raw peak index as shift value (matching C++ behavior)
        // The C++ code uses: shift = maxY * p + maxX (linear index)
        shifts.push(max_idx);
        min_confidence = min_confidence.min(peak_rms);
    }

    if shifts.is_empty() {
        return DetectionResult {
            detected: false,
            message: String::new(),
            sequences_decoded: 0,
            min_confidence: 0.0,
            shifts: vec![],
        };
    }

    // Decode shifts to ASCII
    let p_sq = p * p;
    let message = shifts_to_ascii(&shifts, p_sq);

    DetectionResult {
        detected: true,
        message,
        sequences_decoded: shifts.len(),
        min_confidence,
        shifts,
    }
}

/// Detect watermark with rotation correction via pilot signal.
///
/// This function first attempts direct detection. If that fails and `use_pilot`
/// is enabled in options, it detects the rotation angle from the pilot signal,
/// rotates the image to correct it, and re-attempts detection.
///
/// # Arguments
/// * `pixels` - Grayscale pixel values in [0, 1]
/// * `width` - Image width
/// * `height` - Image height
/// * `options` - Detection options including pilot configuration
///
/// # Returns
/// Extended detection result with rotation correction information.
pub fn detect_blind_with_rotation(
    pixels: &[f64],
    width: usize,
    height: usize,
    options: &DetectOptions,
) -> ExtendedDetectionResult {
    let p = largest_prime_for(width, height);

    // First, try direct detection without rotation correction
    let direct_result = detect_blind(
        pixels,
        width,
        height,
        options.threshold,
        options.max_sequences,
    );

    // Try perspective correction first if markers are enabled
    let (working_pixels, perspective_corrected, marker_confidence) = if options.use_markers {
        let marker_result = detect_markers(pixels, width, height, &options.marker_config);
        if marker_result.all_detected {
            let corrected = correct_perspective(pixels, width, height, &marker_result);
            (corrected, true, marker_result.overall_confidence)
        } else {
            (pixels.to_vec(), false, marker_result.overall_confidence)
        }
    } else {
        (pixels.to_vec(), false, 0.0)
    };

    // Accept direct result only if it's high quality:
    // - Confidence is well above threshold (at least 1.5x)
    // - Or we're not using pilot correction
    let high_quality = direct_result.min_confidence >= options.threshold * 1.5;

    if direct_result.detected && (high_quality || !options.use_pilot) {
        return ExtendedDetectionResult {
            result: direct_result,
            rotation_corrected: false,
            rotation_angle: 0.0,
            pilot_confidence: 0.0,
            perspective_corrected,
            marker_confidence,
        };
    }

    // Direct detection failed - try rotation correction if enabled
    if !options.use_pilot {
        return ExtendedDetectionResult {
            result: direct_result,
            rotation_corrected: false,
            rotation_angle: 0.0,
            pilot_confidence: 0.0,
            perspective_corrected,
            marker_confidence,
        };
    }

    // First try pilot signal detection for rotation angle
    let pilot_result = detect_pilot(&working_pixels, width, height, p, &options.pilot_config);

    // Try correction with pilot-detected angle first
    if pilot_result.detected && pilot_result.rotation_angle.abs() > 0.01 {
        let corrected_pixels = rotate_image(&working_pixels, width, height, -pilot_result.rotation_angle);
        let corrected_result = detect_blind(
            &corrected_pixels,
            width,
            height,
            options.threshold,
            options.max_sequences,
        );

        if corrected_result.detected {
            return ExtendedDetectionResult {
                result: corrected_result,
                rotation_corrected: true,
                rotation_angle: pilot_result.rotation_angle,
                pilot_confidence: pilot_result.confidence,
                perspective_corrected,
                marker_confidence,
            };
        }
    }

    // Pilot didn't help - fall back to brute-force angle search
    // First do a coarse search ±5 degrees with 0.2 degree steps
    let (coarse_angle, coarse_confidence) = detect_rotation_bruteforce(
        &working_pixels, width, height, 5.0, 0.2
    );

    // If we found a promising angle, refine it with finer steps
    if coarse_confidence > options.threshold * 0.8 {
        // Fine search ±0.3 degrees around the coarse angle with 0.02 degree steps
        let fine_range = 0.3;
        let fine_step = 0.02;
        let (fine_angle, fine_confidence) = detect_rotation_bruteforce_around(
            &working_pixels, width, height, coarse_angle, fine_range, fine_step
        );

        if fine_confidence > options.threshold && fine_angle.abs() > 0.01 {
            let corrected_pixels = rotate_image(&working_pixels, width, height, fine_angle);
            let corrected_result = detect_blind(
                &corrected_pixels,
                width,
                height,
                options.threshold,
                options.max_sequences,
            );

            if corrected_result.detected {
                return ExtendedDetectionResult {
                    result: corrected_result,
                    rotation_corrected: true,
                    rotation_angle: -fine_angle,
                    pilot_confidence: fine_confidence,
                    perspective_corrected,
                    marker_confidence,
                };
            }
        }
    }

    // Nothing worked
    ExtendedDetectionResult {
        result: direct_result,
        rotation_corrected: false,
        rotation_angle: 0.0,
        pilot_confidence: pilot_result.confidence,
        perspective_corrected,
        marker_confidence,
    }
}

/// Calculate peak-to-RMS ratio for a correlation surface.
///
/// Returns (peak_rms, max_index)
pub fn peak_to_rms(data: &[f64]) -> (f64, usize) {
    if data.is_empty() {
        return (0.0, 0);
    }

    // Find max and its index
    let (max_idx, &max_val) = data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Calculate RMS (excluding the peak)
    let sum_sq: f64 = data.iter()
        .enumerate()
        .filter(|(i, _)| *i != max_idx)
        .map(|(_, &v)| v * v)
        .sum();

    let rms = (sum_sq / (data.len() - 1) as f64).sqrt();

    let peak_rms = if rms > 1e-10 { max_val / rms } else { 0.0 };

    (peak_rms, max_idx)
}

/// Convert shift values to ASCII string.
///
/// Shifts are stored in LSB-first order (matching C++ getASCII).
/// The big integer is: shifts[0] + shifts[1]*p² + shifts[2]*p⁴ + ...
///
/// Uses arbitrary-precision arithmetic to handle messages of any length.
fn shifts_to_ascii(shifts: &[usize], p_sq: usize) -> String {
    if shifts.is_empty() {
        return String::new();
    }

    // Reconstruct the big integer from shifts (LSB-first, base p²)
    // n = shifts[0] + shifts[1]*p² + shifts[2]*p⁴ + ...
    let base = BigUint::from(p_sq);
    let mut value = BigUint::zero();
    let mut multiplier = BigUint::from(1u32);

    for &shift in shifts {
        value += BigUint::from(shift) * &multiplier;
        multiplier *= &base;
    }

    // Extract 7-bit ASCII characters (LSB first, then reverse)
    let mut chars = Vec::new();
    let mask = BigUint::from(0x7Fu32);

    while !value.is_zero() {
        let c_val = &value & &mask;
        let c = c_val.iter_u64_digits().next().unwrap_or(0) as u8;
        if (32..127).contains(&c) {
            chars.push(c as char);
        }
        value >>= 7usize;
    }

    chars.reverse();
    chars.into_iter().collect()
}

/// Convert ASCII string to shift values for encoding.
///
/// Produces LSB-first shifts to match C++ getShifts behavior.
/// Uses arbitrary-precision arithmetic for messages of any length.
pub fn ascii_to_shifts(message: &str, p_sq: usize, num_shifts: usize) -> Vec<usize> {
    // Pack characters into a big integer (7 bits each, MSB first)
    let mut value = BigUint::zero();
    for c in message.chars() {
        value <<= 7;
        value |= BigUint::from(c as u8 & 0x7F);
    }

    // Convert to base p² (LSB first)
    let base = BigUint::from(p_sq);
    let mut shifts = Vec::new();
    for _ in 0..num_shifts {
        let remainder = &value % &base;
        let shift = remainder.iter_u64_digits().next().unwrap_or(0) as usize;
        shifts.push(shift);
        value /= &base;
    }

    shifts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_to_rms() {
        let data = vec![1.0, 2.0, 10.0, 1.0, 1.0];
        let (ratio, idx) = peak_to_rms(&data);
        assert_eq!(idx, 2);
        assert!(ratio > 5.0); // Peak is 10, RMS of others is ~1.3
    }

    #[test]
    fn test_ascii_roundtrip() {
        let p_sq = 787 * 787;
        let message = "Hello";
        let shifts = ascii_to_shifts(message, p_sq, 3);
        let decoded = shifts_to_ascii(&shifts, p_sq);
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_ascii_roundtrip_longer() {
        let p_sq = 100 * 100;
        let message = "test123";
        let shifts = ascii_to_shifts(message, p_sq, 5);
        let decoded = shifts_to_ascii(&shifts, p_sq);
        assert_eq!(decoded, message);
    }
}
