//! Watermark embedding, detection, and analysis library.
//!
//! This is a Rust port of the C++ watermarking implementation.
//! Watermarks are embedded and detected in the frequency domain using Legendre sequences.
//!
//! ## Embedding
//! Use [`embed_watermark`] to embed a message into an image's frequency domain.
//! Use [`embed_watermark_with_options`] for rotation-invariant embedding with pilot signal.
//!
//! ## Detection
//! Use [`detect_blind`] to detect watermarks without the original image.
//! Use [`detect_blind_with_rotation`] for rotation-corrected detection using pilot signal.
//!
//! ## Pilot Signal
//! The pilot signal is a sinusoidal pattern on a circular ring in the DFT domain.
//! It survives rotation and reveals the rotation angle, enabling correction.
//!
//! ## Analysis
//! Use [`analyze_image`] to compute texture and frequency metrics for robustness prediction.

mod legendre;
mod dft;
mod correlation;
mod detection;
mod embedding;
mod analysis;
mod pilot;
mod markers;

pub use legendre::generate_array;
pub use dft::{extract_mark, dft_2d};
pub use correlation::Correlator2D;
pub use detection::{
    ascii_to_shifts, detect_blind, detect_blind_with_rotation, peak_to_rms,
    DetectionResult, DetectOptions, ExtendedDetectionResult,
};
pub use embedding::{embed_watermark, embed_watermark_with_options, EmbedResult, EmbedOptions};
pub use analysis::{
    analyze_image, ImageMetrics, metrics_csv_header, metrics_to_csv,
    recommend_strength, recommend_strength_f64, format_recommendation, StrengthRecommendation,
};
pub use pilot::{PilotConfig, PilotDetectionResult, embed_pilot, detect_pilot, detect_rotation_bruteforce, detect_rotation_bruteforce_around, rotate_image};
pub use markers::{MarkerConfig, MarkerDetectionResult, embed_markers, detect_markers, correct_perspective};

/// Find the largest prime p such that p <= min(width, height) - 2.
///
/// We subtract 2 to avoid the DC components at row/col 0 in the frequency domain.
/// This matches the C++ implementation and ensures embedding/detection use the same prime.
pub fn largest_prime_for(width: usize, height: usize) -> usize {
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
    if n.is_multiple_of(2) { return false; }
    let sqrt = (n as f64).sqrt() as usize;
    for i in (3..=sqrt).step_by(2) {
        if n.is_multiple_of(i) { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_prime() {
        // Now subtracts 2 from min dimension before finding prime
        assert_eq!(largest_prime_for(100, 100), 97);   // 100-2=98, largest prime <= 98 is 97
        assert_eq!(largest_prime_for(800, 600), 593);  // 600-2=598, largest prime <= 598 is 593
        assert_eq!(largest_prime_for(1920, 1080), 1069); // 1080-2=1078, largest prime <= 1078 is 1069
    }
}
