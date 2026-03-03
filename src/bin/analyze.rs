//! Image analysis CLI for watermark robustness prediction.
//!
//! Usage: analyze <image_path>

use anyhow::{Context, Result};
use clap::Parser;
use image::GenericImageView;
use std::path::PathBuf;
use watermark_detection::{
    analyze_image, ImageMetrics, recommend_strength, format_recommendation, StrengthRecommendation,
};

#[derive(Parser)]
#[command(name = "analyze", about = "Analyze image for watermark robustness")]
struct Args {
    /// Path to the image file
    image_path: PathBuf,

    /// Output as JSON instead of human-readable
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let img = image::open(&args.image_path).context("Failed to open image")?;
    let (width, height) = img.dimensions();

    // Convert to grayscale
    let gray = img.to_luma8();
    let pixels: Vec<u8> = gray.as_raw().clone();

    let metrics = analyze_image(&pixels, width as usize, height as usize);
    let strength_rec = recommend_strength(&pixels, width as usize, height as usize);

    if args.json {
        print_json(&metrics, &strength_rec);
    } else {
        print_human(&args.image_path, &metrics, &strength_rec);
    }

    Ok(())
}

fn print_human(path: &PathBuf, m: &ImageMetrics, rec: &StrengthRecommendation) {
    println!("========================================");
    println!("       IMAGE TEXTURE ANALYSIS");
    println!("========================================\n");
    println!("Image: {}", path.display());
    println!("Size: {}x{}", m.width, m.height);
    println!("Prime (p): {}", m.prime);
    println!("Watermark capacity: {} coefficients\n", m.prime * m.prime);

    println!("--- Intensity Statistics ---");
    println!("  Mean:     {:.3}", m.mean_intensity);
    println!("  Std Dev:  {:.3}", m.std_intensity);
    println!("  Range:    [{:.3}, {:.3}]", m.min_intensity, m.max_intensity);
    println!();

    println!("--- Texture Metrics ---");
    println!("  Local Variance:  {:.6} (higher = more texture)", m.local_variance);
    println!("  Entropy:         {:.2} bits/pixel (max 8)", m.entropy);
    println!("  Edge Density:    {:.1}%", m.edge_density * 100.0);
    println!();

    println!("--- Frequency Analysis ---");
    println!("  DC Component:         {:.2}", m.dc_component);
    println!("  Low-Freq Energy:      {:.2}", m.low_freq_energy);
    println!("  Total Energy:         {:.2}", m.total_energy);
    println!("  Low-Freq Ratio:       {:.4} (higher = more interference)", m.low_freq_ratio);
    println!("  Spectral Flatness:    {:.4} (0=tonal, 1=noise)", m.spectral_flatness);
    println!();

    println!("--- Watermark Region ---");
    println!("  Region Energy:  {:.2}", m.watermark_region_energy);
    println!("  Region SNR:     {:.2}", m.watermark_region_snr);
    println!();

    // Strength recommendation
    println!("========================================");
    println!("     WATERMARK STRENGTH RECOMMENDATION");
    println!("========================================\n");
    println!("{}", format_recommendation(rec));
    println!();

    // Summary
    let can_survive_jpeg90 = rec.jpeg_q90_strength <= rec.max_invisible;
    let can_survive_jpeg80 = rec.jpeg_q80_strength <= rec.max_invisible;

    if can_survive_jpeg80 {
        println!("✓ EXCELLENT: Can use strength {:.1} for JPEG Q80 robustness", rec.jpeg_q80_strength);
    } else if can_survive_jpeg90 {
        println!("○ GOOD: Can use strength {:.1} for JPEG Q90 robustness", rec.jpeg_q90_strength);
    } else if rec.max_invisible >= 2.0 {
        println!("△ MARGINAL: Use strength {:.1}, PNG only (JPEG will destroy watermark)", rec.recommended);
    } else {
        println!("✗ POOR: Very flat image, watermark may be visible even at low strength");
    }
}

fn print_json(m: &ImageMetrics, rec: &StrengthRecommendation) {
    println!(r#"{{"#);
    println!(r#"  "width": {},"#, m.width);
    println!(r#"  "height": {},"#, m.height);
    println!(r#"  "prime": {},"#, m.prime);
    println!(r#"  "mean_intensity": {:.6},"#, m.mean_intensity);
    println!(r#"  "std_intensity": {:.6},"#, m.std_intensity);
    println!(r#"  "local_variance": {:.6},"#, m.local_variance);
    println!(r#"  "entropy": {:.6},"#, m.entropy);
    println!(r#"  "edge_density": {:.6},"#, m.edge_density);
    println!(r#"  "dc_component": {:.6},"#, m.dc_component);
    println!(r#"  "low_freq_energy": {:.6},"#, m.low_freq_energy);
    println!(r#"  "total_energy": {:.6},"#, m.total_energy);
    println!(r#"  "low_freq_ratio": {:.6},"#, m.low_freq_ratio);
    println!(r#"  "spectral_flatness": {:.6},"#, m.spectral_flatness);
    println!(r#"  "watermark_region_energy": {:.6},"#, m.watermark_region_energy);
    println!(r#"  "watermark_region_snr": {:.6},"#, m.watermark_region_snr);
    println!(r#"  "predicted_hideability": {:.4},"#, predict_hideability(m));
    println!(r#"  "predicted_robustness": {:.4},"#, predict_robustness(m));
    println!(r#"  "strength": {{"#);
    println!(r#"    "min_detectable": {:.2},"#, rec.min_detectable);
    println!(r#"    "max_invisible": {:.2},"#, rec.max_invisible);
    println!(r#"    "recommended": {:.2},"#, rec.recommended);
    println!(r#"    "jpeg_q90_strength": {:.2},"#, rec.jpeg_q90_strength);
    println!(r#"    "jpeg_q80_strength": {:.2},"#, rec.jpeg_q80_strength);
    println!(r#"    "texture_score": {:.4},"#, rec.texture_score);
    println!(r#"    "flat_region_percent": {:.2},"#, rec.flat_region_percent);
    println!(r#"    "avg_nvf": {:.4},"#, rec.avg_nvf);
    println!(r#"    "min_nvf": {:.4}"#, rec.min_nvf);
    println!(r#"  }}"#);
    println!(r#"}}"#);
}

/// Predict how well a watermark can be hidden (visually) in this image.
/// Higher texture and edge density = easier to hide.
fn predict_hideability(m: &ImageMetrics) -> f64 {
    // Factors that help hiding:
    // - High local variance (texture hides modifications)
    // - High entropy (complex patterns hide changes)
    // - High edge density (edges mask artifacts)
    // - Not too dark or bright (mid-tones hide better)

    let variance_score = (m.local_variance * 100.0).min(1.0);
    let entropy_score = (m.entropy / 8.0).min(1.0);
    let edge_score = (m.edge_density * 5.0).min(1.0);

    // Penalize extreme intensities
    let intensity_score = 1.0 - (m.mean_intensity - 0.5).abs() * 2.0;

    (variance_score * 0.3 + entropy_score * 0.3 + edge_score * 0.2 + intensity_score * 0.2)
        .clamp(0.0, 1.0)
}

/// Predict detection robustness after degradation.
/// Lower spectral flatness and lower low_freq_ratio = more robust.
fn predict_robustness(m: &ImageMetrics) -> f64 {
    // Factors that help robustness:
    // - Low spectral flatness (concentrated energy, cleaner correlation)
    // - Low low_freq_ratio (less interference in watermark band)
    // - High watermark_region_snr (more room for watermark signal)

    let flatness_score = 1.0 - m.spectral_flatness.clamp(0.0, 1.0);
    let ratio_score = 1.0 - (m.low_freq_ratio * 2.0).clamp(0.0, 1.0);
    let snr_score = (m.watermark_region_snr / 10.0).clamp(0.0, 1.0);

    (flatness_score * 0.3 + ratio_score * 0.4 + snr_score * 0.3)
        .clamp(0.0, 1.0)
}

fn rating_bar(value: f64) -> String {
    let filled = (value * 10.0).round() as usize;
    let empty = 10 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
