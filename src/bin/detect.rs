//! Blind watermark detection CLI.
//!
//! Usage: detect <image_path> [--threshold <value>] [--max-sequences <value>] [--pilot]

use anyhow::{Context, Result};
use clap::Parser;
use image::GenericImageView;
use std::path::PathBuf;
use std::time::Instant;
use watermark_detection::{detect_blind, detect_blind_with_rotation, largest_prime_for, DetectOptions};

#[derive(Parser)]
#[command(name = "detect", about = "Blind watermark detection")]
struct Args {
    /// Path to the image file
    image_path: PathBuf,

    /// Detection threshold (higher = stricter)
    #[arg(short, long, default_value = "6.0")]
    threshold: f64,

    /// Maximum Legendre sequences to try
    #[arg(short, long, default_value = "100")]
    max_sequences: usize,

    /// Use pilot signal for rotation-invariant detection
    #[arg(long, default_value_t = true)]
    pilot: bool,

    /// Disable pilot-based rotation correction
    #[arg(long, conflicts_with = "pilot")]
    no_pilot: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let total_start = Instant::now();

    println!("========================================");
    println!("     BLIND WATERMARK DETECTION");
    println!("========================================\n");
    let use_pilot = args.pilot && !args.no_pilot;

    println!("Image: {}", args.image_path.display());
    println!("Threshold: {}", args.threshold);
    println!("Pilot correction: {}\n", if use_pilot { "enabled" } else { "disabled" });

    // Load image
    let img = image::open(&args.image_path).context("Failed to open image")?;
    let (width, height) = img.dimensions();
    println!("Image loaded: {}x{}", width, height);

    let p = largest_prime_for(width as usize, height as usize);
    println!("Largest prime: {}", p);
    println!("Watermark coefficients: {} x {} = {}\n", p, p, p * p);

    // Convert to grayscale (using max of RGB as "value")
    let luma: Vec<f64> = img.pixels().map(|(_, _, pixel)| {
        let [r, g, b, _] = pixel.0;
        r.max(g).max(b) as f64 / 255.0
    }).collect();

    println!("Detecting watermark...\n");

    // Run detection
    let detected;
    if use_pilot {
        let options = DetectOptions {
            threshold: args.threshold,
            max_sequences: args.max_sequences,
            use_pilot: true,
            ..Default::default()
        };

        let extended_result = detect_blind_with_rotation(
            &luma,
            width as usize,
            height as usize,
            &options,
        );

        println!("========================================");
        println!("              RESULTS");
        println!("========================================\n");

        if extended_result.result.detected {
            println!("Watermark detected: YES");
            println!("Message: \"{}\"", extended_result.result.message);
            println!("Sequences decoded: {}", extended_result.result.sequences_decoded);
            println!("Minimum confidence: {:.2}", extended_result.result.min_confidence);
            println!("Shifts: {:?}", extended_result.result.shifts);

            if extended_result.rotation_corrected {
                println!("\nRotation correction applied:");
                println!("  Rotation angle: {:.4}°", extended_result.rotation_angle);
                println!("  Pilot confidence: {:.2}", extended_result.pilot_confidence);
            }
        } else {
            println!("Watermark detected: NO");
            if extended_result.pilot_confidence > 0.0 {
                println!("  Pilot confidence: {:.2}", extended_result.pilot_confidence);
            }
        }

        detected = extended_result.result.detected;
    } else {
        let result = detect_blind(
            &luma,
            width as usize,
            height as usize,
            args.threshold,
            args.max_sequences,
        );

        println!("========================================");
        println!("              RESULTS");
        println!("========================================\n");

        if result.detected {
            println!("Watermark detected: YES");
            println!("Message: \"{}\"", result.message);
            println!("Sequences decoded: {}", result.sequences_decoded);
            println!("Minimum confidence: {:.2}", result.min_confidence);
            println!("Shifts: {:?}", result.shifts);
        } else {
            println!("Watermark detected: NO");
        }

        detected = result.detected;
    }

    println!("\nTiming:");
    println!("  Total: {:.0} ms\n", total_start.elapsed().as_secs_f64() * 1000.0);

    std::process::exit(if detected { 0 } else { 1 });
}
