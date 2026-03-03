//! Watermark robustness experiment runner.
//!
//! Tests watermark survival across multiple images, degradation types, and strength levels.
//! Outputs CSV for analysis.
//!
//! Usage: experiment <image_dir> <output_csv> [--strengths 1,2,4] [--message "0xTEST:42"]

use anyhow::{Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, ImageFormat};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use watermark_detection::{
    analyze_image, detect_blind, embed_watermark, metrics_csv_header, metrics_to_csv,
    largest_prime_for, ImageMetrics,
};

#[derive(Parser)]
#[command(name = "experiment", about = "Watermark robustness experiment runner")]
struct Args {
    /// Directory containing test images
    image_dir: PathBuf,

    /// Output CSV file for results
    output_csv: PathBuf,

    /// Comma-separated list of strength values to test
    #[arg(short, long, default_value = "1,2,4,8")]
    strengths: String,

    /// Message to embed
    #[arg(short, long, default_value = "0xDEADBEEF1234567890ABCDEF:42")]
    message: String,

    /// Detection threshold
    #[arg(short, long, default_value = "4.0")]
    threshold: f64,

    /// Temp directory for intermediate files
    #[arg(long, default_value = "/tmp/watermark-experiment")]
    temp_dir: PathBuf,

    /// Max sequences to try during detection
    #[arg(long, default_value = "20")]
    max_sequences: usize,
}

/// Degradation types to test
#[derive(Debug, Clone)]
enum Degradation {
    None,
    Jpeg(u8),      // quality
    Blur(f32),     // sigma
    Noise(u32),    // amount
    Brightness(i32), // delta
    Crop(u32),     // percent to keep
}

impl Degradation {
    fn name(&self) -> String {
        match self {
            Degradation::None => "none".to_string(),
            Degradation::Jpeg(q) => format!("jpeg_{}", q),
            Degradation::Blur(s) => format!("blur_{}", s),
            Degradation::Noise(n) => format!("noise_{}", n),
            Degradation::Brightness(b) => format!("bright_{}", b),
            Degradation::Crop(p) => format!("crop_{}", p),
        }
    }

    fn all() -> Vec<Degradation> {
        vec![
            Degradation::None,
            // JPEG compression levels
            Degradation::Jpeg(95),
            Degradation::Jpeg(90),
            Degradation::Jpeg(80),
            Degradation::Jpeg(70),
            Degradation::Jpeg(50),
            // Blur levels
            Degradation::Blur(0.5),
            Degradation::Blur(1.0),
            Degradation::Blur(2.0),
            // Noise levels
            Degradation::Noise(5),
            Degradation::Noise(10),
            Degradation::Noise(25),
            // Brightness adjustments
            Degradation::Brightness(10),
            Degradation::Brightness(-10),
            Degradation::Brightness(20),
            Degradation::Brightness(-20),
            // Crop levels (keep center %)
            Degradation::Crop(95),
            Degradation::Crop(90),
            Degradation::Crop(80),
        ]
    }
}

/// Result of a single experiment trial
struct TrialResult {
    image_name: String,
    strength: f64,
    degradation: String,
    detected: bool,
    correct_message: bool,
    confidence: f64,
    sequences_decoded: usize,
    original_prime: usize,
    degraded_prime: usize,
    embed_time_ms: f64,
    degrade_time_ms: f64,
    detect_time_ms: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Parse strengths
    let strengths: Vec<f64> = args.strengths
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    // Create temp directory
    fs::create_dir_all(&args.temp_dir)?;

    // Find all images
    let images: Vec<PathBuf> = fs::read_dir(&args.image_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| {
                    let ext = ext.to_string_lossy().to_lowercase();
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                })
                .unwrap_or(false)
        })
        .collect();

    println!("========================================");
    println!("   WATERMARK ROBUSTNESS EXPERIMENT");
    println!("========================================\n");
    println!("Images: {} found in {}", images.len(), args.image_dir.display());
    println!("Strengths: {:?}", strengths);
    println!("Degradations: {}", Degradation::all().len());
    println!("Message: \"{}\"", args.message);
    println!("Threshold: {}", args.threshold);
    println!();

    let total_trials = images.len() * strengths.len() * Degradation::all().len();
    println!("Total trials: {}\n", total_trials);

    // Open output CSV
    let file = File::create(&args.output_csv)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "image,strength,degradation,detected,correct_message,confidence,sequences_decoded,\
         original_prime,degraded_prime,embed_time_ms,degrade_time_ms,detect_time_ms,\
         {}",
        metrics_csv_header()
    )?;

    let mut completed = 0;
    let start_time = Instant::now();

    for image_path in &images {
        let image_name = image_path.file_name().unwrap().to_string_lossy().to_string();
        println!("Processing: {}", image_name);

        // Load original image
        let orig_img = image::open(image_path).context("Failed to load image")?;
        let (width, height) = orig_img.dimensions();
        let orig_gray = orig_img.to_luma8();
        let orig_pixels: Vec<u8> = orig_gray.as_raw().clone();

        // Analyze original image
        let metrics = analyze_image(&orig_pixels, width as usize, height as usize);
        let metrics_csv = metrics_to_csv(&metrics);

        for &strength in &strengths {
            // Embed watermark
            let embed_start = Instant::now();
            let embed_result = embed_watermark(
                &orig_pixels,
                width as usize,
                height as usize,
                &args.message,
                strength,
            );
            let embed_time = embed_start.elapsed().as_secs_f64() * 1000.0;

            // Create watermarked image
            let watermarked: GrayImage = ImageBuffer::from_raw(width, height, embed_result.pixels.clone())
                .context("Failed to create watermarked image")?;

            for degradation in Degradation::all() {
                let degrade_start = Instant::now();

                // Apply degradation
                let (degraded_img, degraded_prime) = apply_degradation(
                    &DynamicImage::ImageLuma8(watermarked.clone()),
                    &degradation,
                    &args.temp_dir,
                )?;
                let degrade_time = degrade_start.elapsed().as_secs_f64() * 1000.0;

                // Get degraded pixels
                let (deg_width, deg_height) = degraded_img.dimensions();
                let deg_gray = degraded_img.to_luma8();
                let deg_pixels: Vec<f64> = deg_gray.as_raw().iter()
                    .map(|&v| v as f64 / 255.0)
                    .collect();

                // Detect watermark
                let detect_start = Instant::now();
                let detect_result = detect_blind(
                    &deg_pixels,
                    deg_width as usize,
                    deg_height as usize,
                    args.threshold,
                    args.max_sequences,
                );
                let detect_time = detect_start.elapsed().as_secs_f64() * 1000.0;

                // Check if message is correct
                let correct = detect_result.message == args.message;

                // Record result
                let result = TrialResult {
                    image_name: image_name.clone(),
                    strength,
                    degradation: degradation.name(),
                    detected: detect_result.detected,
                    correct_message: correct,
                    confidence: detect_result.min_confidence,
                    sequences_decoded: detect_result.sequences_decoded,
                    original_prime: embed_result.prime,
                    degraded_prime,
                    embed_time_ms: embed_time,
                    degrade_time_ms: degrade_time,
                    detect_time_ms: detect_time,
                };

                // Write CSV row
                writeln!(
                    writer,
                    "{},{},{},{},{},{:.4},{},{},{},{:.2},{:.2},{:.2},{}",
                    result.image_name,
                    result.strength,
                    result.degradation,
                    result.detected,
                    result.correct_message,
                    result.confidence,
                    result.sequences_decoded,
                    result.original_prime,
                    result.degraded_prime,
                    result.embed_time_ms,
                    result.degrade_time_ms,
                    result.detect_time_ms,
                    metrics_csv
                )?;

                completed += 1;

                // Progress indicator
                if completed % 50 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let rate = completed as f64 / elapsed;
                    let remaining = (total_trials - completed) as f64 / rate;
                    print!("\r  Progress: {}/{} ({:.1}%), ETA: {:.0}s    ",
                           completed, total_trials,
                           100.0 * completed as f64 / total_trials as f64,
                           remaining);
                    std::io::stdout().flush()?;
                }
            }
        }
        println!();
    }

    writer.flush()?;

    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n========================================");
    println!("           EXPERIMENT COMPLETE");
    println!("========================================\n");
    println!("Results saved to: {}", args.output_csv.display());
    println!("Total trials: {}", completed);
    println!("Total time: {:.1}s ({:.1} trials/sec)", total_time, completed as f64 / total_time);

    Ok(())
}

/// Apply a degradation to an image.
fn apply_degradation(
    img: &DynamicImage,
    degradation: &Degradation,
    temp_dir: &PathBuf,
) -> Result<(DynamicImage, usize)> {
    let (width, height) = img.dimensions();

    let result = match degradation {
        Degradation::None => img.clone(),

        Degradation::Jpeg(quality) => {
            // Save as JPEG and reload
            let temp_path = temp_dir.join("temp.jpg");
            {
                let file = File::create(&temp_path)?;
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, *quality);
                encoder.encode_image(img)?;
            }
            image::open(&temp_path)?
        }

        Degradation::Blur(sigma) => {
            img.blur(*sigma)
        }

        Degradation::Noise(amount) => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let rgba = img.to_rgba8();
            let mut noisy: image::RgbaImage = image::ImageBuffer::new(width, height);
            let noise_scale = *amount as f32 / 100.0 * 50.0;

            for (x, y, pixel) in rgba.enumerate_pixels() {
                let noise: i32 = ((rng.gen::<f32>() * 2.0 - 1.0) * noise_scale) as i32;
                let r = (pixel[0] as i32 + noise).clamp(0, 255) as u8;
                let g = (pixel[1] as i32 + noise).clamp(0, 255) as u8;
                let b = (pixel[2] as i32 + noise).clamp(0, 255) as u8;
                noisy.put_pixel(x, y, image::Rgba([r, g, b, pixel[3]]));
            }
            DynamicImage::ImageRgba8(noisy)
        }

        Degradation::Brightness(delta) => {
            img.brighten(*delta)
        }

        Degradation::Crop(percent) => {
            let keep = *percent as f32 / 100.0;
            let new_width = (width as f32 * keep) as u32;
            let new_height = (height as f32 * keep) as u32;
            let x = (width - new_width) / 2;
            let y = (height - new_height) / 2;
            img.crop_imm(x, y, new_width, new_height)
        }
    };

    let (new_width, new_height) = result.dimensions();
    let prime = largest_prime_for(new_width as usize, new_height as usize);

    Ok((result, prime))
}
