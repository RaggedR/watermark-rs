//! Image degradation tool for watermark robustness testing.
//!
//! Usage: degrade <input> <output> <type> [params...]
//!
//! Degradation types:
//!   jpeg <quality>        - JPEG compression (1-100)
//!   resize <percent>      - Resize image (e.g., 50 for 50%)
//!   crop <percent>        - Keep center N% (e.g., 80 keeps center 80%)
//!   blur <sigma>          - Gaussian blur (e.g., 2.0)
//!   noise <amount>        - Add gaussian noise (1-100)
//!   brightness <delta>    - Adjust brightness (-100 to 100)
//!   rotate <degrees>      - Rotate image (1-359)
//!   screenshot            - Simulate screenshot (resize 80% + blur + jpeg 85)
//!   social                - Simulate social media (resize 50% + jpeg 60)

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use image::{ImageBuffer, Rgba, DynamicImage, ImageFormat};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "degrade", about = "Image degradation for watermark testing")]
struct Args {
    /// Input image path
    input: PathBuf,

    /// Output image path
    output: PathBuf,

    #[command(subcommand)]
    degradation: Degradation,
}

#[derive(Subcommand)]
enum Degradation {
    /// JPEG compression
    Jpeg {
        /// Quality (1-100)
        #[arg(default_value = "50")]
        quality: u8,
    },
    /// Resize image
    Resize {
        /// Percentage (e.g., 50 for 50%)
        #[arg(default_value = "50")]
        percent: u32,
    },
    /// Crop to center
    Crop {
        /// Keep center N percent
        #[arg(default_value = "80")]
        percent: u32,
    },
    /// Gaussian blur
    Blur {
        /// Sigma value
        #[arg(default_value = "2.0")]
        sigma: f32,
    },
    /// Add noise
    Noise {
        /// Noise amount (1-100)
        #[arg(default_value = "10")]
        amount: u32,
    },
    /// Adjust brightness
    Brightness {
        /// Delta (-100 to 100)
        #[arg(default_value = "20")]
        delta: i32,
    },
    /// Rotate image
    Rotate {
        /// Degrees (1-359)
        #[arg(default_value = "5")]
        degrees: u32,
    },
    /// Simulate screenshot
    Screenshot,
    /// Simulate social media compression
    Social,
    /// Apply all degradations and save each
    All {
        /// Output directory
        #[arg(default_value = ".")]
        output_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading: {}", args.input.display());
    let img = image::open(&args.input).context("Failed to open input image")?;
    println!("Image size: {}x{}", img.width(), img.height());

    match args.degradation {
        Degradation::All { output_dir } => {
            apply_all_degradations(&img, &args.input, &output_dir)?;
        }
        _ => {
            let degraded = apply_degradation(&img, &args.degradation)?;
            save_image(&degraded, &args.output, &args.degradation)?;
            println!("Saved: {}", args.output.display());
        }
    }

    Ok(())
}

fn apply_degradation(img: &DynamicImage, degradation: &Degradation) -> Result<DynamicImage> {
    match degradation {
        Degradation::Jpeg { quality: _ } => {
            // JPEG degradation happens at save time
            Ok(img.clone())
        }
        Degradation::Resize { percent } => {
            let new_width = (img.width() as f32 * (*percent as f32 / 100.0)) as u32;
            let new_height = (img.height() as f32 * (*percent as f32 / 100.0)) as u32;
            println!("Resizing to {}x{} ({}%)", new_width, new_height, percent);
            Ok(img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3))
        }
        Degradation::Crop { percent } => {
            let keep = *percent as f32 / 100.0;
            let new_width = (img.width() as f32 * keep) as u32;
            let new_height = (img.height() as f32 * keep) as u32;
            let x = (img.width() - new_width) / 2;
            let y = (img.height() - new_height) / 2;
            println!("Cropping to {}x{} (center {}%)", new_width, new_height, percent);
            Ok(img.crop_imm(x, y, new_width, new_height))
        }
        Degradation::Blur { sigma } => {
            println!("Applying blur with sigma={}", sigma);
            Ok(img.blur(*sigma))
        }
        Degradation::Noise { amount } => {
            println!("Adding noise (amount={})", amount);
            add_noise(img, *amount)
        }
        Degradation::Brightness { delta } => {
            println!("Adjusting brightness by {}", delta);
            Ok(img.brighten(*delta))
        }
        Degradation::Rotate { degrees } => {
            println!("Rotating by {} degrees", degrees);
            // image crate only supports 90/180/270, so we approximate
            match degrees {
                0..=45 | 316..=360 => Ok(img.clone()),
                46..=135 => Ok(img.rotate90()),
                136..=225 => Ok(img.rotate180()),
                226..=315 => Ok(img.rotate270()),
                _ => Ok(img.clone()),
            }
        }
        Degradation::Screenshot => {
            println!("Simulating screenshot (resize 80% + blur 0.5 + jpeg 85)");
            let resized = img.resize(
                (img.width() as f32 * 0.8) as u32,
                (img.height() as f32 * 0.8) as u32,
                image::imageops::FilterType::Lanczos3,
            );
            Ok(resized.blur(0.5))
        }
        Degradation::Social => {
            println!("Simulating social media (resize 50% + jpeg 60)");
            Ok(img.resize(
                img.width() / 2,
                img.height() / 2,
                image::imageops::FilterType::Lanczos3,
            ))
        }
        Degradation::All { .. } => {
            bail!("All should be handled separately")
        }
    }
}

fn add_noise(img: &DynamicImage, amount: u32) -> Result<DynamicImage> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    let mut noisy: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let noise_scale = amount as f32 / 100.0 * 50.0; // Max noise of 50 gray levels

    for (x, y, pixel) in rgba.enumerate_pixels() {
        let noise: i32 = (rng.gen::<f32>() * 2.0 - 1.0) as i32 * noise_scale as i32;
        let r = (pixel[0] as i32 + noise).clamp(0, 255) as u8;
        let g = (pixel[1] as i32 + noise).clamp(0, 255) as u8;
        let b = (pixel[2] as i32 + noise).clamp(0, 255) as u8;
        noisy.put_pixel(x, y, Rgba([r, g, b, pixel[3]]));
    }

    Ok(DynamicImage::ImageRgba8(noisy))
}

fn save_image(img: &DynamicImage, path: &PathBuf, degradation: &Degradation) -> Result<()> {
    match degradation {
        Degradation::Jpeg { quality } => {
            let file = std::fs::File::create(path)?;
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, *quality);
            encoder.encode_image(img)?;
        }
        Degradation::Screenshot => {
            // Save as JPEG 85
            let file = std::fs::File::create(path)?;
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 85);
            encoder.encode_image(img)?;
        }
        Degradation::Social => {
            // Save as JPEG 60
            let file = std::fs::File::create(path)?;
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 60);
            encoder.encode_image(img)?;
        }
        _ => {
            // Determine format from extension
            let format = ImageFormat::from_path(path)?;
            img.save_with_format(path, format)?;
        }
    }
    Ok(())
}

fn apply_all_degradations(img: &DynamicImage, input_path: &PathBuf, output_dir: &PathBuf) -> Result<()> {
    let stem = input_path.file_stem().unwrap().to_str().unwrap();

    let degradations = vec![
        ("jpeg90", Degradation::Jpeg { quality: 90 }),
        ("jpeg70", Degradation::Jpeg { quality: 70 }),
        ("jpeg50", Degradation::Jpeg { quality: 50 }),
        ("jpeg30", Degradation::Jpeg { quality: 30 }),
        ("jpeg10", Degradation::Jpeg { quality: 10 }),
        ("resize75", Degradation::Resize { percent: 75 }),
        ("resize50", Degradation::Resize { percent: 50 }),
        ("resize25", Degradation::Resize { percent: 25 }),
        ("crop90", Degradation::Crop { percent: 90 }),
        ("crop80", Degradation::Crop { percent: 80 }),
        ("crop70", Degradation::Crop { percent: 70 }),
        ("blur1", Degradation::Blur { sigma: 1.0 }),
        ("blur2", Degradation::Blur { sigma: 2.0 }),
        ("blur3", Degradation::Blur { sigma: 3.0 }),
        ("noise10", Degradation::Noise { amount: 10 }),
        ("noise25", Degradation::Noise { amount: 25 }),
        ("noise50", Degradation::Noise { amount: 50 }),
        ("bright+20", Degradation::Brightness { delta: 20 }),
        ("bright-20", Degradation::Brightness { delta: -20 }),
        ("screenshot", Degradation::Screenshot),
        ("social", Degradation::Social),
    ];

    std::fs::create_dir_all(output_dir)?;

    println!("\n=== Applying {} degradations ===\n", degradations.len());

    for (name, degradation) in &degradations {
        let ext = match degradation {
            Degradation::Jpeg { .. } | Degradation::Screenshot | Degradation::Social => "jpg",
            _ => "png",
        };
        let output_path = output_dir.join(format!("{}-{}.{}", stem, name, ext));

        let degraded = apply_degradation(img, degradation)?;
        save_image(&degraded, &output_path, degradation)?;
        println!("  ✓ {}", output_path.display());
    }

    println!("\nGenerated {} degraded images in {}", degradations.len(), output_dir.display());

    Ok(())
}
