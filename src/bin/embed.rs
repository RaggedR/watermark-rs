//! Watermark embedding CLI.
//!
//! Usage: embed <input> <output> <message> [--strength <value>] [--pilot]
//!
//! Adaptive mode: Use --adaptive to automatically choose strength based on image texture.
//! Use --robust with --adaptive for JPEG-resistant watermarking (may be visible on flat images).

use anyhow::{Context, Result};
use clap::Parser;
use image::{GenericImageView, GrayImage, ImageBuffer};
use std::path::PathBuf;
use std::time::Instant;
use watermark_detection::{
    embed_watermark_with_options, largest_prime_for, EmbedOptions,
    recommend_strength, StrengthRecommendation,
};

#[derive(Parser)]
#[command(name = "embed", about = "Embed invisible watermark into image")]
struct Args {
    /// Path to the input image file
    input_path: PathBuf,

    /// Path for the output watermarked image (PNG)
    output_path: PathBuf,

    /// Message to embed (e.g., "0xABC123:42")
    message: String,

    /// Watermark strength (1-10 typical, higher = more robust but more visible)
    #[arg(short, long, default_value = "2.0")]
    strength: f64,

    /// Automatically choose strength based on image texture analysis
    #[arg(long)]
    adaptive: bool,

    /// With --adaptive: prioritize JPEG robustness (may cause visible artifacts)
    #[arg(long, requires = "adaptive")]
    robust: bool,

    /// Output as color image (embed in V channel of HSV)
    #[arg(long)]
    color: bool,

    /// Embed pilot signal for rotation-invariant detection
    #[arg(long, default_value_t = true)]
    pilot: bool,

    /// Disable pilot signal embedding
    #[arg(long, conflicts_with = "pilot")]
    no_pilot: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let total_start = Instant::now();

    println!("========================================");
    println!("      WATERMARK EMBEDDING");
    println!("========================================\n");
    let use_pilot = args.pilot && !args.no_pilot;

    println!("Input: {}", args.input_path.display());
    println!("Output: {}", args.output_path.display());
    println!("Message: \"{}\"", args.message);

    // Load image
    let img = image::open(&args.input_path).context("Failed to open input image")?;
    let (width, height) = img.dimensions();
    println!("Image loaded: {}x{}", width, height);

    // Determine strength (adaptive or manual)
    let (strength, recommendation) = if args.adaptive {
        let gray = img.to_luma8();
        let pixels: Vec<u8> = gray.as_raw().clone();
        let rec = recommend_strength(&pixels, width as usize, height as usize);

        let chosen_strength = if args.robust {
            // Prioritize JPEG robustness, even if potentially visible
            rec.jpeg_q90_strength.max(rec.recommended)
        } else {
            // Prioritize invisibility
            rec.recommended
        };

        (chosen_strength, Some(rec))
    } else {
        (args.strength, None)
    };

    // Print strength info
    if let Some(ref rec) = recommendation {
        println!("\n--- Adaptive Strength Analysis ---");
        println!("  Texture score:     {:.3} ({})",
            rec.texture_score,
            if rec.texture_score > 0.3 { "highly textured" }
            else if rec.texture_score > 0.15 { "moderately textured" }
            else if rec.texture_score > 0.05 { "some texture" }
            else { "mostly flat" }
        );
        println!("  Flat regions:      {:.1}%", rec.flat_region_percent);
        println!("  Max invisible:     {:.1}", rec.max_invisible);

        if args.robust {
            println!("  Mode:              ROBUST (JPEG priority)");
            if strength > rec.max_invisible {
                println!("  ⚠️  Warning: Strength {:.1} may cause visible artifacts", strength);
            }
        } else {
            println!("  Mode:              INVISIBLE (quality priority)");
        }
        println!("  Chosen strength:   {:.1}", strength);

        // JPEG survival prediction
        if strength >= rec.jpeg_q80_strength {
            println!("  JPEG survival:     Q80+ ✓");
        } else if strength >= rec.jpeg_q90_strength {
            println!("  JPEG survival:     Q90+ ✓ (Q80 may fail)");
        } else {
            println!("  JPEG survival:     PNG only (JPEG will destroy watermark)");
        }
    } else {
        println!("Strength: {}", strength);
    }

    println!("Pilot signal: {}\n", if use_pilot { "enabled" } else { "disabled" });

    let p = largest_prime_for(width as usize, height as usize);
    println!("Prime for embedding: {}", p);
    println!("Watermark coefficients: {} x {} = {}\n", p, p, p * p);

    let result = if args.color {
        embed_color(&img, &args.message, strength, use_pilot)?
    } else {
        embed_grayscale(&img, &args.message, strength, use_pilot)?
    };

    // Save result
    result.save(&args.output_path).context("Failed to save output image")?;

    println!("\n========================================");
    println!("            COMPLETE");
    println!("========================================\n");
    println!("Output saved: {}", args.output_path.display());
    println!("Total time: {:.0} ms\n", total_start.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

/// Embed watermark in grayscale image.
fn embed_grayscale(img: &image::DynamicImage, message: &str, strength: f64, pilot: bool) -> Result<image::DynamicImage> {
    let (width, height) = img.dimensions();

    // Convert to grayscale
    let gray = img.to_luma8();
    let pixels: Vec<u8> = gray.as_raw().clone();

    println!("Embedding watermark...");

    let options = if pilot {
        EmbedOptions::with_strength(strength)
    } else {
        EmbedOptions::without_pilot(strength)
    };

    let result = embed_watermark_with_options(&pixels, width as usize, height as usize, message, &options);

    println!("  Sequences used: {}", result.sequences_used);
    println!("  Prime: {}", result.prime);
    println!("  Shifts: {:?}", result.shifts);
    println!("  Pilot embedded: {}", result.pilot_embedded);

    // Create output image
    let output: GrayImage = ImageBuffer::from_raw(width, height, result.pixels)
        .context("Failed to create output image")?;

    Ok(image::DynamicImage::ImageLuma8(output))
}

/// Embed watermark in color image (V channel of HSV).
fn embed_color(img: &image::DynamicImage, message: &str, strength: f64, pilot: bool) -> Result<image::DynamicImage> {
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    // Convert RGB to HSV and extract V channel
    let mut h_channel: Vec<f64> = Vec::with_capacity((width * height) as usize);
    let mut s_channel: Vec<f64> = Vec::with_capacity((width * height) as usize);
    let mut v_pixels: Vec<u8> = Vec::with_capacity((width * height) as usize);

    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let (h, s, v) = rgb_to_hsv(r, g, b);
        h_channel.push(h);
        s_channel.push(s);
        v_pixels.push(v);
    }

    println!("Embedding watermark in V channel...");

    let options = if pilot {
        EmbedOptions::with_strength(strength)
    } else {
        EmbedOptions::without_pilot(strength)
    };

    let result = embed_watermark_with_options(&v_pixels, width as usize, height as usize, message, &options);

    println!("  Sequences used: {}", result.sequences_used);
    println!("  Prime: {}", result.prime);
    println!("  Shifts: {:?}", result.shifts);
    println!("  Pilot embedded: {}", result.pilot_embedded);

    // Convert back to RGB
    let mut output_pixels: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);

    for (i, &new_v) in result.pixels.iter().enumerate() {
        let (r, g, b) = hsv_to_rgb(h_channel[i], s_channel[i], new_v);
        output_pixels.push(r);
        output_pixels.push(g);
        output_pixels.push(b);
    }

    let output: image::RgbImage = ImageBuffer::from_raw(width, height, output_pixels)
        .context("Failed to create output image")?;

    Ok(image::DynamicImage::ImageRgb8(output))
}

/// Convert RGB to HSV.
/// Returns (H: 0-360, S: 0-1, V: 0-255)
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, u8) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = (max * 255.0) as u8;

    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Convert HSV to RGB.
/// Expects (H: 0-360, S: 0-1, V: 0-255)
fn hsv_to_rgb(h: f64, s: f64, v: u8) -> (u8, u8, u8) {
    let v = v as f64 / 255.0;

    if s == 0.0 {
        let v = (v * 255.0) as u8;
        return (v, v, v);
    }

    let h = h / 60.0;
    let i = h.floor() as i32;
    let f = h - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => (v, v, v),
    };

    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
