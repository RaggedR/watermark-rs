//! Fast 2D cross-correlation using FFT.
//!
//! Cross-correlation finds where two signals align best.
//! Using 2D FFT: corr(a,b) = IFFT2D(FFT2D(a) * conj(FFT2D(b)))

use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

/// 2D correlator with cached FFT plans.
pub struct Correlator2D {
    width: usize,
    height: usize,
    fft_row: Arc<dyn Fft<f64>>,
    fft_col: Arc<dyn Fft<f64>>,
    ifft_row: Arc<dyn Fft<f64>>,
    ifft_col: Arc<dyn Fft<f64>>,
}

impl Correlator2D {
    pub fn new(width: usize, height: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            width,
            height,
            fft_row: planner.plan_fft_forward(width),
            fft_col: planner.plan_fft_forward(height),
            ifft_row: planner.plan_fft_inverse(width),
            ifft_col: planner.plan_fft_inverse(height),
        }
    }

    /// Compute 2D FFT (row-column decomposition).
    fn fft_2d(&self, data: &mut [Complex64]) {
        // FFT on rows
        for row in 0..self.height {
            let start = row * self.width;
            let row_slice = &mut data[start..start + self.width];
            self.fft_row.process(row_slice);
        }

        // Transpose
        let mut transposed = vec![Complex64::new(0.0, 0.0); self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                transposed[x * self.height + y] = data[y * self.width + x];
            }
        }

        // FFT on columns (now rows after transpose)
        for col in 0..self.width {
            let start = col * self.height;
            let col_slice = &mut transposed[start..start + self.height];
            self.fft_col.process(col_slice);
        }

        // Transpose back
        for y in 0..self.height {
            for x in 0..self.width {
                data[y * self.width + x] = transposed[x * self.height + y];
            }
        }
    }

    /// Compute 2D inverse FFT.
    fn ifft_2d(&self, data: &mut [Complex64]) {
        // IFFT on rows
        for row in 0..self.height {
            let start = row * self.width;
            let row_slice = &mut data[start..start + self.width];
            self.ifft_row.process(row_slice);
        }

        // Transpose
        let mut transposed = vec![Complex64::new(0.0, 0.0); self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                transposed[x * self.height + y] = data[y * self.width + x];
            }
        }

        // IFFT on columns (now rows after transpose)
        for col in 0..self.width {
            let start = col * self.height;
            let col_slice = &mut transposed[start..start + self.height];
            self.ifft_col.process(col_slice);
        }

        // Transpose back
        for y in 0..self.height {
            for x in 0..self.width {
                data[y * self.width + x] = transposed[x * self.height + y];
            }
        }
    }

    /// Compute normalized cross-correlation between two 2D arrays.
    ///
    /// Returns the correlation surface where peaks indicate alignment.
    pub fn correlate(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let size = self.width * self.height;
        assert_eq!(a.len(), size);
        assert_eq!(b.len(), size);

        // Convert to complex
        let mut fa: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut fb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Forward 2D FFT
        self.fft_2d(&mut fa);
        self.fft_2d(&mut fb);

        // Multiply: FFT(a) * conj(FFT(b))
        let mut product: Vec<Complex64> = fa.iter()
            .zip(fb.iter())
            .map(|(a, b)| a * b.conj())
            .collect();

        // Inverse 2D FFT
        self.ifft_2d(&mut product);

        // Normalize and return real parts
        let scale = 1.0 / size as f64;
        product.iter().map(|c| c.re * scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_peak() {
        let correlator = Correlator2D::new(4, 4);

        // Create a simple pattern
        let mut a = vec![0.0; 16];
        a[0] = 1.0;
        a[5] = 1.0;
        a[10] = 1.0;

        // Autocorrelation should peak at (0,0)
        let result = correlator.correlate(&a, &a);

        // Find max
        let max_idx = result.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;

        assert_eq!(max_idx, 0, "Autocorrelation should peak at index 0");
    }

    #[test]
    fn test_shifted_correlation_2d() {
        let width = 8;
        let height = 8;
        let correlator = Correlator2D::new(width, height);
        let size = width * height;

        // Create pattern at origin
        let mut a = vec![0.0; size];
        a[0] = 1.0;

        // Create same pattern shifted by (row=2, col=3)
        let mut b = vec![0.0; size];
        let shift_row = 2;
        let shift_col = 3;
        let shift_linear = shift_row * width + shift_col;
        b[shift_linear] = 1.0;

        let result = correlator.correlate(&a, &b);

        // Find peak
        let max_idx = result.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;

        // Peak should be at the shift position (with 2D FFT correlation)
        // For cross-correlation: peak at shift position or (size - shift)
        let expected_wrapped = (height - shift_row) * width + (width - shift_col);
        assert!(max_idx == shift_linear || max_idx == expected_wrapped,
                "Expected peak at {} or {}, got {}", shift_linear, expected_wrapped, max_idx);
    }
}
