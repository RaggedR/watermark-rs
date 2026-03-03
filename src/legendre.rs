//! Legendre sequence generation for watermarking.
//!
//! Legendre sequences have ideal autocorrelation properties, making them
//! excellent for watermarking - they produce a sharp peak when correlated
//! with themselves, but flat response otherwise.

/// Generate a p×p Legendre array for watermark sequence k.
///
/// This matches the C++ implementation:
/// 1. Create 1D Legendre sequence: legendre[i] = 1 if i is a quadratic residue mod p, -1 otherwise
/// 2. Build 2D array: for each column i, compute shift = (i² × k) mod p
/// 3. Fill column: array[j,i] = legendre[(j + shift) mod p]
///
/// The parameter k determines which "family" of watermark patterns to use.
pub fn generate_array(p: usize, k: usize) -> Vec<f64> {
    // Step 1: Create 1D Legendre sequence of length p
    // legendre[i] = 1 if i is a quadratic residue mod p, -1 otherwise
    let mut legendre = vec![-1i32; p];

    // Mark all squares mod p as 1
    for i in 0..p {
        let j = (i * i) % p;
        legendre[j] = 1;
    }

    // Step 2 & 3: Build 2D array using column-wise shifts
    let mut array = vec![0.0; p * p];

    for i in 0..p {
        // Shift for column i depends on i² × k
        let shift = (i * i * k) % p;

        for j in 0..p {
            // Row j uses legendre[(j + shift) mod p]
            let leg_idx = (j + shift) % p;
            // Array is stored row-major: array[j * p + i] = legendre[leg_idx]
            array[j * p + i] = legendre[leg_idx] as f64;
        }
    }

    array
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_array() {
        let arr = generate_array(5, 1);
        assert_eq!(arr.len(), 25);
        // Values should be -1 or 1 (no zeros in this construction)
        for &v in &arr {
            assert!(v == -1.0 || v == 1.0);
        }
    }

    #[test]
    fn test_generate_array_different_k() {
        // Different k values should produce different arrays
        let arr1 = generate_array(7, 1);
        let arr2 = generate_array(7, 2);
        assert_ne!(arr1, arr2);
    }

    #[test]
    fn test_legendre_sequence() {
        // For p=7, quadratic residues are {0, 1, 2, 4}
        // legendre sequence should be [1, 1, 1, -1, 1, -1, -1] (indices 0-6)
        // Actually 0 gets mapped to 1 because 0 = 0² mod 7
        let arr = generate_array(7, 0);
        // When k=0, shift for column 0 is 0, so first column should be raw legendre
        // Column 0: arr[0], arr[7], arr[14], arr[21], arr[28], arr[35], arr[42]
        // = legendre[0], legendre[1], legendre[2], legendre[3], legendre[4], legendre[5], legendre[6]
        assert_eq!(arr[0], 1.0);   // legendre[0] = 1 (0 is 0²)
        assert_eq!(arr[7], 1.0);   // legendre[1] = 1 (1 is 1²)
        assert_eq!(arr[14], 1.0);  // legendre[2] = 1 (2 is 3² mod 7 = 2)
        assert_eq!(arr[21], -1.0); // legendre[3] = -1 (not a QR)
        assert_eq!(arr[28], 1.0);  // legendre[4] = 1 (4 is 2²)
        assert_eq!(arr[35], -1.0); // legendre[5] = -1
        assert_eq!(arr[42], -1.0); // legendre[6] = -1
    }
}
