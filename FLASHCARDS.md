# Watermark-rs Flash Cards & Knowledge Graph

A study guide for understanding the invisible watermarking system.

---

## 🗺️ Knowledge Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WATERMARK-RS ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   MESSAGE    │
                              │  "0xABC:42"  │
                              └──────┬───────┘
                                     │
                            ┌────────▼────────┐
                            │  7-bit ASCII    │
                            │  Pack to BigInt │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │  Base p² Encode │
                            │  → shift values │
                            └────────┬────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               │                               │
     ▼                               ▼                               ▼
┌─────────┐                    ┌─────────┐                    ┌─────────┐
│ shift₀  │                    │ shift₁  │                    │ shift₂  │
└────┬────┘                    └────┬────┘                    └────┬────┘
     │                              │                              │
     ▼                              ▼                              ▼
┌─────────────┐              ┌─────────────┐              ┌─────────────┐
│ Legendre(1) │              │ Legendre(2) │              │ Legendre(3) │
│  p×p array  │              │  p×p array  │              │  p×p array  │
└──────┬──────┘              └──────┬──────┘              └──────┬──────┘
       │                            │                            │
       │ apply 2D shift             │ apply 2D shift             │ apply 2D shift
       ▼                            ▼                            ▼
┌─────────────┐              ┌─────────────┐              ┌─────────────┐
│  Shifted    │              │  Shifted    │              │  Shifted    │
│  Pattern₀   │              │  Pattern₁   │              │  Pattern₂   │
└──────┬──────┘              └──────┬──────┘              └──────┬──────┘
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │ sum & scale by strength
                                    ▼
                           ┌────────────────┐
                           │ Combined Mark  │
                           │   (p×p f64)    │
                           └────────┬───────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
           ┌────────────────┐              ┌────────────────┐
           │  INPUT IMAGE   │              │   2D DFT       │
           │   (pixels)     │──────────────│  of watermark  │
           └────────┬───────┘              └────────┬───────┘
                    │                               │
                    │ 2D DFT                        │
                    ▼                               │
           ┌────────────────┐                       │
           │   FREQUENCY    │◄──────────────────────┘
           │    DOMAIN      │   add to [1..p+1, 1..p+1]
           │   (complex)    │   (low-frequency region)
           └────────┬───────┘
                    │
                    │ Inverse 2D DFT
                    ▼
           ┌────────────────┐
           │  WATERMARKED   │
           │     IMAGE      │
           └────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DETECTION FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌────────────────┐
│  INPUT IMAGE   │ (possibly degraded: JPEG, blur, noise, resize)
│  (watermarked) │
└───────┬────────┘
        │
        │ 2D DFT
        ▼
┌────────────────┐
│   FREQUENCY    │
│    DOMAIN      │
└───────┬────────┘
        │
        │ extract [1..p+1, 1..p+1]
        ▼
┌────────────────┐
│ Extracted Mark │
│   (p×p real)   │
└───────┬────────┘
        │
        │ for each sequence k = 1, 2, 3...
        ▼
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐  │
│  │ Legendre(k) │ ──► │ Correlate   │ ──► │ Find Peak       │  │
│  │  Reference  │     │  via FFT    │     │ Calculate P/RMS │  │
│  └─────────────┘     └─────────────┘     └────────┬────────┘  │
│                                                   │            │
│                         ┌─────────────────────────┘            │
│                         ▼                                      │
│                  ┌─────────────┐                               │
│                  │ P/RMS > 4?  │──── NO ────► STOP             │
│                  └──────┬──────┘     (noise floor)             │
│                         │ YES                                  │
│                         ▼                                      │
│                  ┌─────────────┐                               │
│                  │ Record shift│                               │
│                  │ (peak pos)  │                               │
│                  └─────────────┘                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
        │
        │ collected shifts
        ▼
┌────────────────┐
│ Decode base p² │
│ → BigInt       │
└───────┬────────┘
        │
        │ extract 7-bit chars
        ▼
┌────────────────┐
│   MESSAGE      │
│  "0xABC:42"    │
└────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCIES                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────┐
                              │  lib.rs  │
                              │ (public  │
                              │   API)   │
                              └────┬─────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        │              │           │           │              │
        ▼              ▼           ▼           ▼              ▼
   ┌─────────┐   ┌─────────┐ ┌─────────┐ ┌──────────┐  ┌──────────┐
   │legendre │   │  dft    │ │correlat.│ │embedding │  │detection │
   │   .rs   │   │   .rs   │ │   .rs   │ │   .rs    │  │   .rs    │
   └────┬────┘   └────┬────┘ └────┬────┘ └────┬─────┘  └────┬─────┘
        │             │           │           │             │
        │             │           │           │             │
        │   ┌─────────┴───────────┴───────────┘             │
        │   │                                               │
        │   │  ┌────────────────────────────────────────────┘
        │   │  │
        ▼   ▼  ▼
   ┌─────────────────────────────────────────┐
   │              USES                        │
   │  • legendre → embedding, detection      │
   │  • dft → embedding, detection, analysis │
   │  • correlation → detection              │
   │  • rustfft → dft, correlation           │
   │  • num_bigint → embedding, detection    │
   └─────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────┐
        │               CLI BINARIES                       │
        ├─────────────────────────────────────────────────┤
        │  embed.rs  ──► embedding.rs + image crate       │
        │  detect.rs ──► detection.rs                     │
        │  analyze.rs──► analysis.rs                      │
        │  degrade.rs──► image crate (no watermarking)    │
        │  experiment──► ALL modules + CSV output         │
        └─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MATHEMATICAL CONCEPT MAP                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

    NUMBER THEORY                    SIGNAL PROCESSING
    ─────────────                    ─────────────────
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│ Quadratic       │              │ Fourier Transform   │
│ Residues        │              │ (Frequency Domain)  │
│                 │              │                     │
│ a ≡ x² (mod p)  │              │ spatial → frequency │
└────────┬────────┘              └──────────┬──────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│ Legendre Symbol │              │ Low Frequencies     │
│                 │              │ (perceptually       │
│ L(a,p) = ±1     │              │  important, robust) │
└────────┬────────┘              └──────────┬──────────┘
         │                                  │
         ▼                                  │
┌─────────────────┐                         │
│ Legendre        │                         │
│ Sequence        │◄────────────────────────┘
│                 │     embed in low-freq
│ [+1,-1,+1,...]  │
└────────┬────────┘
         │
         │ Ideal autocorrelation
         ▼
┌─────────────────┐              ┌─────────────────────┐
│ Sharp Peak      │──────────────│ Cross-Correlation   │
│ Detection       │  enables     │ via FFT             │
└─────────────────┘              │                     │
                                 │ O(n log n) speed    │
                                 └─────────────────────┘
```

---

## 🃏 Flash Cards

### Category 1: Core Mathematical Concepts

---

#### Card 1.1: Quadratic Residue
**Q: What is a quadratic residue modulo prime p?**

<details>
<summary>Answer</summary>

A number `a` is a **quadratic residue** mod p if there exists some `x` such that:
```
a ≡ x² (mod p)
```

**Example** (p=7):
- 1 = 1² mod 7 ✓
- 2 = 3² mod 7 (9 mod 7 = 2) ✓
- 4 = 2² mod 7 ✓
- Quadratic residues: {0, 1, 2, 4}
- Non-residues: {3, 5, 6}

</details>

---

#### Card 1.2: Legendre Symbol
**Q: What does the Legendre symbol L(a, p) compute, and what are its possible values?**

<details>
<summary>Answer</summary>

The Legendre symbol determines if `a` is a quadratic residue mod prime `p`:

```
L(a, p) = +1  if a is a quadratic residue (a ≡ x² mod p)
L(a, p) = -1  if a is NOT a quadratic residue
L(a, p) =  0  if a ≡ 0 (mod p)
```

**Why it matters for watermarking**: Creates pseudo-random ±1 sequences with ideal autocorrelation properties.

</details>

---

#### Card 1.3: Autocorrelation Property
**Q: Why are Legendre sequences good for watermarking? What special property do they have?**

<details>
<summary>Answer</summary>

Legendre sequences have **ideal autocorrelation**:

- **Sharp peak** at shift=0 (perfect match with itself)
- **Near-zero** values at all other shifts

This means:
1. Easy to detect: correlation produces a clear spike
2. Robust to noise: random interference averages to zero
3. Multiple watermarks don't interfere (low cross-correlation)

```
Autocorrelation:  ▃▁▁▁▁█▁▁▁▁▃  (peak at center)
                       ↑
                   shift = 0
```

</details>

---

#### Card 1.4: DFT Purpose
**Q: Why does the watermark use the frequency domain (DFT) instead of directly modifying pixels?**

<details>
<summary>Answer</summary>

Three key reasons:

1. **Imperceptibility**: Low frequencies are perceptually important but small changes are invisible
2. **Robustness**: Low frequencies survive JPEG compression, blur, and noise better than high frequencies
3. **Spread spectrum**: Watermark energy spreads across the image spatially, making local attacks ineffective

**Trade-off**: The DC component [0,0] is avoided because changes there affect overall brightness noticeably.

</details>

---

#### Card 1.5: Why Low Frequencies?
**Q: The watermark is embedded in frequency region [1..p+1, 1..p+1]. Why not [0,0] or high frequencies?**

<details>
<summary>Answer</summary>

| Region | Why NOT use it |
|--------|----------------|
| **[0,0] (DC)** | Controls average brightness - changes very visible |
| **High frequencies** | Destroyed by JPEG compression, blur, resize |
| **Edges [0,j] and [i,0]** | Often affected by boundary artifacts |

**[1..p+1, 1..p+1]** is the sweet spot:
- Low enough to survive degradation
- High enough to be imperceptible
- Large enough area (p² coefficients) for encoding

</details>

---

### Category 2: Message Encoding

---

#### Card 2.1: 7-bit ASCII Packing
**Q: How is a text message like "Hi" converted to a number for embedding?**

<details>
<summary>Answer</summary>

1. Convert each character to 7-bit ASCII:
   - 'H' = 72 = 0b1001000
   - 'i' = 105 = 0b1101001

2. Pack MSB-first into a big integer:
   ```
   value = 72 << 7 | 105
         = 9216 + 105
         = 9321
   Binary: 10010001101001
   ```

**Why 7 bits?** Standard printable ASCII (32-126) fits in 7 bits. Saves space vs 8-bit.

</details>

---

#### Card 2.2: Base p² Encoding
**Q: Why encode the message in base p² (where p is the prime)? What's special about p²?**

<details>
<summary>Answer</summary>

Each Legendre sequence can encode a 2D shift position (row, col) in a p×p array.

- **Total shift values**: p × p = p²
- **Bits per sequence**: log₂(p²) ≈ 2 × log₂(p)

**Example** (p=593 for 800×600 image):
- p² = 351,649 possible shifts
- Each sequence encodes ~18.4 bits
- Message "Hello" = 35 bits → fits in just 2 sequences!

**Encoding**:
```
shifts[0] = value mod p²
shifts[1] = (value / p²) mod p²
shifts[2] = (value / p⁴) mod p²
...
```

</details>

---

#### Card 2.3: 2D Shift Encoding
**Q: Given a shift value `s` and array dimensions p×p, how do you compute the horizontal and vertical shifts?**

<details>
<summary>Answer</summary>

```rust
h_shift = s % p;        // horizontal (column) shift
v_shift = s / p;        // vertical (row) shift
```

Then apply cyclic shift:
```rust
shifted[row][col] = original[(row + v_shift) % p][(col + h_shift) % p]
```

**Example** (p=7, shift=23):
- h_shift = 23 % 7 = 2
- v_shift = 23 / 7 = 3
- Pattern shifts right 2, down 3 (with wraparound)

</details>

---

### Category 3: Embedding Process

---

#### Card 3.1: Embedding Steps
**Q: List the 6 main steps to embed a watermark message into an image.**

<details>
<summary>Answer</summary>

1. **Find prime p** = largest_prime(min(width, height) - 2)

2. **Encode message** → pack to BigInt → convert to base-p² shifts

3. **For each shift k**:
   - Generate Legendre array(p, k)
   - Apply 2D shift by shift_value
   - Scale by strength parameter

4. **Sum all shifted patterns** → combined watermark (p×p)

5. **2D DFT** of original image → frequency domain

6. **Add watermark** to region [1..p+1, 1..p+1], then **inverse DFT**

</details>

---

#### Card 3.2: Strength Parameter
**Q: What does the "strength" parameter control, and what's the trade-off?**

<details>
<summary>Answer</summary>

**Strength** multiplies the Legendre pattern magnitude before adding to frequency coefficients.

| Strength | Visibility | Robustness |
|----------|------------|------------|
| 1-2 | Invisible | May fail after heavy JPEG |
| 3-5 | Barely visible in flat areas | Good for most uses |
| 6-10 | Visible artifacts | Survives extreme degradation |

**Trade-off**: Higher strength = more robust but more visible distortion.

**Typical choice**: 2-4 for invisible watermarks that survive social media.

</details>

---

#### Card 3.3: Why Multiple Sequences?
**Q: Why might you need multiple Legendre sequences (k=1, 2, 3...) for one message?**

<details>
<summary>Answer</summary>

**Capacity**: Each sequence encodes log₂(p²) bits via its shift position.

For longer messages:
- Message bits > log₂(p²) → need multiple sequences
- Each sequence k uses a **different** Legendre array
- Sequences are **orthogonal** (don't interfere)

**Example**:
- p = 100 → p² = 10,000 → ~13 bits per sequence
- 100-character message = 700 bits
- Need: 700 / 13 ≈ 54 sequences

</details>

---

### Category 4: Detection Process

---

#### Card 4.1: Blind Detection
**Q: What makes detection "blind"? What information is NOT needed?**

<details>
<summary>Answer</summary>

**Blind detection** means recovering the watermark **without the original image**.

**Not needed**:
- Original unwatermarked image
- The exact message (we're finding it!)
- Embedding strength used

**Only needed**:
- The watermarked image
- Knowledge of the algorithm (prime selection, Legendre generation)
- Detection threshold

This is crucial for copyright verification - you don't need to store originals.

</details>

---

#### Card 4.2: Correlation Peak Finding
**Q: After correlating the extracted mark with a Legendre reference, what do you look for?**

<details>
<summary>Answer</summary>

Look for a **sharp peak** in the correlation surface:

```
Correlation surface (p×p):
┌─────────────────────┐
│ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │
│ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │
│ ▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁ │  ← Peak at (row=2, col=8)
│ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │
└─────────────────────┘
```

**Peak position** = the 2D shift value = encoded message digit
**Peak height / RMS** = confidence (should be > threshold ~4.0)

</details>

---

#### Card 4.3: Peak-to-RMS Ratio
**Q: What is the Peak-to-RMS ratio and why is it the key detection metric?**

<details>
<summary>Answer</summary>

```
peak = max(correlation_surface)
rms = sqrt(mean(correlation² excluding peak))

Peak/RMS ratio = peak / rms
```

**Interpretation**:
| Ratio | Meaning |
|-------|---------|
| ~1 | Random noise, no watermark |
| 2-3 | Weak signal, possibly watermark |
| 4+ | Confident detection |
| 10+ | Strong, undegraded watermark |
| 100+ | Very clean signal |

**Threshold typically 4.0**: Balances false positives vs missed detections.

</details>

---

#### Card 4.4: Detection Termination
**Q: When does the detection loop stop checking for more sequences?**

<details>
<summary>Answer</summary>

Detection stops when **Peak/RMS drops below threshold** (e.g., 4.0).

**Why this works**:
- Sequences are checked k=1, 2, 3...
- If sequence k was used → strong peak
- If sequence k was NOT used → random correlation ≈ 1.0

So when you hit a "no signal" result:
- All remaining sequences are also unused
- Message has been fully decoded
- Stop to save computation

</details>

---

### Category 5: Robustness

---

#### Card 5.1: JPEG Survival
**Q: Why does the watermark survive JPEG compression?**

<details>
<summary>Answer</summary>

JPEG compression primarily destroys **high frequencies**:

1. **DCT quantization** zeros out high-frequency coefficients
2. **Low frequencies** are preserved (they carry most visual info)
3. **Watermark lives in low frequencies** [1..p+1, 1..p+1]

**Limit**: Extreme compression (quality < 30) starts affecting low frequencies too.

```
Frequency importance:
Low ████████████░░░░░░░░ High
    ↑ watermark here
    (preserved by JPEG)
```

</details>

---

#### Card 5.2: Blur Resistance
**Q: Why does Gaussian blur not destroy the watermark?**

<details>
<summary>Answer</summary>

**Blur = low-pass filter** in frequency domain:

- **Removes high frequencies** (edges, details)
- **Preserves low frequencies** (where watermark lives)

Mathematically:
```
Blur in spatial domain = multiply by Gaussian in frequency domain
Gaussian is ~1.0 at low frequencies, ~0 at high frequencies
```

**Limit**: Extreme blur (σ > 5) starts attenuating even low frequencies.

</details>

---

#### Card 5.3: Resize Handling
**Q: What happens to the watermark when an image is resized to 50%?**

<details>
<summary>Answer</summary>

**Key insight**: Prime p is recalculated based on new dimensions!

Original: 800×600 → p = 593
Resized: 400×300 → p = 293

**What happens**:
1. Frequency domain shrinks proportionally
2. Watermark region [1..p+1] also shrinks
3. Lower resolution means less information, but pattern preserved

**Challenge**: If original used p=593 but detection uses p=293, they don't match exactly.

**Mitigation**: Low-frequency patterns are relatively scale-invariant for moderate resizing.

</details>

---

#### Card 5.4: What Breaks the Watermark?
**Q: Name 4 attacks that WILL destroy or severely damage the watermark.**

<details>
<summary>Answer</summary>

1. **Rotation** (even small angles)
   - DFT changes dramatically with rotation
   - Would need rotation-invariant transform

2. **Extreme JPEG** (quality < 30)
   - Low frequencies start getting quantized

3. **Severe cropping** (< 50% of image)
   - Not enough frequency information survives

4. **Geometric distortion** (perspective, warping)
   - Frequency relationships destroyed

**Also problematic**:
- Printing and re-scanning (adds noise + geometric distortion)
- Heavy noise addition (σ > 50)
- Multiple re-encodings (YouTube, Instagram chains)

</details>

---

### Category 6: Implementation Details

---

#### Card 6.1: Prime Selection
**Q: How is the prime p chosen for an image, and why subtract 2?**

<details>
<summary>Answer</summary>

```rust
p = largest_prime(min(width, height) - 2)
```

**Why min(width, height)**:
- Watermark region must fit in both dimensions
- Use smaller dimension as constraint

**Why subtract 2**:
- Leaves room for DC component at [0,0]
- Avoids boundary effects at edges [0,j] and [i,0]
- Watermark goes in [1..p+1, 1..p+1]

**Example**: 800×600 image
- min = 600
- 600 - 2 = 598
- largest_prime(598) = 593

</details>

---

#### Card 6.2: FFT Correlation Trick
**Q: Write the formula for computing correlation using FFT. Why is this faster?**

<details>
<summary>Answer</summary>

```
correlation(a, b) = IFFT( FFT(a) × conj(FFT(b)) )
```

**Speed comparison**:
- Naive correlation: O(n²) - compare every shift
- FFT correlation: O(n log n) - three FFT operations

For p=593 (p² ≈ 350,000 elements):
- Naive: ~122 billion operations
- FFT: ~6 million operations
- **20,000× faster!**

</details>

---

#### Card 6.3: BigInt for Messages
**Q: Why does the code use arbitrary-precision integers (BigUint) for message encoding?**

<details>
<summary>Answer</summary>

**Problem**: Messages can be arbitrarily long.

- "0xDEADBEEF1234567890ABCDEF:42" = 29 chars = 203 bits
- u64 max = 64 bits ❌
- u128 max = 128 bits ❌

**Solution**: `num_bigint::BigUint` handles any size:

```rust
// Can represent numbers with thousands of digits
let value: BigUint = chars.iter()
    .fold(BigUint::zero(), |acc, &c| (acc << 7) | c);
```

This matches the C++ implementation's use of `boost::multiprecision::cpp_int`.

</details>

---

#### Card 6.4: Color Image Handling
**Q: How are color images watermarked? Which channel is modified?**

<details>
<summary>Answer</summary>

**Approach**: Convert to HSV, watermark only the V (Value/brightness) channel.

```
RGB → HSV
         ↓
    H: Hue        (unchanged)
    S: Saturation (unchanged)
    V: Value      ← WATERMARK HERE
         ↓
HSV → RGB
```

**Why V channel?**:
- Human vision more sensitive to hue/saturation changes
- Brightness changes less noticeable
- Watermark in V affects all RGB channels subtly

</details>

---

### Category 7: Analysis & Metrics

---

#### Card 7.1: Hideability Score
**Q: What image properties make a watermark more "hideable" (less visible)?**

<details>
<summary>Answer</summary>

High hideability = watermark artifacts blend in:

| Property | Good for hiding | Why |
|----------|-----------------|-----|
| **High texture** (local variance) | Noise masks artifacts |
| **High entropy** | Complex patterns hide changes |
| **Many edges** | Edge regions tolerate more distortion |
| **Mid-range intensity** | Not too dark/bright (clipping) |

**Formula**:
```
hideability = 0.3 × variance_score
            + 0.3 × entropy_score
            + 0.2 × edge_score
            + 0.2 × intensity_score
```

</details>

---

#### Card 7.2: Robustness Score
**Q: What image properties predict better watermark survival after degradation?**

<details>
<summary>Answer</summary>

High robustness = watermark survives attacks:

| Property | Good for robustness | Why |
|----------|---------------------|-----|
| **Low spectral flatness** | Energy concentrated → cleaner peaks |
| **Low low-freq ratio** | Less interference in watermark band |
| **High watermark SNR** | Clean region for embedding |

**Formula**:
```
robustness = 0.3 × (1 - spectral_flatness)
           + 0.4 × (1 - low_freq_ratio)
           + 0.3 × watermark_snr_score
```

**Intuition**: Images with "structured" frequency content (not noise-like) work best.

</details>

---

#### Card 7.3: Spectral Flatness
**Q: What is spectral flatness and what does a value close to 0 vs 1 indicate?**

<details>
<summary>Answer</summary>

```
Spectral Flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)
```

| Value | Spectrum Shape | Image Type |
|-------|---------------|------------|
| **≈ 0** | Concentrated peaks | Tonal, structured (good!) |
| **≈ 1** | Flat, uniform | Noise-like, random (bad) |

**Why it matters**:
- Concentrated energy → clear correlation peaks
- Flat spectrum → watermark competes with noise everywhere
- Natural images typically have low flatness (structured)

</details>

---

### Category 8: Quick Reference

---

#### Card 8.1: Module → Responsibility
**Q: Match each module to its primary responsibility.**

<details>
<summary>Answer</summary>

| Module | Responsibility |
|--------|----------------|
| `legendre.rs` | Generate p×p Legendre arrays for sequence k |
| `dft.rs` | 2D FFT, extract low-frequency watermark region |
| `correlation.rs` | FFT-based 2D cross-correlation |
| `embedding.rs` | Encode message, combine patterns, add to image |
| `detection.rs` | Extract mark, correlate, decode shifts to message |
| `analysis.rs` | Compute image metrics, predict quality |

</details>

---

#### Card 8.2: CLI Commands
**Q: What does each CLI binary do?**

<details>
<summary>Answer</summary>

```bash
# Embed watermark
embed input.png output.png "message" --strength 2.0

# Detect watermark (blind)
detect image.png --threshold 4.0

# Analyze image quality
analyze image.png --json

# Apply degradation
degrade input.png output.png jpeg 75
degrade input.png output.png blur 2.0
degrade input.png outputs/ all    # 21 degradations

# Batch experiment
experiment ./images/ results.csv --strengths 1,2,4,8
```

</details>

---

#### Card 8.3: Key Constants
**Q: What are typical/default values for these parameters?**

<details>
<summary>Answer</summary>

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| **Detection threshold** | 4.0 | Peak/RMS ratio |
| **Embedding strength** | 2.0 | Range 1-10 |
| **Max sequences** | 20-100 | Depends on message length |
| **ASCII bits** | 7 | Per character |
| **Prime p** | ~500-1000 | For typical photos |

</details>

---

## 🧠 Self-Test Questions

### Conceptual Understanding

1. Why can't you just XOR the message bits directly into pixel values?
2. What would happen if you used sequence k=0 (all zeros)?
3. Why is the watermark "spread spectrum"?
4. How would you watermark a 100×100 thumbnail?

### Implementation

5. Trace the encoding of message "AB" with p=7 through all steps.
6. What's the maximum message length for a 1920×1080 image using 10 sequences?
7. If Peak/RMS = 3.5, should you report a detection? Why or why not?

### Debugging

8. Detection returns garbage characters. What's likely wrong?
9. Watermark visible in sky regions. How do you fix it?
10. Detection works on original but fails after JPEG 85. What parameter helps?

<details>
<summary>Answers</summary>

1. **XOR into pixels**: Would be destroyed by ANY processing (JPEG, resize, even rounding). No robustness.

2. **Sequence k=0**: Would produce all-same values, no variation, correlation undefined.

3. **Spread spectrum**: Each bit affects the entire p×p frequency region, not a single coefficient. Attacker can't target specific bits.

4. **100×100 thumbnail**: p = largest_prime(98) = 97. Only 97² ≈ 9,409 shifts per sequence. ~13 bits. Very limited capacity.

5. **"AB" with p=7**:
   - A=65, B=66 → BigInt = (65 << 7) | 66 = 8386
   - Base 49: 8386 = 171×49 + 7 → shifts = [7, 171 mod 49 = 24, 3]
   - Generate Legendre(1), shift by 7; Legendre(2), shift by 24; Legendre(3), shift by 3

6. **1920×1080 max message**: p = largest_prime(1078) ≈ 1069. p² ≈ 1.14M. 10 sequences × 20 bits ≈ 200 bits ≈ 28 characters.

7. **Peak/RMS 3.5**: Borderline. Default threshold is 4.0. Could be weak watermark or degraded. Report with low confidence or investigate.

8. **Garbage characters**: Likely wrong prime (resized image?), or message encoding mismatch (7-bit vs 8-bit?), or threshold too low (detecting noise).

9. **Visible in sky**: Reduce strength parameter (try 1.0-1.5). Or use analysis to identify problematic regions.

10. **Fails after JPEG 85**: Increase embedding strength to 3-4. Or verify with strength 8 to confirm it's a robustness issue.

</details>

---

## 📚 Further Reading

- **Legendre sequences**: Number theory, quadratic residues
- **Spread spectrum watermarking**: Cox et al. "Digital Watermarking"
- **DFT properties**: Autocorrelation theorem, convolution theorem
- **Image quality metrics**: SSIM, PSNR for measuring visibility
