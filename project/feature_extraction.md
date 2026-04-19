# Feature Extraction Flexibility & Model Updates

## Question
**If I keep feature extraction out of the chiplet, can the feature designations be updated as improvements are made to the model?**

## Answer: Yes - Maximum Flexibility!

Keeping feature extraction on the MCU (outside the chiplet) provides crucial flexibility for model improvements and iterations without requiring hardware changes.

---

## Advantages of MCU-Based Feature Extraction

### 1. Model Updates Without Hardware Changes

**What You Can Update:**
- ✅ Retrain SVM with different feature sets
- ✅ Change which 256 features to extract
- ✅ Add new feature types (e.g., nonlinear features)
- ✅ Remove less discriminative features
- ✅ Optimize feature computation algorithms

**The Chiplet Doesn't Care:**
- The chiplet only sees 256 floating-point numbers per heartbeat
- It doesn't know or care what those numbers represent
- Same distance and kernel computation regardless of feature meaning

### 2. Feature Selection Improvements

**Research → Deploy Workflow:**
1. Run feature importance analysis on laptop/desktop
2. Identify which features contribute most to classification
3. Update MCU firmware to compute only the best features
4. Upload new model (support vectors) to MCU
5. No chiplet redesign needed!

**Example Improvements:**
- Replace time-domain with frequency-domain features
- Switch wavelet families (Daubechies → Symlet)
- Add heart rate variability features
- Remove redundant statistical features

### 3. Algorithm Evolution

**Dimension Changes:**
- Reduce: 256 → 128 features (lower computation, faster)
- Expand: 256 → 512 features (more information, better accuracy)
- As long as chiplet hardware supports the maximum dimension

**Method Changes:**
- Different wavelet decomposition levels
- Alternative frequency domain transforms (FFT → Welch)
- New statistical measures
- Deep learning-derived features

### 4. Adaptive & Personalized Features

**Patient-Specific Customization:**
- Age-specific feature sets (pediatric vs adult vs elderly)
- Gender-specific features
- Condition-specific features (post-MI, CHF, etc.)
- Activity-level adjusted features

**All via firmware update - no hardware change!**

---

## What Stays Fixed vs What's Flexible

### Fixed in Chiplet Hardware (Can't Change Without Redesign)

| Component | Fixed Parameter |
|-----------|----------------|
| **Distance units** | Number of parallel units (16) |
| **Distance dimension** | Maximum feature dimension (256 or 512) |
| **Kernel type** | RBF kernel (exp approximation) |
| **Matrix size** | Max samples × max support vectors |
| **Output classes** | Number of classes (5 for One-vs-Rest) |

### Flexible via Firmware Update (MCU)

| Component | Flexible Parameter |
|-----------|-------------------|
| **Feature extraction** | Algorithm, methods, parameters |
| **Feature selection** | Which 256 features to compute |
| **Support vectors** | Values (retrain → upload new SVs) |
| **RBF gamma** | Kernel width parameter |
| **Scaling** | Feature normalization parameters |
| **Decision thresholds** | Class prediction cutoffs |
| **Number of SVs** | Actual count (up to hardware max) |

---

## Design Pattern: Separation of Concerns

```
┌─────────────────────────────────────────┐
│         MCU (Flexible)                  │
│  ┌───────────────────────────────────┐  │
│  │ Feature Extraction Algorithm      │  │  ← Update via firmware
│  │ - Time domain methods             │  │
│  │ - Frequency domain methods        │  │
│  │ - Wavelet parameters              │  │
│  │ - Statistical computations        │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ Model Parameters                  │  │  ← Update via data upload
│  │ - Support vector values           │  │
│  │ - Feature scaling (mean/std)      │  │
│  │ - RBF gamma                       │  │
│  │ - Decision thresholds             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    ↓ 256 floats
┌─────────────────────────────────────────┐
│       Chiplet (Fixed Hardware)          │
│  ┌───────────────────────────────────┐  │
│  │ Distance Matrix Computation       │  │  ← Fixed: 256-dimensional
│  │ - 16 parallel Euclidean units     │  │           Euclidean distance
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ RBF Kernel Evaluation             │  │  ← Fixed: exp(γ·D²)
│  │ - Horner polynomial approx        │  │           approximation
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Example Update Scenarios

### Scenario 1: Feature Optimization (Month 6)

**Discovery:** Research shows wavelet features are most discriminative

**Action:**
1. Retrain model emphasizing wavelet features
2. Update MCU firmware:
   - Increase wavelet features: 64 → 128
   - Reduce statistical features: 64 → 0
   - Still 256 total features
3. Upload new support vectors (trained on new features)

**Chiplet:** Unchanged! Still computes distance on 256 dimensions

**Result:** Better accuracy, same hardware

### Scenario 2: Patient Personalization (Year 1)

**Discovery:** Elderly patients need different features than young patients

**Action:**
1. Create two feature extraction profiles
2. MCU detects patient age (from calibration)
3. MCU selects appropriate feature extraction method
4. MCU loads corresponding support vectors

**Chiplet:** Unchanged! Just receives different 256 numbers

**Result:** Personalized medicine via firmware

### Scenario 3: Algorithm Evolution (Year 2)

**Discovery:** Deep learning can generate better features

**Action:**
1. Train autoencoder on laptop
2. Deploy encoder network to MCU
3. MCU runs lightweight encoder: ECG → 256 features
4. Upload new SVM trained on encoded features

**Chiplet:** Unchanged! Still distance + kernel on 256 floats

**Result:** State-of-the-art features, same chiplet

---

## The Only Constraint: Maximum Dimension

**Design Decision:** The chiplet hardware must support the **maximum** feature dimension you might ever need.

**Conservative Approach:**
- Design chiplet for 512 dimensions
- Currently use 256 dimensions
- Future-proofs for expansion
- Unused hardware doesn't consume power (clock gated)

**Cost:** Minimal (~2× area for distance units)

**Benefit:** Never need hardware revision for feature changes

---

## Update Workflow

### When You Improve the Model:

```
Step 1: Research (Laptop/Cloud)
├─ Experiment with different features
├─ Train multiple SVM variants
├─ Evaluate on validation set
└─ Select best model → Get new support vectors

Step 2: Prepare Firmware Update
├─ Write new feature extraction code
├─ Package support vector data
├─ Create scaling parameter file
└─ Test in simulation

Step 3: Deploy to Device
├─ Send OTA firmware update to MCU
├─ Upload new support vectors
├─ Update scaling parameters
└─ Validate on device

Step 4: Chiplet
└─ No change required! ✓
```

### Version Control Example:

```
Model v1.0 (Launch):
├─ MCU: Extract 256 features (current method)
├─ MCU: 91 support vectors
├─ Chiplet: Distance + RBF kernel
└─ Accuracy: 95%

Model v2.0 (6 months later):
├─ MCU: Extract DIFFERENT 256 features (improved algorithm)
├─ MCU: 150 support vectors (retrained)
├─ Chiplet: Still distance + RBF kernel ✓
└─ Accuracy: 98% ← Better with no hardware change!

Model v3.0 (1 year later):
├─ MCU: Deep learning features (encoder network)
├─ MCU: 200 support vectors
├─ Chiplet: Still distance + RBF kernel ✓
└─ Accuracy: 99.5% ← Even better!
```

---

## Recommendation: Design for Flexibility

### Chiplet Hardware Design Guidelines:

1. **Support maximum expected dimension** (recommend 512 even if using 256)
2. **Parameterizable matrix sizes** (support 100-500 SVs)
3. **Configurable kernel** (load gamma parameter from MCU)
4. **Generic distance computation** (works for any feature semantics)

### MCU Firmware Architecture:

1. **Modular feature extraction** (easy to swap algorithms)
2. **Configuration files** (feature methods, scaling params)
3. **OTA update capability** (wireless model updates)
4. **A/B testing** (try new models on subset of data)

---

## Conclusion

**Yes, absolutely!** Keeping feature extraction on the MCU provides **maximum flexibility to improve the model over time** via firmware updates without ever touching the chiplet hardware.

**Key Benefits:**
- ✅ Update features without hardware redesign
- ✅ Continuous model improvement via firmware
- ✅ Patient-specific personalization
- ✅ Algorithm evolution (traditional → deep learning)
- ✅ A/B testing of new features in the field
- ✅ Rapid iteration based on real-world data

**This is a smart design choice** that separates concerns:
- **Chiplet:** Fast, fixed-function acceleration (distance + kernel)
- **MCU:** Flexible, updateable intelligence (features + model)

The chiplet becomes a reusable, general-purpose RBF kernel accelerator that remains valuable even as the feature extraction and model evolve over multiple product generations!

---

**Generated:** April 19, 2026
