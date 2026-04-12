# Analysis Partition Rationale

## Question 8

### (a) Accelerator Selection and Performance Analysis

Both `dist_matrix` and `horner` functions would achieve better performance with the H100 accelerator. 

**Current Performance (100×100 data matrix):**
- Data matrix: 100 points with 100 features each
- Support vector matrix: ~50 support vectors (half the data size)
- **Roofline analysis shows:** Both functions are **memory-bound**

**Scaling Analysis (250×250 data matrix):**
- When the data matrix increases to 250×250:
  - `dist_matrix` becomes **compute-bound**
  - Operational intensity shifts to the right on the Roofline model

**Recommendation:**
Since we want the accelerator to process matrices of **any size**, the **largest processor (H100) is the first choice**. Further analysis may show that the H100 may be oversized for some applications.


### (b) SVM Application Acceleration Strategy

**Functions to accelerate:**
- `dist_matrix()` - Distance matrix calculation
- `horner()` - RBF kernel approximation using Horner's method

**Remaining bottlenecks (not accelerated):**
- Loading data from external drive
- One vector-matrix multiplication to determine the decision function

These remain as the largest time/memory software elements in the overall SVM pipeline.


### (c) Minimum Bandwidth Requirements

**Analysis for 100×100 X matrix:**

Both functions operate on the memory-bound region of the Roofline model. 

**Minimum bandwidth to avoid becoming interface-bound:** **3,120 GiB/s**

This ensures the H100's compute capacity can be fully utilized without being limited by memory bandwidth.


### (d) Memory vs Compute Bound Analysis

**Question 1: Is `dist_matrix` memory-bound or compute-bound?**

**Answer:** **Depends on the size of matrix X**
- For the baseline Roofline analysis (100×100): **Memory-bound**. For larger matrices (≥250×250): **Compute-bound** (as mentioned in part (a))
- The function transitions from memory-bound to compute-bound as matrix size increases


**Question 2: Is `horner` function memory-bound or compute-bound?**

**Answer:** **Always memory-bound on the H100**

**Rationale:**
- Since `dist_matrix` has a **higher operational intensity** than `horner`, `dist_matrix` will become compute-bounded **first** as data matrix size increases
- Therefore, `horner` will **always remain memory-bounded** on the H100, regardless of matrix size. **Important:** The operational intensity of `horner` **does not scale with X size** and is **not affected by matrix dimensions** - it depends only on the number of Horner iterations (K) and remains constant at 0.24 FLOP/byte


---

## Summary

| Function       | 100×100 Status  | 250×250 Status  | Long-term Behavior        |
|----------------|-----------------|-----------------|---------------------------|
| `dist_matrix` | Memory-bound    | Compute-bound   | Transitions with size     |
| `horner`      | Memory-bound    | Memory-bound    | Always memory-bound       |

**Key Insight:** The operational intensity of `horner` is so low that it will never reach the H100's ridge point, making it perpetually memory-bound regardless of dataset size.
