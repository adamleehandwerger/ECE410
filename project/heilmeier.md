Heilmeier Questions

Question#1 What are you trying to do with the new design?

Answer: Improve and simplify the implementation of the Radial Basis Kernel for use in SVM and other machine learning applications. We suggest Horner's approximation of the Hadamard Expansion of the Gaussian Kernel Matrix with K=12 to approximate the RBF kernel.

Revision#1: K=14 loop interations was used for the design. Also a function dist_matrix was added to the accelerator design since a profile analysis showed it was using the time in the SVM algorithm. In more general setting, the dist_matrix function can calculate the norm between all points of the dataset X by using X.T for support_matrix. This will allow the accelerator to be used in applications beyond the SVM.

Question#2 How is the RBF kernel used today and what are the limits of it current use?

Answer: In many cases such as in Embedded Systems, like wearable health monitors for ECG/EEG which operate on a limited power budget, KSVM inference is not practical. They could become practical with a Horner chip. Also, the algorithm inside IoT sensors and Smart home devices that incorporate on-device pattern recognition could be greatly improved.

Revision#1: No revsion yet

Question#3 What is the new approach and what difference will it make?

Answer: It will allow machine learning algorithms to incorporate RBF kernel inference which is a standard for many machine learning applications into systems that have little or no CPU support and/or power consumption restrictions that make this type of inference to costly with standard methods. Also having a fixed latency will provide more reliable reaction times for systems such as autononous vehicles that use classification algorithms in real-time. The computational latency is currently 60-120 cycles for current FPGA programs that use CORDIC lookup tables whereas it would be 12 cycles with a K=12 Horner chip.

Revsion#1: Here is a detailed table of latency values for both CPU and H100.


---


## Latency Analysis for SVM Functions

**Configuration:** N=100, M=100, D=50, K=14

**horner() fixed latency:** CPU = 19.29 μs, H100 = 0.27 μs


### Performance Comparison Table

| Matrix Size | Support Vectors | dist_matrix (ms) |          | horner (ms) |          | Total (ms) |          | Speedup |
|-------------|-----------------|------------------|----------|-------------|----------|------------|----------|---------|
|             |                 | CPU              | H100     | CPU         | H100     | CPU        | H100     |         |
|-------------|-----------------|------------------|----------|-------------|----------|------------|----------|---------|
| 100         | 50              | 0.0030           | 0.000042 | 0.0193      | 0.000269 | 0.0223     | 0.000311 | 71.8×   |
| 248         | 124             | 0.0185           | 0.000258 | 0.0193      | 0.000269 | 0.0378     | 0.000527 | 71.7×   |
| 500         | 250             | 0.0751           | 0.002100 | 0.0193      | 0.000269 | 0.0944     | 0.002369 | 39.9×   |
| 1000        | 500             | 0.3003           | 0.016733 | 0.0193      | 0.000269 | 0.3196     | 0.017002 | 18.8×   |

*Note: horner() latency is constant across all matrix sizes (depends only on K=14 iterations)*


### Key Insights

- **Small matrices (N=100):** Both functions are memory-bound, achieving **71.8× speedup** on H100
- **Medium matrices (N=248):** dist_matrix reaches H100's ridge point, speedup remains **71.7×**
- **Large matrices (N≥500):** dist_matrix becomes compute-bound on H100, speedup decreases to **39.9-18.8×**
- **horner() behavior:** Always memory-bound with constant latency regardless of matrix size
- **H100 advantage:** Provides 0.31 μs total latency for typical inference (N=100) vs 22.3 μs on CPU


### Hardware Specifications

| Architecture | Peak Performance | Memory Bandwidth | Ridge Point      |
|--------------|------------------|------------------|------------------|
| CPU DDR4     | 1,000 GFLOP/s    | 46.66 GB/s       | 21.43 FLOP/byte  |
| H100 HBM3    | 60,000 GFLOP/s   | 3,350 GB/s       | 17.91 FLOP/byte  |
| **Ratio**    | **60×**          | **71.8×**        | **0.84×**        |


---
