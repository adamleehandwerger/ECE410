# Matrix Multiplication DRAM Traffic Analysis

Consider multiplying two square matrices with floating point entries (FP32) with the number of rows N, where A×B=C.

## (a) Naive Implementation DRAM Traffic

For each column of C, one element B is accessed N times. For each element of C, one row of A and one column of B is accessed. Hence 2N elements. This implies that for all of the N×N matrix C, a total of 2N(N²) = 2N³ elements are moved from DRAM.

For N=32, this gives:
- 2(32)³ = 65,536 floats = 262,144 bytes

**Note:** This does NOT include writing N² elements to DRAM.

## (b) Tiled Implementation DRAM Traffic

Since for an optimally sized tile each element of C need only to be accessed once, the number of times DRAM needs to be accessed would be the same as if the original matrices were reduced in size to N/T elements per side times the number of DRAM loads per tile.

Therefore the DRAM traffic would become:
- 2(N/T)³ × T² = 2N³/T

For T=8, we get:
- 65,536/8 floats = 8,192 floats = 32,768 bytes

## (c) Ratio of Tiled to Naive

The ratio tiled/naive is:
- 2N³/(T×2N³) = 1/T

## (d) Performance Analysis

### Naive Case
The naive case requires a transfer of 8N³ bytes at 320 × 2³⁰ bytes/s.

This would require:
- 8(32)³/(320 × 2³⁰) sec = 7.6 × 10⁻⁷ sec

This case is **memory bounded** since:
- Arithmetic Intensity = N/6 = 32/6 ≈ 5.3 FLOP/byte
- Ridge point is at 10¹³/(320×2³⁰) ≈ 29 FLOP/byte

### Tiled Case
For the tiled case:
- Number of FLOPS: 2N³
- Bytes transferred: 8N³/T
- Arithmetic Intensity = 0.25T = 0.25(8) = 2 FLOP/byte

Using this formulation with T=1 gives the naive Arithmetic Intensity as 0.25.

The linear dependence on T means that the Operational Intensity reaches the ridge at:
- T ≈ 4(29) = 58

**Conclusion:** Until the matrices are of this dimension and the L1 caches are large enough to support this specification, the algorithm remains memory bounded.
