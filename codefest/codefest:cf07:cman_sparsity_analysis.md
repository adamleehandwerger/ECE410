# CMAN

## (a) N = 512

### Dense MVM

Each element of the output requires 2N FLOPs, therefore for an N-dim
output vector, the total FLOPs is:

> 2N² = 2(512)² = 524,288 FLOPs

Since there is an N×N matrix, N-dim input vector and a N-dim output vector with 4 bytes in each element, the memory use will be:

> 4(N² + 2N) = 1,052,672 bytes

### Sparse MVM

Let `s` be the fraction of zeros.

For each row of the sparse matrix, there are (1−s)N non-zero elements. Since there are N rows, the number of FLOPs will be:

> 2(1−s)N²

The number of non-zero elements in the matrix is (1−s)N², so both the element array and column array for compression will make equal contributions to the memory use. The pointer takes N+1 elements. The input and output vectors are length N, so the total memory usage is approximately:

> 4(2(1−s)N² + 3N + 1) bytes

---

## (b) 2× Speed-Up Condition

For compression to produce a 2× speed-up, the number of FLOPs under compression should be half of the uncompressed version:

> N² = 2(1−s)N²  ⇒  1−s = 1/2  ⇒  **s = 1/2**

---

## (c) Memory Break-Even Point

The break-even point for memory saving using sparse compression occurs approximately when:

> N² + 2N = 2(1−s)N² + 3N + 1

This implies:

> (N² − N − 1) / 2N² = 1−s  ⇒  **s = 1/2 + 1/(2N) + 1/(2N²)**

For large matrices, s ≅ 1/2. This only makes sense when N >> 2.

---

## (d) Speed-Up for s = 0.9

Assume the hardware exploits sparsity perfectly. Since the process is memory bound, the difference in memory usage determines the runtime:

> Δm = |4(N² + 2N − (2(1−s)N² + 3N + 1))|
>
> = |4(N² − 0.2N² − N − 1)|
>
> = 4(0.8N² − N − 1)
>
> = 4(0.8(512)² − 513) = **836,808.8 bytes**

At 320 GB/s:

> Δt = 8.368×10⁵ / (320×2²⁰) sec = **2.435 µs**
