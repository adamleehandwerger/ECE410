# CMAN

## (a) N = 512

### Dense MVM

Each element of the output requires 2N FLOPs, therefore for an N-dim output vector, the total FLOPs is:

$$2N^2 = 2(512)^2 = 524{,}288 \text{ FLOPs}$$

Since there is an N×N matrix, N-dim input vector and N-dim output vector with 4 bytes per element, the memory use will be:

$$4(N^2 + 2N) = 1{,}052{,}672 \text{ bytes}$$

### Sparse MVM

Let $s$ be the fraction of zeros.

For each row of the sparse matrix, there are $(1-s)N$ non-zero elements. Since there are N rows, the number of FLOPs will be:

$$2(1-s)N^2$$

The number of non-zero elements in the matrix is $(1-s)N^2$, so both the element array and column array for compression will make equal contributions to the memory use. The pointer takes $N+1$ elements. The input and output vectors are length N, so the total memory usage is approximately:

$$4(2(1-s)N^2 + 3N + 1) \text{ bytes}$$

---

## (b) 2× Speed-Up Condition

For compression to produce a 2× speed-up, the number of FLOPs under compression should be half of the uncompressed version:

$$N^2 = 2(1-s)N^2 \implies 1-s = \frac{1}{2} \implies \boxed{s = \frac{1}{2}}$$

---

## (c) Memory Break-Even Point

The break-even point for memory saving using sparse compression occurs approximately when:

$$N^2 + 2N = 2(1-s)N^2 + 3N + 1$$

This implies:

$$\frac{N^2 - N - 1}{2N^2} = 1-s \implies \boxed{s = \frac{1}{2} + \frac{1}{2N} + \frac{1}{2N^2}}$$

For large matrices, $s \cong \frac{1}{2}$. This only makes sense when $N \gg 2$.

---

## (d) Speed-Up for s = 0.9

Assume the hardware exploits sparsity perfectly. Since the process is memory bound, the difference in memory usage determines the runtime:

$$\Delta m = \left|4\left(N^2 + 2N - \left(2(1-s)N^2 + 3N + 1\right)\right)\right|$$

$$= \left|4(N^2 - 0.2N^2 - N - 1)\right| = 4(0.8N^2 - N - 1)$$

$$= 4\left(0.8(512)^2 - 513\right) = \mathbf{836{,}808.8 \text{ bytes}}$$

At 320 GB/s:

$$\Delta t = \frac{8.368 \times 10^5}{320 \times 2^{20}} \text{ sec} = \mathbf{2.435 \ \mu s}$$
