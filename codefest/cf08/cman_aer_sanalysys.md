# CMAN - AER Bandwidth Analysis

**N = 1024**

1. The mean aggregate spike rate is the number of neurons times the mean firing rate:
   R_ag = N × f = 1024 × 50 = 5.12 × 10^5 spikes/s.

2. An AER packet contains 10 address bits, 6 timestamp bits, and 4 bits for framing.
   This is N_b = 20 bits per packet.

   The bandwidth B (bits/second) = (bits/spike) × (spikes/s) = R_ag × N_b = 20 × 5.12 × 10^5 =
   1.024 × 10^6 bit/s ≈ 1 Mbit/s.

3. The I²C interface has the appropriate specs at 3.4 Mbit/s.

4. B_b = the bandwidth needed for bursting behavior. The total number of spikes at 0.25 × N = 256
   spikes with 20 bits each. Hence:

   B = (# of bits)/s = (256 × 20) / 1 ms = 5.12 × 10^6 bits/s = 5.12 Mbit/s.

   The mean spiking rate is approximately 1 Mbit/s. This gives a bursting/mean ratio of
   approximately 5:1. The minimum buffer size is:

   (5.12 − 3.4) Mbit/s × 10^-3 s = 1.72 Kbits ≈ 215 bytes minimum buffer size.

5. N neurons at 1 bit/ms = 1024 × 10^3 bits/s = 1 Mbit/s. Therefore, the crossover frequency
   is exactly the mean firing rate. If there is substantial bursting behavior then frame-based
   representation would be the best choice, not AER.
