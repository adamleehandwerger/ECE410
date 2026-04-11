# FLOP Count Analysis - Table Format


## Parameters

| Parameter                 | Symbol | Value | Description                                     |
|---------------------------|--------|-------|-------------------------------------------------|
| Samples                   | N      | 100   | Number of data points                           |
| Features                  | M      | 100   | Dimensionality of each point                    |
| Support Vectors           | D      | 50    | Number of support vectors                       |
| Horner Iterations         | K      | 15    | Number of iterations in Horner's method         |


---


## Function 1: dist_matrix()

Purpose: Compute distance between N points of dimension M and D support vectors


### Line-by-Line Analysis

| Line        | Code                                                                 | FLOPs Formula                      | Bytes Formula                   |
|-------------|----------------------------------------------------------------------|------------------------------------|---------------------------------|
| 1           | X_norm = np.sum(X**2, axis=1)                                        | 2N^2 - N                           | 4(N^2 + N)                      |
| 2           | support_norm = np.sum(support_matrix**2, axis=1)                     | 2DM - D                            | 4D(N + 1)                       |
| 3           | dist = X_norm + support_norm - 2*np.dot(X, support_matrix.T)         | 2ND + 2DN^2                        | 4(NM + 2ND + N + D)             |
| TOTAL       |                                                                      | 2N^2 + 2DN^2 + 4DN - N - D         | 4(2N^2 + 2N + 3DN + 2D)         |


### Summary (N=100, M=100, D=50)

| Metric                          | Formula                            | Value                        |
|---------------------------------|------------------------------------|------------------------------|
| Total FLOPs                     | 2N^2 + 2DN^2 + 4DN - N - D         | 1,039,850                    |
| Total Bytes                     | 4(2N^2 + 2N + 3DN + 2D)            | 141,200 (137.9 KB)           |
| Operational Intensity           | FLOPs / Bytes                      | 7.36 FLOP/byte               |


---


## Function 2: horner()

Purpose: RBF kernel approximation using Horner's method


### Line-by-Line Analysis

| Line        | Code                                                         | FLOPs Formula          | Bytes Formula          |
|-------------|--------------------------------------------------------------|------------------------|------------------------|
| 1           | Z = gamma * dist                                             | ND                     | 8ND                    |
| 2           | acc = np.zeros_like(Z)                                       | 0                      | 4ND                    |
| 3-4         | for k in range(K, 1, -1): acc = 1 + (Z/k) * acc             | 3(K-1)ND               | 12(K-1)ND              |
| TOTAL       |                                                              | ND(3K + 1)             | 12KND                  |


### Summary (N=100, D=50, K=15)

| Metric                          | Formula                            | Value                        |
|---------------------------------|------------------------------------|------------------------------|
| Total FLOPs                     | ND(3K + 1) = ND(46)                | 215,000                      |
| Total Bytes                     | 12KND = 12(15)(5000)               | 900,000 (878.9 KB)           |
| Operational Intensity           | (3K + 1) / 12K = 46/180            | 0.26 FLOP/byte               |


---


## Combined Analysis


### Overall Comparison

| Function            | FLOPs           | Percent of Total     | Bytes           | Percent of Total     | Intensity     | Status                      |
|---------------------|-----------------|----------------------|-----------------|----------------------|---------------|------------------------------|
| dist_matrix()       | 1,039,850       | 82.9 percent         | 141,200         | 13.6 percent         | 7.36          | Memory bound                 |
| horner()            | 215,000         | 17.1 percent         | 900,000         | 86.4 percent         | 0.26          | Severely memory bound        |
| TOTAL               | 1,254,850       | 100 percent          | 1,041,200       | 100 percent          | 1.21          | Memory bound                 |


### Total Memory

| Unit                  | Value                 |
|-----------------------|-----------------------|
| Bytes                 | 1,041,200             |
| Kilobytes             | 1,016.8 KB            |
| Megabytes             | 0.99 MB               |


---


## Roofline Model Analysis


### Ridge Points (Typical Hardware)

| Hardware            | Ridge Point           | Your Functions              | Status                                 |
|---------------------|-----------------------|-----------------------------|----------------------------------------|
| CPU (DDR4)          | 10-20 FLOP/byte       | dist_matrix(): 7.36         | Below ridge - Memory bound             |
|                     |                       | horner(): 0.26              | Far below ridge - Memory bound         |
| GPU (HBM)           | 10-15 FLOP/byte       | dist_matrix(): 7.36         | Below ridge - Memory bound             |
|                     |                       | horner(): 0.26              | Far below ridge - Memory bound         |


### Key Insights

| Observation                               | Value                           | Implication                                    |
|-------------------------------------------|---------------------------------|------------------------------------------------|
| dist_matrix() dominates compute           | 82.9 percent of FLOPs           | Does most of the work                          |
| horner() dominates memory traffic         | 86.4 percent of bytes           | Bottleneck for memory bandwidth                |
| Combined intensity                        | 1.21 FLOP/byte                  | Both functions are memory-bound                |
| horner() loop overhead                    | 14 iterations x 60KB/iter       | Heavy memory traffic, little compute           |


---


## Formulas Quick Reference

| Function            | FLOP Formula                       | Bytes Formula                   | Intensity Formula       |
|---------------------|------------------------------------|---------------------------------|-------------------------|
| dist_matrix()       | 2N^2 + 2DN^2 + 4DN - N - D         | 4(2N^2 + 2N + 3DN + 2D)         | FLOPs / Bytes           |
| horner()            | ND(3K + 1)                         | 12KND                           | (3K + 1) / 12K          |


