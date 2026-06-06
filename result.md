# Results

## Hardware
Intel Core i7-12700H — 14 physical cores (6 P-core + 8 E-core), 20 logical threads.  
See `hw.info` for full details.

## L2 Error Convergence

All configurations produce identical L2 errors, confirming that the parallel
decomposition does not introduces numerical error.

| n   | h        | L2 error   | ratio |
|-----|----------|------------|-------|
| 16  | 6.67e-02 | 2.8551e-02 | —     |
| 32  | 3.23e-02 | 9.5450e-03 | 3.0×  |
| 64  | 1.59e-02 | 3.2900e-03 | 2.9×  |
| 128 | 7.87e-03 | 1.1492e-03 | 2.9×  |
| 256 | 3.92e-03 | 4.0390e-04 | 2.8×  |

The error decreases by a factor of ~3 each time n doubles (h halves),
consistent with first-order convergence in h for the Jacobi iteration
stopped at the discretisation error level.


## Wall-clock Time (seconds)

| n   | P=1 T=1 | P=2 T=1 | P=4 T=1 | P=4 T=2 |
|-----|---------|---------|---------|---------|
| 16  | 0.0017  | 0.0003  | 0.0007  | 0.0024  |
| 32  | 0.0109  | 0.0064  | 0.0044  | 0.0160  |
| 64  | 0.1529  | 0.1084  | 0.0591  | 0.0970  |
| 128 | 3.0380  | 2.1685  | 1.3193  | 1.3392  |
| 256 | 56.6807 | 38.7864 | 23.6890 | 22.8192 |


## Speedup over Serial (P=1 T=1)

| n   | P=2 T=1 | P=4 T=1 | P=4 T=2 |
|-----|---------|---------|---------|
| 16  | 5.7×    | 2.4×    | 0.7×    |
| 32  | 1.7×    | 2.5×    | 0.7×    |
| 64  | 1.4×    | 2.6×    | 1.6×    |
| 128 | 1.4×    | 2.3×    | 2.3×    |
| 256 | 1.5×    | 2.4×    | 2.5×    |


## Discussion

**Correctness** — the L2 error is identical across all parallel configurations
and decreases monotonically with h.

**MPI scaling (P=1 → P=4, T=1)** — for large grids (n ≥ 64) the speedup
stabilises around 2.4×, well below the ideal 4×.

**Small grids (n = 16, 32)** — parallel overhead exceeds the computational work per rank. 
Adding ranks can actually increase wall-clock time, as seen for P=4 T=2 at n=16/32.
This is expected behaviour: MPI is only beneficial when each rank has
enough work to amortise communication costs.

**OpenMP (P=4 T=1 vs P=4 T=2)** — adding a second OpenMP thread per rank
gives negligible improvement for n ≤ 128 and only ~4% for n=256. The reason
is that with P=4 each rank owns only n/4 rows; at n=256 that is 64 rows,
giving each thread only ~8k floating-point operations per sweep, OpenMP parallelism 
could not pays off. Additionally, the hybrid
P-core/E-core architecture of the i7-12700H means OpenMP threads may be
scheduled on slower E-cores, causing load imbalance at the `MPI_Allreduce`
barrier.

**Laptop vs HPC cluster** — results were obtained on a laptop CPU not designed
for HPC workloads. On a cluster with homogeneous cores, dedicated memory
channels per node, and a fast interconnect, both MPI and hybrid scaling could
be significantly closer to ideal.