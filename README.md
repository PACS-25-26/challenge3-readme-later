[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tKSbaXxd)
# challenge3
The third challenge

# Parallel Jacobi Solver

Parallel solver for the Laplace equation using Jacobi iteration with hybrid MPI + OpenMP.


## Build
```bash
make
```

## Run
```bash
# single solve: n=128, tol=1e-6, maxIter=200000, 4 MPI ranks, 2 OMP threads
OMP_NUM_THREADS=2 mpirun -np 4 ./laplace_solver 128 1e-6 200000

# scalability benchmark
OMP_NUM_THREADS=2 mpirun -np 4 ./benchmark
```

## Scalability test
```bash
bash test/run_scalability.sh
```
Results saved in `test/data/`.

## Results
See [RESULT.md](RESULT.md).