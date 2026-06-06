#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "JacobiSolver.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

JacobiSolver::JacobiSolver(int n, Force f, double tol, int maxIter, Boundary bc_bottom, 
    Boundary bc_top, Boundary bc_left, Boundary bc_right)
    : n(n), h(1.0 / (n - 1)), h2(h * h), tol(tol), maxIter(maxIter),
    f(std::move(f)), bc_bottom(std::move(bc_bottom)), bc_top(std::move(bc_top)), 
    bc_left(std::move(bc_left)),bc_right(std::move(bc_right))
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    computeDecomposition();
    initializeGrid();
}

void JacobiSolver::computeDecomposition()
{
    const int totalRows = n;
    const int split_rows = totalRows / size;
    const int remainder = totalRows % size;
    rowStart  = rank * split_rows + std::min(rank, remainder);
    if (rank < remainder)
        localRows = split_rows + 1;
    else
        localRows = split_rows;
    rowEnd    = rowStart + localRows - 1;
    U.assign((localRows + 2) * n, 0.0);
    Unew.assign((localRows + 2) * n, 0.0);
}

void JacobiSolver::initializeGrid()
{
    for (int i = 1; i <= localRows; ++i) {
        int gi = rowStart + (i - 1);
        double y = yCoord(gi);

        for (int j = 0; j < n; ++j) {
            double x = xCoord(j);
            double val = 0.0;
            bool onBottom = (gi == 0);
            bool onTop = (gi == n - 1);
            bool onLeft = (j == 0);
            bool onRight = (j == n - 1);
            if (onBottom) val = bc_bottom(x, y);
            else if (onTop) val = bc_top(x, y);
            else if (onLeft) val = bc_left(x, y);
            else if (onRight) val = bc_right(x, y);
            U[idx(i, j)] = val;
            Unew[idx(i, j)] = val;
        }
    }
}

void JacobiSolver::exchangeGhostRows()
{
    MPI_Request reqs[4];
    int nreqs = 0;
    const int prev = rank - 1;
    const int next = rank + 1;
    if (prev >= 0) {
        MPI_Isend(&U[idx(1, 0)], n, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(&U[idx(0, 0)], n, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (next < size) {
        MPI_Isend(&U[idx(localRows, 0)], n, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(&U[idx(localRows + 1, 0)], n, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    // i put waitall to guarantee the synchronization among the parallel process
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

double JacobiSolver::localJacobiSweep()
{
    double localSqSum = 0.0;
    #pragma omp parallel for reduction(+:localSqSum) 
    for (int i = 1; i <= localRows; ++i) {
        int gi = rowStart + (i - 1);

        if (gi == 0 || gi == n - 1) continue;

        for (int j = 1; j < n - 1; ++j) {
            double x = xCoord(j);
            double y = yCoord(gi);
            double newVal = 0.25 * ( U[idx(i - 1, j)] + U[idx(i + 1, j)] + U[idx(i, j - 1)] + U[idx(i, j + 1)] + f(x, y) * h2 );
            double diff = newVal - U[idx(i, j)];
            localSqSum += diff * diff;
            Unew[idx(i, j)] = newVal;
        }
    }
    return localSqSum;
}

double JacobiSolver::globalConvergence(double localSqSum) const
{
    double globalSqSum = 0.0;
    MPI_Allreduce(&localSqSum, &globalSqSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(h * globalSqSum);
}

int JacobiSolver::solve()
{
    double t0 = MPI_Wtime();
    int iter = 0;
    const int checkEvery = std::max(10, n / 4);
    const int printEvery = std::max(1, std::min(5000, maxIter / 10));
    for (iter = 0; iter < maxIter; ++iter) {
        exchangeGhostRows();
        double localSqSum = localJacobiSweep();
        std::swap(U, Unew);
        //i don't check convergence each iteration, to avoid time consuming operations at each iteration
        if (iter % checkEvery == 0) {
            double err = globalConvergence(localSqSum);
            if (err < tol) {
                ++iter;
                if (rank == 0)
                    std::cout << "Converged after " << iter
                              << " iterations (err=" << err << ")\n";
                break;
            }
        }
    }
    solveTime = MPI_Wtime() - t0;
    return iter;
}

void JacobiSolver::exportVTK(const std::string& filename) const
{
    std::vector<double> localData(localRows * n);
    for (int lr = 0; lr < localRows; ++lr)
        std::memcpy(&localData[lr * n],
                    &U[idx(lr + 1, 0)],
                    n * sizeof(double));

    if (rank == 0) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const int base      = n / size;
            const int remainder = n % size;
            counts[r] = (base + (r < remainder ? 1 : 0)) * n;
        }
        displs[0] = 0;
        for (int r = 1; r < size; ++r)
            displs[r] = displs[r-1] + counts[r-1];

        std::vector<double> globalData(n * n);
        MPI_Gatherv(localData.data(), (int)localData.size(), MPI_DOUBLE, globalData.data(), counts.data(), 
        displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::ofstream ofs(filename);
        if (!ofs) throw std::runtime_error("Cannot open " + filename);

        ofs << "# vtk DataFile Version 3.0\n"
            << "Laplace solver\n"
            << "ASCII\n"
            << "DATASET STRUCTURED_POINTS\n"
            << "DIMENSIONS " << n << " " << n << " 1\n"
            << "ORIGIN 0 0 0\n"
            << "SPACING " << h << " " << h << " 1\n"
            << "POINT_DATA " << n * n << "\n"
            << "SCALARS u double 1\n"
            << "LOOKUP_TABLE default\n";

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                ofs << globalData[i * n + j] << "\n";

    
        std::cout << "VTK written to " << filename << "\n";
    } else {
        MPI_Gatherv(localData.data(), (int)localData.size(), MPI_DOUBLE,
        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

double JacobiSolver::computeL2Error(const std::function<double(double,double)>& exact) const {
    double localSq = 0.0;
    #pragma omp parallel for reduction(+:localSq) 
    for (int i = 1; i <= localRows; ++i) {
        int gi = rowStart + (i - 1);
        double y = yCoord(gi);
        for (int j = 0; j < n; ++j) {
            double x = xCoord(j);
            double diff = U[idx(i, j)] - exact(x, y);
            localSq += diff * diff;
        }
    }
    double globalSq = 0.0;
    MPI_Reduce(&localSq, &globalSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return (rank == 0) ? std::sqrt(h * globalSq) : 0.0;
}