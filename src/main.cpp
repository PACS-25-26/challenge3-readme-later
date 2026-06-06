/**
 * @brief Main for the parallel Jacobi solver.
 * 
 * The problem solved is:
 *   -Delta u = 8 pi^2 sin(2 pi x) sin(2 pi y)  on (0,1)^2
 *   u = 0 on all boundaries
 * 
 * Exact solution is u(x,y) = sin(2 pi x) sin(2 pi y)
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "JacobiSolver.hpp"

static constexpr double PI = M_PI;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    n = (argc > 1) ? std::stoi(argv[1]) : 64;
    double tol = (argc > 2) ? std::stod(argv[2]) : 1e-6;
    int    maxIter = (argc > 3) ? std::stoi(argv[3]) : 200000;

    if (rank == 0) {
        std::cout << " Parallel Jacobi Solver — Laplace eq.\n"
                  << " Grid size   : " << n << " x " << n << "\n"
                  << " Tolerance   : " << tol << "\n"
                  << " Max iters   : " << maxIter << "\n"
                  << " MPI ranks   : " << size << "\n";

        std::cout << " OpenMP threads : " << omp_get_max_threads() << "\n";
    }

    auto f = [](double x, double y) {
        return 8.0 * PI * PI * std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
    };
    auto exact = [](double x, double y) {
        return std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
    };

    JacobiSolver solver(n, f, tol, maxIter,
    [](double,double){ return 0.0; }, [](double,double){ return 0.0; }, [](double,double){ return 0.0; }, [](double,double){ return 0.0; });
    int iters = solver.solve();

    double l2err = solver.computeL2Error(exact);

    if (rank == 0) {
        std::cout << " Iter  : " << iters << "\n" << " Solve time  : " << std::fixed << std::setprecision(4)
        << solver.get_solveTime() << " s\n" << " L2 error : " << std::scientific << std::setprecision(4) << l2err << "\n";
    }

    std::ostringstream oss;
    oss << "solution_n" << n << "_p" << size << ".vtk";
    solver.exportVTK(oss.str());
    MPI_Finalize();
    return 0;
}