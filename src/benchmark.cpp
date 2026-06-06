/**
 * @brief
 *  
 * Scalability over grid sizes n = 2^k, n = 16..256.
 *
 * For each grid size, constructs a JacobiSolver instance and measures
 * solve time and L2 error against the exact solution
 *
 * Tolerance and maximum iterations scale with h to ensure convergence
 * reaches the discretisation error level at each resolution.
 * 
 */

#include <cmath>
#include <iostream>
#include <iomanip>
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

    auto f = [](double x, double y) {
        return 8.0 * PI * PI * std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
    };
    auto f_exact = [](double x, double y) {
        return std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
    };

    if (rank == 0) {
        std::cout << "MPI ranks: " << size;
        std::cout << "\nOpenMP threads: " << omp_get_max_threads();
        std::cout << "\n" << std::string(62, '-') << "\n" << std::setw(6)  << "n" << std::setw(10) << "h"
        << std::setw(10) << "tol" << std::setw(8)  << "iters" << std::setw(12) << "time(s)" << std::setw(14) << "L2_error"
        << "\n" << std::string(62, '-') << "\n";
        std::cout.flush();
    }

    for (int k = 4; k <= 8; ++k) {

        const int n = static_cast<int>(std::pow(2, k));         
        const double h = 1.0 / (n - 1);

        //Here i used a really aggressive tolerance to ensure a decrease in L2 error
        const double tol = std::pow(h, 4) * 0.1;

        const int maxIter = static_cast<int>(3.0 / (h * h));

        
        JacobiSolver solver(n, f, tol, maxIter, [](double,double){ return 0.0; }, [](double,double){ return 0.0; },
        [](double,double){ return 0.0; }, [](double,double){ return 0.0; });

        int iters = solver.solve();
        double err = solver.computeL2Error(f_exact);

        if (rank == 0) {
            std::cout << std::setw(6)  << n << std::setw(10) << std::scientific << std::setprecision(2) << h
            << std::setw(10) << std::scientific << std::setprecision(1) << tol << std::setw(8)  << std::fixed << iters
            << std::setw(12) << std::fixed << std::setprecision(4) << solver.get_solveTime() << std::setw(14) << std::scientific << std::setprecision(4) << err << "\n";
            std::cout.flush();
        }
    }
    if (rank == 0)
        std::cout << std::string(62, '-') << "\n";
    MPI_Finalize();
    return 0;
}