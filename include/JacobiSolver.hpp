#pragma once
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <mpi.h>

/**
 * @brief Parallel Jacobi solver for the Laplace equation
 *
 * Distributes rows of the n×n grid across all the MPI ranks.
 * Each rank gets a band of rows and exchanges ghost rows
 * with its neighbours before each Jacobi sweep.
 * OpenMP is used to parallelize the inner loops.
 */
class JacobiSolver {
public:
    using Force = std::function<double(double, double)>;
    using Boundary = std::function<double(double, double)>;

    /**
     * @brief Constructor of the class
     * 
     * @param n Number of grid points along each axix, boundaries included
     * @param f Forcing term 
     * @param tol Convergence tolerance on the L2 increment norm
     * @param maxIter Maximum number of Jacobi iterations
     * @param bc_bottom Dirichlet BC on y=0, default: 0
     * @param bc_top Dirichlet BC on y=1, default: 0
     * @param bc_left Dirichlet BC on x=0, default: 0
     * @param bc_right Dirichlet BC on x=1, default: 0
     */
    JacobiSolver(int n,
                 Force    f,
                 double tol = 1e-6,
                 int maxIter = 500000,
                 Boundary bc_bottom = [](double,double){ return 0.0; },
                 Boundary bc_top = [](double,double){ return 0.0; },
                 Boundary bc_left = [](double,double){ return 0.0; },
                 Boundary bc_right = [](double,double){ return 0.0; }
                 );

    /**
     * @brief Function to solve the Laplace equation using Jacobi solver
     * 
     * @return Number of iterations
     */
    int solve();

    /**
     * @brief Export solution to a VTK-style file.
     * Rank 0 gathers all local solutions and writes the file.
     * 
     * @param filename  Output path of the file
     */

    void exportVTK(const std::string& filename) const;

    /**
     * @brief Compute the L2 error with respect to an exact solution.
     * Rank 0 collects all partial contributions and returns the global norm.
     * @param exact  Exact solution
     */

    double computeL2Error(const std::function<double(double,double)>& exact) const;

    /**
     * @brief Various getters
     */  

    double get_solveTime() const { return solveTime; }
    int get_rank()   const { return rank; }
    int nprocs() const { return size; }

private:
    int n, maxIter;        
    double h, h2, tol;
    Force    f;
    Boundary bc_bottom, bc_top, bc_left, bc_right;
    int rank, size, rowStart, rowEnd, localRows;  
    std::vector<double> U;   
    std::vector<double> Unew; 
    double solveTime = 0.0;

    /** @brief Compute the row decomposition among MPI ranks.
    * Each rank gets a contiguous band of rows as equal as possible.
    */       
    void   computeDecomposition();

    /** @brief Initialize the grid with boundary condition values.
    * Interior nodes are set to zero; boundary nodes are set according to the user boundary functions.
    */
    void   initializeGrid();

    /** @brief Exchange ghost rows with neighbouring MPI ranks.
    */
    void   exchangeGhostRows();

    /** @brief Perform one Jacobi sweep over the local rows.
    * Updates Unew from U using the 5-point stencil.
    * OpenMP parallelizes the outer loop over local rows.
    * @return Local sum of squared increments.
    */
    double localJacobiSweep(); 
    
    /** @brief Reduce local squared increment sums across all ranks
    * and return the global weighted L2 norm of the increment.
    * @param localSqSum Local sum of squared increments.
    * @return Global convergence norm
    */
    double globalConvergence(double localSqSum) const;

    /** @brief Linear index into the extended local array.
    * @param localRow Local row index, with 0 = bottom ghost, localRows+1 = top ghost.
    * @param col Column index.
    * @return Flat index into U or Unew.
    */
    inline int idx(int localRow, int col) const {
        return localRow * n + col;
    }

    /** @brief x-coordinate for column index j.
    *@param j  Column index.
    *@return   x = j * h
    */
    inline double xCoord(int j) const { return j * h; }

    /** @brief y-coordinate for global row index i.
    * @param i  Global row index.
    * @return   y = i * h
    */
    inline double yCoord(int i) const { return i * h; }
};