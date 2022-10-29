#include "mpi.h"
#include <iostream>

// fills number of processes and process ID
void whoami(int& numprocs, int& myid) {
  int ierr;
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (ierr != 0) {
    std::cerr << " error in MPI_Comm_size = " << ierr << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (ierr != 0) {
    std::cerr << " error in MPI_Comm_rank = " << ierr << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}