#include "MMA.h"

PetscErrorCode optimize(int niter) {
  int ncons = 1;
  int nvars = 1000;

  // Create design variable
  Vec x;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, nvars));
  PetscCall(VecSetFromOptions(x));

  MMA* mma = new MMA(nvars, ncons, x);

  int iter = 0;
  while (iter < niter) {
    iter++;

    // Evaluate functions and gradients
    // ...

    // Set move limits
    // ...

    // Update design
    // ...

    // Check infty norm of dv change, if needed
    // ...
  }

  delete mma;

  return 0;
}

int main(int argc, char* argv[]) {
  const char help[] = "A simple optimization example using MMA";

  // Initialize PETSc / MPI and pass input arguments to PETSc
  PetscCall(PetscInitialize(&argc, &argv, PETSC_NULL, help));

  // Optimize
  int niter = 20;
  optimize(niter);

  // Finalize PETSc / MPI
  PetscCall(PetscFinalize());

  return 0;
}