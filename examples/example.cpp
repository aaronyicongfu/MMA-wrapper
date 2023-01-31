#include "MMA.h"

/**
 * @brief Helper function: allocate the petsc vector x
 *
 * @param x vector
 * @param comm MPI global communicator
 * @param gsize global vector size
 * @param lsize local vector size, determined by petsc by default
 * @return PetscErrorCode
 */
PetscErrorCode allocate_petsc_vec(Vec* x, const MPI_Comm comm,
                                  const PetscInt gsize,
                                  const PetscInt lsize = PETSC_DECIDE) {
  // Create an empty vector object
  PetscCall(VecCreate(comm, &(*x)));

  // Set global and optionally local size
  PetscCall(VecSetSizes(*x, lsize, gsize));

  // This vector is an MPI vector
  PetscCall(VecSetType(*x, VECMPI));

  return 0;
}

/**
 * @brief An abstract optimization problem class, to perform optimization, user
 * needs to implement the following functions:
 *
 *  - evalObjCon()
 *  - evalObjConGrad()
 *
 * Note: we assume that the local partitions of global dv, obj gradient and
 * constraint gradients are partitioned in the same way, i.e. local sizes of
 * these vectors are the same
 */
class OptProblem {
 public:
  /**
   * @param nvars global design variable vector size
   * @param nvars_l local design variable vector size
   * @param ncons number of constraints
   */
  OptProblem(MPI_Comm comm, PetscInt nvars, PetscInt nvars_l, PetscInt ncons)
      : comm(comm), nvars(nvars), nvars_l(nvars_l), ncons(ncons){};

  virtual PetscErrorCode evalObjCon(const Vec x, PetscScalar* obj,
                                    PetscScalar* cons) = 0;
  virtual PetscErrorCode evalObjConGrad(const Vec x, Vec g, Vec* gcon) = 0;

  const PetscInt get_num_cons() const { return ncons; }
  const PetscInt get_num_vars() const { return nvars; }
  const PetscInt get_num_vars_local() const { return nvars_l; }

 protected:
  MPI_Comm comm;
  PetscInt nvars;    // global number of dvs
  PetscInt nvars_l;  // local number of dvs
  PetscInt ncons;    // number of constraints
};

class ToyProblem : public OptProblem {
 public:
  ToyProblem(MPI_Comm comm, PetscInt nvars, PetscInt nvars_l)
      : OptProblem(comm, nvars, nvars_l, 1) {}

  PetscErrorCode evalObjCon(const Vec x, PetscScalar* obj, PetscScalar* cons) {
    // Zero-out values
    *obj = 0.0;
    for (int i = 0; i < ncons; i++) {
      cons[i] = 0.0;
    }

    // Get vector values
    PetscScalar* xvals;
    PetscCall(VecGetArray(x, &xvals));

    // evaluate local objective contribution
    PetscScalar _obj = 0.0;
    for (int i = 0; i < nvars_l; i++) {
      _obj += xvals[i] * xvals[i];
    }

    // evaluate local constraint contribution
    PetscScalar _con = 0.0;
    for (int i = 0; i < nvars_l; i++) {
      _con += xvals[i];
    }

    // MPI reduce
    MPI_Allreduce(&_obj, obj, 1, MPIU_SCALAR, MPI_SUM, comm);
    MPI_Allreduce(&_con, cons, ncons, MPIU_SCALAR, MPI_SUM, comm);

    // Formulate constraint as c(x) >= 0
    cons[0] = 1.0 - cons[0];

    // in pair with VecGetArray()
    PetscCall(VecRestoreArray(x, &xvals));

    return 0;
  }

  PetscErrorCode evalObjConGrad(const Vec x, Vec g, Vec* gcon) {
    // Get vector values
    PetscScalar *xvals, *gvals, *gcvals;
    PetscCall(VecGetArray(x, &xvals));
    PetscCall(VecGetArray(g, &gvals));
    PetscCall(VecGetArray(gcon[0], &gcvals));

    // evaluate objective and constraint gradients
    for (int i = 0; i < nvars_l; i++) {
      gvals[i] = 2.0 * xvals[i];
      gcvals[i] = -1.0;
    }

    // in pair with VecGetArray()
    PetscCall(VecRestoreArray(x, &xvals));
    PetscCall(VecRestoreArray(g, &gvals));
    PetscCall(VecRestoreArray(gcon[0], &gcvals));

    return 0;
  }
};

PetscErrorCode optimize(MPI_Comm comm, int niter) {
  int mpirank, mpisize;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  printf("optimize(): [%d]size: %d\n", mpirank, mpisize);

  // hard code
  int nvars = 1000;

  // Determine local size
  int nvars_l = nvars / mpisize;
  for (int i = 0; i < nvars % mpisize; i++) {
    nvars_l++;  // Adjust if not divided evenly
  }

  // Create the optimization problem driver
  ToyProblem* prob = new ToyProblem(comm, nvars, nvars_l);
  PetscInt ncons = prob->get_num_cons();

  double movelim = 0.2;
  double lb = 0.0, ub = 1.0, x0 = 0.0;

  // Allocate design variable and gradient vectors
  Vec x, g;
  Vec* gcon = new Vec[ncons];
  PetscCall(allocate_petsc_vec(&x, comm, nvars));
  PetscCall(allocate_petsc_vec(&g, comm, nvars));
  for (PetscInt i = 0; i < ncons; i++) {
    PetscCall(allocate_petsc_vec(&gcon[i], comm, nvars));
  }

  PetscCall(VecSet(x, x0));

  // Allocate and initialize bounds
  Vec lbvec, ubvec;
  PetscCall(allocate_petsc_vec(&lbvec, comm, nvars));
  PetscCall(allocate_petsc_vec(&ubvec, comm, nvars));
  PetscCall(VecSet(lbvec, lb));
  PetscCall(VecSet(ubvec, ub));

  // Allocate obj and cons
  PetscScalar obj;
  PetscScalar* cons = new PetscScalar[ncons];

  // Allocate mma operator
  MMA* mma = new MMA(nvars, ncons, x);

  // Optimization loop body
  int iter = 0;
  while (iter < niter) {
    // Evaluate functions and gradients
    prob->evalObjCon(x, &obj, cons);
    prob->evalObjConGrad(x, g, gcon);

    // Set move limits
    PetscCall(mma->SetOuterMovelimit(lb, ub, movelim, x, lbvec, ubvec));

    // Update design
    mma->Update(x, g, cons, gcon, lbvec, ubvec);

    // Check infty norm of dv change, if needed
    // ...

    // Check KKT error
    PetscScalar kkterr_l2, kkterr_linf;
    mma->KKTresidual(x, g, cons, gcon, lbvec, ubvec, &kkterr_l2, &kkterr_linf);

    // Compute dv l1 norm
    PetscScalar x_l1;
    PetscCall(VecNorm(x, NORM_1, &x_l1));

    // Print out
    if (iter % 10 == 0) {
      PetscCall(PetscPrintf(comm, "\n%6s%20s%20s%20s%20s\n", "iter", "obj",
                            "KKT_l2", "KKT_linf", "|x|_1"));
    }
    PetscCall(PetscPrintf(comm, "%6d%20.10e%20.10e%20.10e%20.12e\n", iter, obj,
                          kkterr_l2, kkterr_linf, x_l1));

    iter++;
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&g));
  for (PetscInt i = 0; i < ncons; i++) {
    PetscCall(VecDestroy(&gcon[i]));
  }

  PetscCall(VecDestroy(&lbvec));
  PetscCall(VecDestroy(&ubvec));

  delete[] cons;
  delete[] gcon;
  delete prob;
  delete mma;

  return 0;
}

PetscErrorCode test_optimization(int argc, char* argv[]) {
  const char help[] = "A simple optimization example using MMA";

  // Initialize PETSc / MPI and pass input arguments to PETSc
  PetscCall(PetscInitialize(&argc, &argv, PETSC_NULL, help));

  // Test MPI
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  printf("[%d]size: %d\n", rank, size);

  // Optimize
  int niter = 20;
  optimize(comm, niter);

  // Finalize PETSc / MPI
  PetscCall(PetscFinalize());

  return 0;
}

int main(int argc, char* argv[]) {
  test_optimization(argc, argv);
  return 0;
}