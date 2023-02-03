#include "MMA.h"
#include "optimizer.h"

class ToyProblem : public OptProblem {
 public:
  ToyProblem(MPI_Comm comm, int nvars, int nvars_l)
      : OptProblem(comm, nvars, nvars_l, 1) {}

  int evalObjCon(const double* xvals, PetscScalar* obj, PetscScalar* cons) {
    // Zero-out values
    *obj = 0.0;
    for (int i = 0; i < ncons; i++) {
      cons[i] = 0.0;
    }

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

    return 0;
  }

  int evalObjConGrad(const double* xvals, double* gvals, double** gconvals) {
    // evaluate objective and constraint gradients
    for (int i = 0; i < nvars_l; i++) {
      gvals[i] = 2.0 * xvals[i];
      gconvals[0][i] = -1.0;
    }

    return 0;
  }
};

int main(int argc, char* argv[]) {
  const char help[] = "A simple optimization example using MMA";

  // Initialize PETSc / MPI and pass input arguments to PETSc
  PetscCall(PetscInitialize(&argc, &argv, PETSC_NULL, help));
  MPI_Comm comm = MPI_COMM_WORLD;

  // Create problem
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int nvars = 1000;  // hard-code
  int nvars_l = nvars / size;
  for (int i = 0; i < nvars % size; i++) {
    nvars_l++;  // Adjust if not divided evenly
  }

  ToyProblem prob(comm, nvars, nvars_l);

  // Optimize
  int niter = 20;
  Optimizer* opt = new Optimizer(&prob);
  opt->optimize(niter);

  delete opt;

  // Finalize PETSc / MPI
  PetscCall(PetscFinalize());

  return 0;
}