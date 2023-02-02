#ifndef OPTIMIZER_H
#define OPTIMIZER_H

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
                                  const PetscInt gsize, const PetscInt lsize);

/**
 * @brief An abstract optimization problem class, to perform optimization, user
 * needs to implement the following methods:
 *
 *  - evalObjCon()
 *  - evalObjConGrad()
 *
 * Note1: we assume that the local partitions of global dv, obj gradient and
 * constraint gradients are partitioned in the same way, i.e. local sizes of
 * these vectors are the same.
 *
 * Note2: Assume constraints take the following form:
 *        c(x) >= 0
 */
class OptProblem {
 public:
  /**
   * @param nvars global design variable vector size
   * @param nvars_l local design variable vector size
   * @param ncons number of constraints
   */
  OptProblem(MPI_Comm comm, int nvars, int nvars_l, int ncons)
      : comm(comm), nvars(nvars), nvars_l(nvars_l), ncons(ncons){};

  /**
   * @brief Destructor
   *
   * Note: Define it as virtual as we might need to delete object using base
   * type pointer
   */
  virtual ~OptProblem() = default;

  virtual PetscErrorCode evalObjCon(const double* x, double* obj,
                                    double* cons) = 0;
  virtual PetscErrorCode evalObjConGrad(const double* x, double* g,
                                        double** gcon) = 0;

  inline const MPI_Comm get_mpi_comm() const { return comm; }
  inline const int get_num_cons() const { return ncons; }
  inline const int get_num_vars() const { return nvars; }
  inline const int get_num_vars_local() const { return nvars_l; }

 protected:
  MPI_Comm comm;
  int nvars;    // global number of dvs
  int nvars_l;  // local number of dvs
  int ncons;    // number of constraints
};

/**
 * @brief The optimizer
 */
class Optimizer final {
 public:
  Optimizer(OptProblem* prob);
  ~Optimizer();

  /**
   * @brief Perform optimization
   *
   * @param prob optimization problem instance
   * @param niter number of iterations
   */
  PetscErrorCode optimize(int niter);

 private:
  OptProblem* prob;
  double obj, *cons;  // objective and constraint values
  Vec x, g;           // design variable and objective gradient
  Vec* gcon;          // constraint function gradients
  double** gconvals;  // array of constraint gradients
};

#endif