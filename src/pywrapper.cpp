#include <pybind11/pybind11.h>

#include "MMA.h"
#include "optimizer.h"

namespace py = pybind11;

class PyOptProblem : public OptProblem {
 public:
  // Inherit constructor
  using OptProblem::OptProblem;

  // Trampoline for each pure virtual function
  PetscErrorCode evalObjCon(const Vec x, PetscScalar* obj,
                            PetscScalar* cons) override {
    PYBIND11_OVERRIDE_PURE(PetscErrorCode,  // return type
                           OptProblem,      // Parent class
                           evalObjCon,      // name of function
                           x,               // Argument(s), if any
                           obj, cons

    );
  }
};

PYBIND11_MODULE(pywrapper, m) {
  py::class_<OptProblem, PyOptProblem>(m, "OptProblem")
      .def(py::init<MPI_Comm, PetscInt, PetscInt, PetscInt>());
}