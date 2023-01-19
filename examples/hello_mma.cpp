#include "MMA.h"

int main() {
  PetscInt n = 1, m = 2;
  Vec x;
  MMA mma(n, m, x);
  return 0;
}