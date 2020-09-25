#include <mkl.h>
#include "distribution.h"


//vectorize this function based on instruction on the lab page                                                                         
int diffusion(const int n_particles,
              const int n_steps,
              const float x_threshold,
              const float alpha,
              VSLStreamStatePtr rnStream) {
  float rn[n_particles];
  float pos[n_particles];
  int n_escaped=0;
  pos[0:n_particles] = 1.0f;
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, rn, -1.0, 1.0);

  for (int j = 0; j < n_steps; j++) {
    #pragma omp simd
    for (int i = 0; i < n_particles; i++) {
      pos[i] += dist_func(alpha, rn[i]);
      n_escaped = pos[i] > x_threshold ? n_escaped++ : n_escaped;
    }
  }
  return n_escaped;
}
