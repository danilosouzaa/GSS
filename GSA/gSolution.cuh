#ifndef GSOLUTION_CUH_
#define GSOLUTION_CUH_

#include "gpulib/gpu.cuh"
#include <curand.h>
#include <curand_kernel.h>

extern "C" {

#include "Instance.h"
#include "Solution.h"
}

__global__ void TS_GAP(Instance *inst, Solution *sol,EjectionChain *ejection, int *tabuListshort,unsigned int *seed, curandState_t* states, int iteration, int n_busca);

#endif /* GSOLUTION_CUH_ */
