#ifndef INSTANCE_CUH_
#define INSTANCE_CUH_

#include "gpulib/gpu.cuh"

extern "C" {

#include "Instance.h"
#include "Solution.h"
#include "configGPU.h"

}

// INCLUDE PROTOTYPE OF CUDA functions __device__ and __global__
Instance* createGPUInstance(Instance* h_instance, TnJobs nJobs, TmAgents mAgents);




#endif /* INSTANCE_CUH_ */
