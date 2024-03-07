// Copyright Eyder Rios, 2015
// MIT License

/**
 * @file	gapkernel.h
 *
 * @brief   Handle a kernel monitoring thread.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef SOURCE_GAPKERNEL_H_
#define SOURCE_GAPKERNEL_H_

#include <pthread.h>
//
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
//
#include "gpulib/gpu.h"
//#include "./graph.hpp"
#include "./gapads.h"
#include "./gapsolution.h"
#include "gpulib/types.h"
#include "./utils.h"

// ################################################################################
// //
// ## ## //
// ##                               CONSTANTS & MACROS ## //
// ## ## //
// ################################################################################
// //

/*
 * GPU macros
 */

#define GPU_DIST_COORD(i, j)                                 \
  int(sqrtf((float)(GPU_SQR(sm_coordx[i] - sm_coordx[j]) +   \
                    GPU_SQR(sm_coordy[i] - sm_coordy[j]))) + \
      0)  // TODO: PUT THIS VALUE sm_round=0 IN INSTANCE, EYDER... PLEASE!!!!

#define GPU_DIST_COORD_GENERIC(i, j, varx, vary)                              \
  int(sqrtf(                                                                  \
          (float)(GPU_SQR(varx[i] - varx[j]) + GPU_SQR(vary[i] - vary[j]))) + \
      0)  // TODO: PUT THIS VALUE sm_round=0 IN INSTANCE, EYDER... PLEASE!!!!

#define GPU_MOVE_PACKID(i, j, s) ((uint(j) << 18) | (uint(i) << 4) | uint(s))
#define GPU_MOVE_PACK64(hi, lo) ((ullong(hi) << 32) | ullong(lo))

#define GPU_MOVE_ID(m) ((m))
#define GPU_MOVE_I(m)
#define GPU_MOVE_J(m)

#define GAPKF_COPY_NONE 0
#define GAPKF_COPY_SOLUTION 1
#define GAPKF_COPY_MOVES 2

// ################################################################################
// //
// ## ## //
// ##                                GLOBAL VARIABLES ## //
// ## ## //
// ################################################################################
// //

// ################################################################################
// //
// ## ## //
// ##                                  DATA TYPES ## //
// ## ## //
// ################################################################################
// //

/*!
 * Incomplete classes
 */
// class GAPGPUTask;
class GAPKernel;

class GAPSolution;

/*!
 * PGAPKernelTask
 */
typedef GAPKernel* PGAPKernel;

/*!
 * GAPMoveGraph
 */
typedef Graph<GAPMove64*> GAPMoveGraph;

// ################################################################################
// //
// ## ## //
// ##                                 GAPKernel ## //
// ## ## //
// ################################################################################
// //

// NOLINTNEXTLINE
class GAPKernel {
  // protected:
 public:
  //    GAPGPUTask      &gpuTask;                        ///< GPU task related to
  //    this kernel
  GAProblem& problem;  ///< Instance problem

  int id;            ///< Kernel id
  uint tag;          ///< Extra tag (used as K value)
  const char* name;  ///< Task name

  dim3 grid;    ///< Kernel grid
  dim3 block;   ///< Kernel block
  uint shared;  ///< Kernel shared memory
  ullong time;  ///< Last execution time

  cudaStream_t stream;   ///< Stream for current kernel
  cudaEvent_t evtStart;  ///< GPU start event
  cudaEvent_t evtStop;   ///< GPU stop  event

  uint solSize;        ///< Solution size
  GAPADSData* adsData;  ///< ADS data buffer
  uint adsDataSize;    ///< ADS data buffer size in bytes
  uint adsRowElems;    ///< Number of elements of an eval matrix row

  GAPMovePack* moveData;  ///< Solution buffer
  uint moveDataSize;     ///< Solution buffer size in bytes
  uint moveElems;        ///< Number of move elements

  GAPPointer transBuffer;  ///< Data transfer buffer
  uint transBufferSize;   ///< Data transfer buffer size in bytes

  GAPSolution* solution;  ///< Kernel solution
  bool solDestroy;       ///< Should kernel destroy solution?

  GAPMoveGraph graphMerge;  ///< Move merge graph
  uint maxMerge;           ///< Maximum number of elements to merge

  int flagOptima;  ///< Flag for local optima condition
  int flagExec;    ///< Flag for kernel running

  uint callCount;   ///< Kernel calls counter
  uint mergeCount;  ///< Number of move merges
  uint imprvCount;  ///< Number of improvements
  ullong timeMove;  ///< Time spent processing moves

  bool isTotal;

 private:
  /*!
   * Reset class fields.
   */
  void reset();

 public:
  /*!
   * Create a GAPKernelTask instance.
   */
  // GAPKernel(GAPGPUTask &parent, int kid, uint ktag = 0);
  GAPKernel(GAPProblem& _problem, bool _isTotal, int kid, uint ktag = 0);
  /*!
   * Destroy a GAPKernelTask instance.
   */
  virtual ~GAPKernel();
  /*!
   * Prepare GPU device for execution allocating and initializing resources.
   */
  void init(bool solCreate);
  /*!
   * Terminate device releasing any allocated resources.
   */
  void term();
  /*!
   * Launch an empty kernel.
   */
  void launchEmptyKernel();
  /*!
   * Launch sleep kernel.
   */
  void launchSleepKernel(ullong sleep = 100000);
  /*!
   * Launch test kernel.
   */
  void launchTestKernel();
  /*!
   * Launch an empty kernel.
   */
  void launchShowDataKernel(uint width, uint max = 0);
  /*!
   * Launch distance matrix kernel.
   */
  void launchWeightKernel();
  /*!
   * Launch kernel to compute ADS checksum.
   */
  void launchChecksumKernel(uint max = 0, bool show = false);
  /*!
   * Show distance matrix
   */
  void launchShowDistKernel();

  /*!
   * Reset kernel calls counter.
   */
  inline void resetStats() {
    callCount = 0;
    mergeCount = 0;
    imprvCount = 0;
    timeMove = 0;
  }
  /*!
   * Set base solution
   */
  inline void setSolution(GAPSolution* sol, bool ads = true) {
    solution->assign(sol, ads);
  }
  /*!
   * Get base solution
   */
  inline void getSolution(GAPSolution* sol, bool ads = true) {
    sol->assign(solution, ads);
  }
  /*!
   * Copy kernel solution from host to device.
   */
  inline void sendSolution() {
    // Copy solution to GPU
    // TODO adsDataSize
    // gpuMemcpyAsync(adsData,solution->adsData,solution->adsDataSize,cudaMemcpyHostToDevice,stream);
    gpuMemcpyAsync(adsData, solution->adsData, adsDataSize,
                   cudaMemcpyHostToDevice, stream);
  }

  /*!
   * Receive kernel result.
   */
  void recvResult() {
    // Copy results from GPU
    gpuMemcpyAsync(transBuffer.p_void, moveData, moveElems * sizeof(GAPMove64),
                   cudaMemcpyDeviceToHost, stream);
  }

  struct OBICmp {
    __host__ __device__ bool operator()(const GAPMovePack& o1,
                                        const GAPMovePack& o2) {
      const GAPMove64& mo1 = (const GAPMove64&)o1;
      const GAPMove64& mo2 = (const GAPMove64&)o2;
      return mo1.cost < mo2.cost;
    }
  };

  void mergeGPU();
  /*!
   * Synchronize with kernel stream.
   */
  inline void sync() { gpuStreamSynchronize(stream); }
  /*!
   * Set time start event.
   */
  inline void timerStart() { gpuEventRecord(evtStart, stream); }
  /*!
   * Set time end event.
   */
  inline void timerStop() { gpuEventRecord(evtStop, stream); }
  /*!
   * Get elapsed time between a call of timeStart() and timeStop()
   */
  inline ullong timerElapsed() {
    float time{0.0};
    gpuEventSynchronize(evtStop);
    gpuEventElapsedTime(&time, evtStart, evtStop);
    return MS2US(time);
  }
  /*!
   * Add callback function for stream completion
   */
  inline void addCallback(cudaStreamCallback_t func) {
    gpuStreamAddCallback(stream, func, (void*)this, 0);
  }
  /*!
   * Get best movement from buffer
   *
   * @return   Returns the best move improvement
   */
  int bestMove(GAPMove& move);
  /*!
   * Compute best independent movements (greedy algorithm)
   *
   * @param    merge  Buffer to receive mergeable moves
   * @param    count  Number of moves in merge buffer
   *
   * @return   Returns the sum of cost of all merged moves.
   */
  int mergeGreedy(GAPMove64* merge, int& count);
  /*!
   * Define kernel launching grid
   */
  virtual void defineKernelGrid() = 0;
  /*!
   * Launch kernel for SWAP local search
   */
  virtual void launchKernel() = 0;
  /*!
   * Apply movement on current base solution.
   */
  virtual void applyMove(GAPMove& move) = 0;
  /*!
   * Friend classes
   */
  // friend class  GAPSolver;
  // friend class  GAPTask;
  // friend class  GAPGPUTask;
  friend class GAPKernelSharedSize;
  friend struct GAPKernelData;
};

// ################################################################################
// //
// ## ## //
// ##                             GAPKernelSharedSize ## //
// ## ## //
// ################################################################################
// //

/*!
 * Functor for shared memory size calculation
 */
class GAPKernelSharedSize {
 private:
  GAPKernel* kernel;

 public:
  explicit GAPKernelSharedSize(GAPKernel* krn) : kernel{krn} {}
  int operator()(int blockSize) {
    int size = 3 * GPU_BLOCK_CEIL(int, kernel->solSize) +
               2 * GPU_BLOCK_CEIL(uint, blockSize);

    //        lprintf("%s: solSize = %u, blockSize = %d\n",
    //                        kernel->kernelName,kernel->solSize,blockSize);
    //        lprintf("%s: coords = 2 x %u x sz(int)=%lu = %lu\n",
    //                        kernel->kernelName,kernel->solSize,
    //                        sizeof(int),
    //                        2*GPU_BLOCK_CEIL(int,kernel->solSize));
    //        lprintf("%s: soldata= 1 x %u x sz(int)=%lu = %lu\n",
    //                        kernel->kernelName,kernel->solSize,
    //                        sizeof(int),
    //                        1*GPU_BLOCK_CEIL(int,kernel->solSize));
    //        lprintf("%s: moves  = 2 x %u x sz(uint)=%lu = %lu\n",
    //                        kernel->kernelName,blockSize,
    //                        sizeof(uint),
    //                        2*GPU_BLOCK_CEIL(uint,blockSize));

    return size;
  }
};

#endif  // SOURCE_GAPKERNEL_H_
