#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "Instance.h"
#include "configGPU.h"

#include "gpulib/types.h"

EXTERN_C_BEGIN
typedef int TcostFinal;
typedef unsigned short int Ts;
typedef short int TresUsage;

typedef short int Tpos;
typedef short int Top;
typedef short int TSizeChain;
typedef int Tdelta;
typedef struct{
	TcostFinal *costFinal;
    Ts *s;
    TresUsage *resUsage;
}Solution;

typedef struct{
	Tpos *pos;
	Top *op;
	TSizeChain *sizeChain;
	Tdelta *delta;
}EjectionChain;

Solution* allocationPointersSolution(Instance *inst, int nBlocks);

void freePointersSolution(Solution *sol);

EjectionChain* allocationPointerEjectionChain(Instance *inst, int nBlocks, int nThreads);

void freePointerEjectionChain(EjectionChain *ejection);

void createOutputFileSolution(Solution *sol, Instance *inst,int pos_best, const char *fileName);

void createOutputFileFrequencyVersion1(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName);

void createOutputFileFrequencyVersion2(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName);

int returnIndice(Solution *h_solution, EjectionChain *h_ejection, int block, int menor,int *h_long_list, int nJobs,int mAgents, int nThreads);

Solution* createGPUsolution(Solution* h_solution,TnJobs nJobs, TmAgents mAgents, int nBlocks);

EjectionChain* createGPUejection(EjectionChain* h_ejection,TnJobs nJobs, TmAgents mAgents, int nBlocks, int nThreads);

EXTERN_C_END

#endif
