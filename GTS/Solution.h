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

Solution* allocationPointersSolution(Instance *inst);

EjectionChain* allocationPointerEjectionChain(Instance *inst);

void create_solution(Solution *sol, Instance *inst,int pos_best, const char *fileName);

void create_frequency(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName);

void create_frequency_2(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName);

int returnIndice(Solution *h_solution, EjectionChain *h_ejection, int block, /*int nBlocks, int nThreads,*/ int menor,int *h_long_list, int nJobs,int mAgents);

Solution* createGPUsolution(Solution* h_solution,TnJobs nJobs, TmAgents mAgents);

EjectionChain* createGPUejection(EjectionChain* h_ejection,TnJobs nJobs, TmAgents mAgents);

EXTERN_C_END

#endif
