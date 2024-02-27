#ifndef INSTANCE_H
#define INSTANCE_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "gpulib/types.h"

EXTERN_C_BEGIN


typedef int TnJobs;
typedef int TmAgents;
typedef int Tcost;
typedef short int TresourcesAgent;
typedef short int Tcapacity;

typedef struct{

	  TnJobs nJobs; /* number of jobs */
	  TmAgents mAgents; /* number of agents */
	  Tcost *cost;/* Matrix with cost of each job j allocated for agent i(nXm)*/
	  TresourcesAgent *resourcesAgent; /* matrix with amount of resources that each jobs j consumes when performed by the agent i (nxm)*/
	  Tcapacity *capacity; /*vector with capacity of agent's*/
}Instance;

int iReturn(int i, int j, int n, int m );

Instance* allocationPointersInstance(int nJobs, int mAgents);

void freePointersInstance(Instance *inst);


Instance* loadInstance(const char *fileName);

void showInstance(Instance *inst);

Instance* createGPUInstance(const Instance* h_instance, TnJobs nJobs, TmAgents);
EXTERN_C_END

#endif
