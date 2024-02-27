#ifndef GREEDY_H
#define GREEDY_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "Instance.h"
#include "Solution.h"
#include "configGPU.h"

EXTERN_C_BEGIN

int* inicializeVector(Instance *inst, float p1, float p2);

int* createAssigmentPriorityVector(Instance *inst, int job, float p1, float p2);

int quicksortCof(float *values, int *idc, int began, int end);

int greedy(Instance *inst,Solution *sol, float p1, float p2,int block);



EXTERN_C_END
#endif
