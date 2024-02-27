nvcc --compiler-options "-Wall -pg -g -lpthread -fopenmp -lgomp" -O3 TS-GAP.cu gInstance.cu gSolution.cu Instance.c greedy.c Solution.c -o GTS
