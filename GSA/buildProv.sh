nvcc --compiler-options "-Wall -pg -g -lpthread -fopenmp -lgomp" -O3 mainProv.cpp gaproblem.cpp gapinstance.cpp except.cpp log.cpp utils.cpp -o testInstance