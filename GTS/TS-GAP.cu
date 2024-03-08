/*
 ============================================================================
 name        : TS-GAP.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */


#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>


#include "gpulib/types.h"
#include "gpulib/gpu.cuh"

#include "Instance.h"
#include "Solution.h"
#include "gSolution.cuh"
#include "greedy.h"

//const int nThreads = 896;
//const int nBlocks = 5;
//const int maxChain = 10;

int main(int argc, char *argv[])
{

	//verify arguments 
	if(argc!=6){
	    printf(RED "Erro: " RESET);
	    printf("\tInvalid number arguments;\n");
	    printf("\tNumber of arguments informed:");
	    printf( BOLDRED "(%i)" RESET ,argc-1);
            printf("\n\tExpected arguments:");
	    printf(BOLDGREEN "(5)\n" RESET);
	    printf("\t\t ./GTS " BOLDGREEN "[1] [2] [3] [4] [5]\n" RESET);
	    printf(BOLDGREEN "\t[1]" RESET ": Instance name;\n");
	    printf(BOLDGREEN "\t[2]" RESET ": Method iteration limit;\n");
	    printf(BOLDGREEN "\t[3]" RESET ": Ejection Chain - Thread limit;\n");
	    printf(BOLDGREEN "\t[4]" RESET ": Best know object function;\n");
	    printf(BOLDGREEN "\t[5]" RESET ": Method runtime limit (minutes).\n");
	    exit(1);
	}


	//Variable with GPU's number
	int deviceCount = 0;
	//Commands for verify use correct of GPU
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount); //valgrindError
	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(1);
	}else{
		if(deviceCount == 0)
		{
			printf("No GPU found :(");
			exit(1);
		}
		else
		{
			printf("Found %d GPUs! GPU 0 initialized!\n", deviceCount);
			gpuSetDevice(0);
		}
	}
	//free(error_id);
	//iterator of use in for
	int i, j;
	//Pointer of states for use in curand (GPU)
	curandState_t *states;
	cudaMalloc((void**)&states, (nThreads*nBlocks)*sizeof(curandState_t));

	//Pointer of seed for use with curand (host)
	unsigned int *h_seed = (unsigned int*)malloc(sizeof(unsigned int)*(nThreads*nBlocks));
	srand(time(NULL));
	for(i=0;i<(nThreads*nBlocks);i++){
		h_seed[i] = rand()%100000;
	}


	//Pointers of intance, solution and ejection for use in CPU(Host) and GPU(Device)
	//Instance *h_instance, *d_instance;
	//Solution *h_solution, *d_solution, *best_solution;
	//EjectionChain *h_ejection, *d_ejection;

	//Load Instance
	char nameAux[50] = "../Instances/";
	const char *temp_teste = argv[1];
	strcat(nameAux,argv[1]);
	const char *fileName = nameAux; //argv[1];
	
	//Create, memory allocation and read instance struct
	Instance *h_instance = loadInstance(fileName);
	float runtimeLimit= atof(argv[5])*60000;// runtime limit (convert minutes from ms)
	
	//showInstance(h_instance);
	//Create and memory allocation from Solution and Ejection structs
	Solution *best_solution = allocationPointersSolution(h_instance); //valgrindError: free
	Solution *h_solution = allocationPointersSolution(h_instance);//valgrindError: free
	EjectionChain *h_ejection = allocationPointerEjectionChain(h_instance); //valgrindError: free
	
	//weight greedy
	float w1,w2;
	struct timeval time_rand;
        struct timeval inicio;
        struct timeval t_inicio;
        struct timeval fim;
        struct timeval t_fim;


	//Generate Initial Solution from greedy method
	gettimeofday(&t_inicio,NULL);
	for(i=0;i<nBlocks;i++){
		
		if(temp_teste[0]=='e'){
			do{ 
				gettimeofday(&time_rand,NULL);
				srand(time_rand.tv_usec);
				for(j=0;j<h_instance->mAgents;j++){
					h_solution->resUsage[j+i*h_instance->mAgents] = 0;
				}
				w1 = (float)(rand())/(float)(RAND_MAX) + 0.5;
				w2 = 19 + w1;
			}while(greedy(h_instance,h_solution,w1,w2,i)==0);
		}else{
			do{
				gettimeofday(&time_rand,NULL);
				srand(time_rand.tv_usec);
				for(j=0;j<h_instance->mAgents;j++){
					h_solution->resUsage[j+i*h_instance->mAgents] = 0;
				}
				w1 = (float)(rand())/(float)(RAND_MAX) + 0.5;
				w2 = 1 + w1;
			}while(greedy(h_instance,h_solution,w1,w2,i)==0);
		}
		
	}
	gettimeofday(&t_fim, NULL);
	float t_aux =   (float) (1000 * (t_fim.tv_sec - t_inicio.tv_sec) + (t_fim.tv_usec - t_inicio.tv_usec) / 1000);
	printf("Time Greedy Random: %.1fms\n", t_aux);
	//best_solution = h_solution;
	//Size Struct Solution
	size_t size_solution =  sizeof(Solution)
		+ sizeof(TcostFinal)*nBlocks
		+ sizeof(Ts)*(h_instance->nJobs*nBlocks)
		+ sizeof(TresUsage)*(h_instance->mAgents*nBlocks);
	if(size_solution<=0){
		printf(BOLDRED "Error: verify nBlocks value and parameters read from the instance.\n" RESET);
	}
	//copies solution initial (greedy) into the structure of the best solution 
	for(i=0;i<nBlocks;i++){
		best_solution->costFinal[i] = h_solution->costFinal[i]; 
		for(j=0;j<h_instance->nJobs;j++){
			best_solution->s[j+i*h_instance->nJobs] = h_solution->s[j+i*h_instance->nJobs];//erroValgrind
		}
		for(j=0;j<h_instance->mAgents;j++){
					best_solution->resUsage[j+i*h_instance->mAgents] = h_solution->resUsage[j+i*h_instance->mAgents];
		}
	}
	//Size Struct of Ejection Chain
	size_t size_ejection = sizeof(EjectionChain)
													+ sizeof(Tpos)*(nBlocks*nThreads*maxChain)
													+ sizeof(Top)*(nBlocks*nThreads)
													+ sizeof(TSizeChain)*(nBlocks*nThreads)
													+ sizeof(Tdelta)*(nBlocks*nThreads);
	if(size_ejection<=0){
		printf(BOLDRED "Error: verify nBlocks and nThreads values.\n" RESET);
	}
	//Size Struct of Instance
	size_t size_instance = sizeof(Instance)
							+ sizeof(Tcost)*(h_instance->nJobs*h_instance->mAgents)  //cost
							+ sizeof(TresourcesAgent)*(h_instance->nJobs*h_instance->mAgents) //resources
							+ sizeof(Tcapacity)*h_instance->mAgents; //capacity
	if (size_instance<=0){
		printf(BOLDRED "Error: check parameters read from the instance.\n" RESET);
	}
	int *h_short_list = (int*)malloc(sizeof(int)*(nBlocks*h_instance->nJobs));
	int *h_long_list = (int*)malloc(sizeof(int)*(h_instance->nJobs*h_instance->mAgents));
	for(i=0;i<nBlocks*h_instance->nJobs;i++){
		h_short_list[i] = 0;
	}
	for(i=0;i<h_instance->nJobs*h_instance->mAgents;i++){
		h_long_list[i] = 0;
	}
	int bestCost = 1000000; //bigM
	for(i=0;i<nBlocks;i++){
		for(j=0;j<h_instance->nJobs;j++){
			h_long_list[j + h_solution->s[j+i*h_instance->nJobs]*h_instance->nJobs]++;
		}
	}

	for(i=0;i<nBlocks;i++){ 
		#ifdef PRINTALL
			printf("Initial cost obtained by the greedy method: %d\n", h_solution->costFinal[i]);
		#endif
		if(h_solution->costFinal[i]< bestCost){
			bestCost = h_solution->costFinal[i]; //update costOut with value of greedy solution found
		}
	}

	

	int *d_short_list; //verify objective of struct
	gpuMalloc((void*)&d_short_list,sizeof(int)*(nBlocks*h_instance->nJobs) );
	gpuMemcpy(d_short_list, h_short_list,sizeof(int)*(nBlocks*h_instance->nJobs), cudaMemcpyHostToDevice);


	// int blockSize;      // The launch configurator returned block size
	// int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
	// int gridSize;
	// int N = 1000000;

	// cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,TS_GAP, 0, N);

	// printf("block size %d\n",blockSize);
	// printf("Min Grid %d\n",minGridSize);
	// 	getchar();



	//Reallocation of pointers Instance and Solution for GPU (device)
	TnJobs tmpNJobs = h_instance->nJobs;
	TmAgents tmpMAgents = h_instance->mAgents;
	Instance *d_instance = createGPUInstance(h_instance, h_instance->nJobs, h_instance->mAgents);
	Solution *d_solution = createGPUsolution(h_solution,tmpNJobs, tmpMAgents);
	EjectionChain *d_ejection = createGPUejection(h_ejection,tmpNJobs, tmpMAgents);

	//Pointers seed in device (GPU)
	unsigned int *d_seed;

	//Event and gpu for contability time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocation of pointer and copy value in d_seed (Device)
	gpuMalloc((void*)&d_seed, sizeof(unsigned int)*(nThreads*nBlocks));
	gpuMemcpy(d_seed, h_seed, sizeof(unsigned int)*(nThreads*nBlocks), cudaMemcpyHostToDevice);
	cudaEventRecord(start);//verify objective of function

	int iterationLimit = atoi(argv[2]); //iteration limit call GTS
	int ite=1; // iteration current
	int searchLimitCall = atoi(argv[3]); //search limit TS in each thread of GPU
	int knowBestSolution = atoi(argv[4]); 
	//int ite_b = 0;//verify mistery????
	int szTabuList; //size Tabu List
	int lowestDelta;//value lowest Delta
	int aux; //auxiliar variable
	
	
	float currentTime = 0.0; //current runtime 
	float timeWOutImp = 0.0; //current time without improvement
	int propSZMaxTabuList = 1.25*(tmpNJobs/(maxChain+1)); //25% more than the proportion of jobs distributed per chain
//	propSZMaxTabuList = 15;
	gettimeofday(&inicio, NULL);
	//size_t freeMem, totalMem;
	gettimeofday(&t_inicio,NULL);
	float timeLimitImprov = 15000; //ms
	while((ite<=iterationLimit)&&(currentTime<=runtimeLimit)&&(timeWOutImp<timeLimitImprov)){
		szTabuList = rand()%propSZMaxTabuList + 1; //random tabu list RANDOM(1,propSZMaxTabuList)
		//verify in globals TS_GAP szTabuList???
		TS_GAP<<<nBlocks,nThreads>>>(d_instance, d_solution,d_ejection, d_short_list, d_seed, states, ite, searchLimitCall);
		gpuDeviceSynchronize();
	//	cudaMemGetInfo(&freeMem, &totalMem);
	//	printf("Free = %zu, Total = %zu\n, size_intance = %zu", freeMem, totalMem,size_instance);
		gpuMemcpy(h_instance, d_instance, size_instance, cudaMemcpyDeviceToHost);
		gpuMemcpy(h_solution, d_solution, size_solution, cudaMemcpyDeviceToHost);
		gpuMemcpy(h_ejection, d_ejection, size_ejection, cudaMemcpyDeviceToHost);
		gpuMemcpy(h_short_list, d_short_list,sizeof(int)*(nBlocks*tmpNJobs), cudaMemcpyDeviceToHost);
		gpuMemcpy(h_seed, d_seed, sizeof(unsigned int)*(nThreads*nBlocks), cudaMemcpyDeviceToHost);
		
		
		//reallocation pointers of Instance
		h_instance->cost = (Tcost*)(h_instance+1);
		h_instance->resourcesAgent =(TresourcesAgent*) (h_instance->cost +(tmpNJobs*tmpMAgents));
		h_instance->capacity =(Tcapacity*) (h_instance->resourcesAgent + (tmpNJobs*tmpMAgents));
		h_instance->nJobs = tmpNJobs;
		h_instance->mAgents = tmpMAgents;
		
		
		//reallocation pointers of Solution
		h_solution->costFinal = (TcostFinal*)(h_solution+1);
		h_solution->s = (Ts*)(h_solution->costFinal + nBlocks);
		h_solution->resUsage = (TresUsage*)(h_solution->s + (h_instance->nJobs*nBlocks));


		//reallocation pointers of Ejection
		h_ejection->pos=(Tpos*)(h_ejection + 1);
		h_ejection->op = (Top*)(h_ejection->pos+ (nBlocks*nThreads*maxChain));
		h_ejection->sizeChain = (TSizeChain*)(h_ejection->op + (nBlocks*nThreads));
		h_ejection->delta = (Tdelta*)(h_ejection->sizeChain + (nBlocks*nThreads));


		//		printf("%d time %f \n",ite,currentTime);
		for(i=0;i<nBlocks;i++){
			lowestDelta = 100000;//bigM
			for(j=0;j<nThreads;j++){
				if(h_ejection->delta[j + i*nThreads]<lowestDelta){
					lowestDelta = h_ejection->delta[j + i*nThreads];
				}
				//printf("value of delta for thread %d in block %d: :%d \n", j, i, h_ejection->delta[j + i*nThreads]);
			}
			lowestDelta = returnIndice(h_solution,h_ejection,i,/*nBlocks,nThreads,*/lowestDelta,h_long_list,h_instance->nJobs,h_instance->mAgents);
			//	printf("lowestDelta delta do bloco %d: %d\n",i,lowestDelta);
			if(h_ejection->op[lowestDelta + i*nThreads]==1){
				int aux1 = h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads];
				//aux2 = ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads];
				h_short_list[aux1 + i*h_instance->nJobs] = ite + szTabuList;

			}else{
					int varAux = rand()%h_ejection->sizeChain[lowestDelta+i*nThreads];
				//for(j = 0; j<h_ejection->sizeChain[lowestDelta + i*nThreads];j++){
					int aux1 = h_ejection->pos[varAux + lowestDelta*maxChain + i*maxChain*nThreads];
					h_short_list[aux1 + i*h_instance->nJobs] = ite + szTabuList;
				//}

			}

			h_solution->costFinal[i] += h_ejection->delta[lowestDelta+i*nThreads];
			if(h_ejection->op[lowestDelta + i*nThreads]==1){
				int t1 = h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads];
				unsigned short int m2 = h_ejection->pos[1 + lowestDelta*maxChain + i*maxChain*nThreads];
				int m1 = ((int)h_solution->s[t1 + i*h_instance->nJobs]);
				h_solution->resUsage[m1 + i*h_instance->mAgents] -= h_instance->resourcesAgent[t1*h_instance->mAgents + m1];
				h_solution->resUsage[m2 + i*h_instance->mAgents] += h_instance->resourcesAgent[t1*h_instance->mAgents + m2];
				h_solution->s[t1 + i*h_instance->nJobs] = (m2);
				//		if(m2>4){
				//			printf("op 1");
				//		}

			}else{
				h_solution->resUsage[((int)h_solution->s[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs]) + i*h_instance->mAgents] += h_instance->resourcesAgent[h_ejection->pos[(h_ejection->sizeChain[lowestDelta + i*nThreads]-1)  + lowestDelta*maxChain + i*maxChain*nThreads]*h_instance->mAgents + ((int)h_solution->s[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs])];
				h_solution->resUsage[((int)h_solution->s[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs]) + i*h_instance->mAgents] -= h_instance->resourcesAgent[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads]*h_instance->mAgents + ((int)h_solution->s[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs])];
				aux = ((int)h_solution->s[h_ejection->pos[0 + lowestDelta*maxChain + i*maxChain*nThreads]+ i*h_instance->nJobs]);
				for(j=1; j<h_ejection->sizeChain[lowestDelta + i*nThreads]; j++){
					h_solution->resUsage[((int)h_solution->s[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs]) + i*h_instance->mAgents] += h_instance->resourcesAgent[h_ejection->pos[(j-1) + lowestDelta*maxChain + i*maxChain*nThreads]*h_instance->mAgents + ((int)h_solution->s[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs])];
					h_solution->resUsage[((int)h_solution->s[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs]) + i*h_instance->mAgents] -= h_instance->resourcesAgent[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads]*h_instance->mAgents + ((int)h_solution->s[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs])];
					h_solution->s[h_ejection->pos[(j-1) + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs] = h_solution->s[h_ejection->pos[j + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs];
				}
				h_solution->s[h_ejection->pos[(h_ejection->sizeChain[lowestDelta + i*nThreads]-1) + lowestDelta*maxChain + i*maxChain*nThreads] + i*h_instance->nJobs] = ((char)aux);
			}
			//update current runTime
			gettimeofday(&fim, NULL);
			currentTime = (float) (1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000);
		
			if(h_solution->costFinal[i]<bestCost){
				bestCost = h_solution->costFinal[i];
				timeWOutImp = 0;
				gettimeofday(&t_inicio,NULL);
				#ifdef PRINTALL
					printf(BOLDGREEN "IMPROMENT SOLUTION: " RESET);
					printf("\tcost %d K.B.S. %d\n", bestCost, knowBestSolution);
					if(bestCost<knowBestSolution){
						printf(BOLDBLUE "IMPROMENT KNOW BEST SOLUTION!\n" RESET);
					}
				#endif
			}else{
				//update time without improviment
				gettimeofday(&t_fim, NULL);
				timeWOutImp = (float) (1000 * (t_fim.tv_sec - t_inicio.tv_sec) + (t_fim.tv_usec - t_inicio.tv_usec) / 1000);
			}
			//update bestSolution
			if(h_solution->costFinal[i] < best_solution->costFinal[i]){
				best_solution->costFinal[i] = h_solution->costFinal[i];
				for(j=0;j<h_instance->nJobs;j++){
					best_solution->s[j + i*h_instance->nJobs] = h_solution->s[j + i*h_instance->nJobs]; 
				}
				for(j=0;j<h_instance->mAgents;j++){
					best_solution->resUsage[j + i*h_instance->mAgents] = h_solution->resUsage[j + i*h_instance->mAgents]; 
				}
			}

		}

		for(i=0;i<nBlocks;i++){
			for(j=0;j<h_instance->nJobs;j++){
				h_long_list[j + h_solution->s[j+i*h_instance->nJobs]*h_instance->nJobs]++;
			}
		}


		/*for(i=0;i<nBlocks;i++){ //aqui
			for(j=0;j<h_instance->nJobs;j++){
				h_long_list[j + h_solution->s[j+i*h_instance->nJobs]*h_instance->nJobs]++;
				if(h_solution->s[j + i*h_instance->nJobs]>4){
					printf("cpu teste: %d\n",h_solution->s[j + i*h_instance->nJobs]);
				}
			}		
		}*/ //aqui
	
		gettimeofday(&fim, NULL);
		gettimeofday(&t_fim, NULL); 
		currentTime = (float) (1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000);
		timeWOutImp = (float) (1000 * (t_fim.tv_sec - t_inicio.tv_sec) + (t_fim.tv_usec - t_inicio.tv_usec) / 1000);
		#ifdef PRINTALL
			printf("//----------------------------------------------------------------//\n");
			printf("current iteration: %d iteration limit: %d\n", ite, iterationLimit);
			printf("current time: %.0fms time limit: %.0fms\n", currentTime,runtimeLimit);
			printf("current time without improvement: %.0fms time limit for found improvement: %.0f(ms)\n", timeWOutImp,timeLimitImprov);
			printf("//----------------------------------------------------------------//\n");
		#endif	
		if((ite!=iterationLimit)&&(currentTime<runtimeLimit)&&(timeWOutImp<timeLimitImprov)){
			//reallocation pointers of Instance
			h_instance->cost = (Tcost*)(d_instance+1);
			h_instance->resourcesAgent =(TresourcesAgent*) (h_instance->cost +(h_instance->nJobs*h_instance->mAgents));
			h_instance->capacity =(Tcapacity*) (h_instance->resourcesAgent + (h_instance->nJobs*h_instance->mAgents));
			gpuMemcpy(d_instance, h_instance,size_instance, cudaMemcpyHostToDevice);

			//reallocation pointers of Solution
			h_solution->costFinal = (TcostFinal*)(d_solution+1);
			h_solution->s = (Ts*)(h_solution->costFinal + nBlocks);
			h_solution->resUsage = (TresUsage*)(h_solution->s + (h_instance->nJobs*nBlocks));
			gpuMemcpy(d_solution, h_solution, size_solution, cudaMemcpyHostToDevice);

			//reallocation pointers of Ejection
			memset(h_ejection,0,size_ejection);
			h_ejection->pos=(Tpos*)(d_ejection + 1);
			h_ejection->op = (Top*)(h_ejection->pos+ (nBlocks*nThreads*maxChain));
			h_ejection->sizeChain = (TSizeChain*)(h_ejection->op + (nBlocks*nThreads));
			h_ejection->delta = (Tdelta*)(h_ejection->sizeChain + (nBlocks*nThreads));
			gpuMemcpy(d_ejection, h_ejection, size_ejection, cudaMemcpyHostToDevice);

			gettimeofday(&time_rand,NULL);
			srand(time_rand.tv_usec);
			for(i=0;i<(nThreads*nBlocks);i++){
				h_seed[i] = rand()%100000;
			}
			gpuMemcpy(d_seed, h_seed, sizeof(unsigned int)*(nThreads*nBlocks), cudaMemcpyHostToDevice);
			gpuMemcpy(d_short_list, h_short_list,sizeof(int)*(nBlocks*h_instance->nJobs), cudaMemcpyHostToDevice);

		}
		// if((ite_b==0)&&(bestCost<= 1.01 * knowBestSolution)){ //verify really objective???
		// 	ite_b = ite;
		// }
		ite++;
		//		gettimeofday(&fim, NULL);
		//		currentTime = (float) (1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000);
	}
	free(h_long_list);
	#ifdef PRINTALL
		printf("cost: %d\n ite: %d\n time: %.2f ms\n", bestCost, ite, currentTime);
	#endif
	int k;	
	int *cont_similarity = (int*)malloc(sizeof(int)*(h_instance->nJobs*nBlocks));
	int *total_similarity = (int*)malloc(sizeof(int)*nBlocks);
	for(k=0;k<h_instance->nJobs*nBlocks;k++){
		if(k<nBlocks){
			total_similarity[k]=0;
		}
		cont_similarity[k]=0;
	}
	
	//h_solution->costFinal[0]=bestCost;


	int* cont_freq = (int*)malloc(sizeof(int)*h_instance->nJobs*h_instance->mAgents);
	for(i=0;i<h_instance->nJobs;i++){
            for(j=0;j<h_instance->mAgents;j++){
                cont_freq[i+j*h_instance->nJobs]=0;
            }
    }


	for(i=0;i<nBlocks;i++){ 
		for(j=i+1;j<nBlocks;j++){
			for(k=0;k<h_instance->nJobs;k++){
				if(best_solution->s[k + i*h_instance->nJobs] == best_solution->s[k + j*h_instance->nJobs]){
					total_similarity[i]++;
					total_similarity[j]++;
					cont_similarity[k + i*h_instance->nJobs]++;
					cont_similarity[k + j*h_instance->nJobs]++;
				}
			}
		}
	}
	#ifdef PRINTALL
		for(i=0;i<nBlocks;i++){
			printf("cost solution of the block %d: %d - BFS(%d)\n",i, h_solution->costFinal[i],best_solution->costFinal[i]);//best found solution
			printf("total similarity :%d\n",total_similarity[i]);
		}
	#endif

	aux = 0;
	k = best_solution->costFinal[0];
	for(i=1;i<nBlocks;i++){ //position bestSolution
		if(best_solution->costFinal[i]< k){
			aux = i;
			k = best_solution->costFinal[i];
		}
	}
	for(i=0;i<nBlocks;i++){
		for(j=0;j<h_instance->nJobs;j++){
		    cont_freq[j+best_solution->s[j+i*h_instance->nJobs]*h_instance->nJobs]++;
		}
	}
	gettimeofday(&fim, NULL);
	currentTime = (float) (1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000);
	printf("time: %.2f ms (%d) - solution with most similarity is %d with %d, cost: %d\n",currentTime,ite, aux,total_similarity[aux],best_solution->costFinal[aux]);
	createOutputFileSolution(best_solution,h_instance,aux,temp_teste); //change function name 
	createOutputFileFrequencyVersion1(best_solution,h_instance,cont_similarity,aux,temp_teste);//change function name 
	createOutputFileFrequencyVersion2(best_solution,h_instance,cont_freq,aux,temp_teste);//change function name 
	
	cudaFree(states);
	cudaFree(d_instance);
	cudaFree(d_solution);
	cudaFree(d_ejection);
	cudaFree(d_seed);
	cudaFree(d_short_list);

	free(cont_freq);
	free(h_short_list);
	free(h_seed);
	free(h_instance);
	free(h_solution);
	free(h_ejection);
	free(best_solution);
	// freePointersInstance(h_instance);
	// freePointersSolution(h_solution);
	// freePointerEjectionChain(h_ejection);
	// freePointersSolution(best_solution);
	free(cont_similarity);
	free(total_similarity);
	// cudaFree(states);
	// cudaFree(d_instance);
	// cudaFree(d_solution);
	// cudaFree(d_ejection);
	// cudaFree(d_seed);
	// cudaFree(d_short_list);
	return 0;
}


