#include "Instance.h"

int iReturn(int i, int j, int n, int m ){
	return i*m+j;
}

Instance* allocationPointersInstance(int nJobs, int mAgents){
	size_t size_instance = sizeof(Instance)
							+ sizeof(Tcost)*(nJobs*mAgents)  //cost - d_ji
							+ sizeof(TresourcesAgent)*(nJobs*mAgents) // resource consumed of agent - r_ji
							+ sizeof(Tcapacity)*mAgents; // capacity agents b_i
	Instance *inst =(Instance*)malloc(size_instance);
	assert(inst!=NULL);
	memset(inst,0,size_instance);
	inst->cost = (Tcost*)(inst+1);
	inst->resourcesAgent =(TresourcesAgent*) (inst->cost +(nJobs*mAgents));
	inst->capacity =(Tcapacity*) (inst->resourcesAgent + (nJobs*mAgents));
	inst->nJobs = nJobs;
	inst->mAgents = mAgents;
	return inst;
}

void freePointersInstance(Instance *inst){
	free(inst->cost);
	free(inst->resourcesAgent);
	free(inst->capacity);
	free(inst);
}


Instance* loadInstance(const char *fileName){
	    FILE *arq;
	    int aux_2;
	    int m,n, i,j;
	    short int aux;
	    Instance *inst;
	    arq=fopen(fileName, "r");
	    if(arq==NULL)
	    {
	        printf("Erro, couldn't open the file.\n");
			return NULL;
	    }
	    else
	    {
			#ifdef PRINTALL
				printf("Instance %s opened successfully!\n",fileName);
			#endif
	        fscanf(arq, "%d" , &m);
	        fscanf(arq, "%d" , &n);
			#ifdef PRINTALL
		        printf("Parameters load\n");
			#endif
	        inst = allocationPointersInstance(n,m);
			#ifdef PRINTALL
		        printf("Pointers Allocated!\n");
			#endif
	        for(i=0; i<m; i++)
	        {
	            for(j=0; j<n; j++)
	            {
	                fscanf(arq,"%d", &aux_2);
	                inst->cost[iReturn(j,i,n,m)]=aux_2;
	            }
	        }
	        for(i=0; i<m; i++)
	        {
	            for(j=0; j<n; j++)
	            {
	                fscanf(arq,"%hi", &aux);
	                inst->resourcesAgent[iReturn(j,i,n,m)]=aux;
	            }
	        }
	        for(j=0; j<m; j++)
	        {
	            fscanf(arq,"%hi", &aux);
	            inst->capacity[j]=aux;
	        }
			fclose(arq);
	    	return inst;
	    }
	    
}

void showInstance(Instance *inst){
    int i, j;
    printf("Data of problem GAP\n");
    printf("Number of jobs: %d \n",inst->nJobs);
    printf("Number of agent: %d \n",inst->mAgents);
    for(j=0; j<inst->mAgents; j++)
    {
       for(i=0; i< inst->nJobs; i++)
        {
            printf("Cost of job %d allocated the agent %d: %d \n",i+1,j+1,inst->cost[iReturn(i,j,inst->nJobs,inst->mAgents)]);
        }
    }
    for(j=0; j<inst->mAgents; j++)
    {
        for(i=0; i< inst->nJobs; i++)
        {
            printf("Amount of resource of job %d consumes by the agent %d: %hi \n", i+1 ,j+1,inst->resourcesAgent[iReturn(i,j,inst->nJobs,inst->mAgents)]);

        }
    }
    for(j=0; j<inst->mAgents; j++)
    {
        printf("Capacity of agent %d: %hi\n",j+1,inst->capacity[j]);
    }
}





