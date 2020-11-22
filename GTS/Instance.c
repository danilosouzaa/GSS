#include "Instance.h"

int iReturn(int i, int j, int n, int m ){
	return i*m+j;
}

Instance* allocationPointersInstance(int n, int m){
	size_t size_instance = sizeof(Instance)
							+ sizeof(Tcost)*(n*m)  //cost
							+ sizeof(TresourcesAgent)*(m*n)
							+ sizeof(Tcapacity)*m;
	Instance *inst =(Instance*)malloc(size_instance);
	assert(inst!=NULL);
	memset(inst,0,size_instance);
	inst->cost = (Tcost*)(inst+1);
	inst->resourcesAgent =(TresourcesAgent*) (inst->cost +(n*m));
	inst->capacity =(Tcapacity*) (inst->resourcesAgent + (n*m));
	inst->nJobs = n;
	inst->mAgents = m;
	return inst;
}


Instance* loadInstance(const char *fileName){
	    FILE *arq;
	    int aux_2;
	    char ch;
	    int m,n, cont=0, i,j,a;
	    short int aux;
	    Instance *inst;
	    arq=fopen(fileName, "r");
	    if(arq==NULL)
	    {
	        printf("Erro, couldn't open the file.\n");
	    }
	    else
	    {
	        a = fscanf(arq, "%d" , &m);
	        a = fscanf(arq, "%d" , &n);
	        printf("Parameters load\n");
	        inst = allocationPointersInstance(n,m);
	        printf("Pointers Allocated!\n");
	        for(i=0; i<m; i++)
	        {
	            for(j=0; j<n; j++)
	            {
	                a = fscanf(arq,"%d", &aux_2);
	                inst->cost[iReturn(j,i,n,m)]=aux_2;
	            }
	        }



	        for(i=0; i<m; i++)
	        {
	            for(j=0; j<n; j++)
	            {
	                a = fscanf(arq,"%hi", &aux);
	                inst->resourcesAgent[iReturn(j,i,n,m)]=aux;
	            }
	        }
	        for(j=0; j<m; j++)
	        {
	            a = fscanf(arq,"%hi", &aux);
	            inst->capacity[j]=aux;
	        }

	    }
	    fclose(arq);
	    return inst;
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





