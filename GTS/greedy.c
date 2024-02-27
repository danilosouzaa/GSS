#include  "greedy.h"

 



//Create vector with priority of allocation jobs in Agents 
int* inicializeVector(Instance *inst, float p1, float p2)
{
    int *vOrdem = (int*)malloc(sizeof(int)*(inst->nJobs *inst->mAgents));
    float *vParametro = (float*)malloc(sizeof(float)*(inst->nJobs *inst->mAgents));
    int i, j;
    int aux1,aux2,iAux1,iAux2;
    for(i=0; i<inst->nJobs ; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            vParametro[iReturn(i,j,inst->nJobs ,inst->mAgents)] = p1*inst->cost[iReturn(i,j,inst->nJobs ,inst->mAgents)] + p2*inst->resourcesAgent[iReturn(i,j,inst->nJobs ,inst->mAgents)];
            vOrdem[iReturn(i,j,inst->nJobs ,inst->mAgents)]=j;
        }
    }
    for(i=0; i<inst->nJobs ; i++)
    {
        for(j= inst->mAgents-1; j>=0; j--)
        {
            for(aux1= j-1; aux1>=0; aux1--)
            {
                if(vParametro[iReturn(i,j,inst->nJobs ,inst->mAgents)]<vParametro[iReturn(i,aux1,inst->nJobs ,inst->mAgents)])
                {
                    aux2 = vParametro[iReturn(i,j,inst->nJobs ,inst->mAgents)];
                    iAux2 = vOrdem[iReturn(i,j,inst->nJobs ,inst->mAgents)];
                    vParametro[iReturn(i,j,inst->nJobs ,inst->mAgents)]= vParametro[iReturn(i,aux1,inst->nJobs ,inst->mAgents)];
                    vOrdem[iReturn(i,j,inst->nJobs ,inst->mAgents)]=vOrdem[iReturn(i,aux1,inst->nJobs ,inst->mAgents)];;
                    vParametro[iReturn(i,aux1,inst->nJobs ,inst->mAgents)]=aux2;
                    vOrdem[iReturn(i,aux1,inst->nJobs ,inst->mAgents)]=iAux2;
                }
            }
        }
    }
    free(vParametro);
    return vOrdem;
}

int* inicializeVector2(Instance *inst, int job, float p1, float p2)
{ 
    int *vOrdem = (int*)malloc(sizeof(int)*inst->mAgents);
    float *vParametro = (float*)malloc(sizeof(float)*inst->mAgents);
    int j;
    for(j=0; j<inst->mAgents; j++){        
        vParametro[j] = (-1)*(p1*inst->cost[iReturn(job,j,inst->nJobs ,inst->mAgents)] + p2*inst->resourcesAgent[iReturn(job,j,inst->nJobs ,inst->mAgents)]);
        vOrdem[j]=j;
    }
    quicksortCof(vParametro,vOrdem,0,inst->mAgents);
    free(vParametro);
    return vOrdem;

}

int quicksortCof(float *values, int *idc, int began, int end)
{
    int i, j;
    float pivo, aux;
    i = began;
    j = end - 1;
    pivo = values[(began + end) / 2];
    while (i <= j)
    {
        while (values[i] > pivo && i < end)
        {
            i++;
        }
        while (values[j] < pivo && j > began)
        {
            j--;
        }
        if (i <= j)
        {
            aux = values[i];
            values[i] = values[j];
            values[j] = aux;

            aux = idc[i];
            idc[i] = idc[j];
            idc[j] = aux;

            i++;
            j--;
        }
    }
    if (j > began)
        quicksortCof(values, idc, began, j + 1);
    if (i < end)
        quicksortCof(values, idc, i, end);
    return 1;
}


int greedy(Instance *inst, Solution *sol, float p1, float p2, int block)
{
    int *vOrdem;
    int *allocated=(int*)malloc(sizeof(int)*inst->nJobs );
    int i,j;
    unsigned short int agent;
    int cont=0;
    memset(allocated,0,sizeof(int)*inst->nJobs );
    sol->costFinal[block]=0;
    for(i=0; i<inst->nJobs ; i++)
    {   
        vOrdem = inicializeVector2(inst,i,p1,p2);
        for(j=0; j<inst->mAgents; j++)
        {
            agent = vOrdem[j];
            if((allocated[i]==0)&&(inst->resourcesAgent[iReturn(i,agent,inst->nJobs ,inst->mAgents)]+sol->resUsage[agent + block*inst->mAgents]<=inst->capacity[agent]))
            {
                allocated[i]=1;
                sol->s[i + block*inst->nJobs] = agent;
                sol->costFinal[block]+=inst->cost[iReturn(i,agent,inst->nJobs ,inst->mAgents)];
                sol->resUsage[agent + block*inst->mAgents]+=inst->resourcesAgent[iReturn(i,agent,inst->nJobs ,inst->mAgents)];
                cont++;
                break;
            }
        }
        free(vOrdem);
    }
    free(allocated);
    if(cont!=inst->nJobs ){
        sol->costFinal[block]=0;
        return 0;
    }
    return 1;
}

