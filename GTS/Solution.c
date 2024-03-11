#include "Solution.h"

//const int nBlocks =5;
//const int nThreads = 896;
//const int maxChain = 10;

Solution* allocationPointersSolution(Instance *inst, int nBlocks){
	size_t size_solution =  sizeof(Solution)
							+ sizeof(TcostFinal)*nBlocks
							+ sizeof(Ts)*(inst->nJobs*nBlocks)
							+ sizeof(TresUsage)*(inst->mAgents*nBlocks);
	Solution *sol;
	sol = (Solution*)malloc(size_solution);
	assert(sol!=NULL);
	memset(sol,0,size_solution);
	sol->costFinal = (TcostFinal*)(sol+1);
	sol->s = (Ts*)(sol->costFinal + nBlocks);
	sol->resUsage = (TresUsage*)(sol->s + (inst->nJobs*nBlocks));
	return sol;
}
void freePointersSolution(Solution *sol){
	free(sol->costFinal);
	free(sol->s);
	free(sol->resUsage);
	free(sol);
}


EjectionChain* allocationPointerEjectionChain(Instance *inst, int nBlocks, int nThreads){
	size_t size_ejection = sizeof(EjectionChain)
							+ sizeof(Tpos)*(nBlocks*nThreads*maxChain)
							+ sizeof(Top)*(nBlocks*nThreads)
							+ sizeof(TSizeChain)*(nBlocks*nThreads)
							+ sizeof(Tdelta)*(nBlocks*nThreads);
	EjectionChain *ejection;
	ejection =(EjectionChain*)malloc(size_ejection);
	assert(ejection!=NULL);
	ejection->pos=(Tpos*)(ejection + 1);
	ejection->op = (Top*)(ejection->pos+ (nBlocks*nThreads*maxChain));
	ejection->sizeChain = (TSizeChain*)(ejection->op + (nBlocks*nThreads));
	ejection->delta = (Tdelta*)(ejection->sizeChain + (nBlocks*nThreads));
	return ejection;
}

void freePointerEjectionChain(EjectionChain *ejection){
	free(ejection->pos);
	free(ejection->op);
	free(ejection->sizeChain);
	free(ejection->delta);
	free(ejection);
}

void createOutputFileSolution(Solution *sol, Instance *inst,int pos_best, const char *fileName){
		char nf[50]="";
		strcat(nf,"../MIP/MIP_");
		strcat(nf,fileName);
		strcat(nf,".txt");
		int i;
		FILE *f = fopen (nf,"w");
		if(f==NULL){
			printf("erro: verify permission in folder: ../MIP/ \n \n ");
		}else{
			for(i=0;i<inst->nJobs;i++){
				fprintf(f,"x(%d,%d)\n",i+1,sol->s[i + inst->nJobs*pos_best]+1);
			}
			#ifdef PRINTALL
				printf("create solution output file sucessed : %s!\n",nf);
			#endif
			fclose(f);
		}
		
}

void createOutputFileFrequencyVersion1(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName){
		FILE *f;
		char nf[50]="";
		strcat(nf,"../Residence/Freq_");
		strcat(nf,fileName);
		strcat(nf,".txt");
		int i;
		f = fopen (nf,"w");
		if(f==NULL){
			printf("erro: verify permission in folder: ../Residence/ \n");
		}else{
			for(i=0;i<inst->nJobs;i++){
				fprintf(f,"x(%d,%d) = %d \n",i+1, sol->s[i + pos_best*inst->nJobs]+1 , cont_similarity[i + pos_best*inst->nJobs]);
			}
			#ifdef PRINTALL
				printf("create frequency (v.1) output file sucessed : %s!\n",nf);		
			#endif
			fclose(f);
		}
		
		
}

void createOutputFileFrequencyVersion2(Solution *sol, Instance *inst,int *cont_similarity,int pos_best, const char *fileName){
    FILE *f;
    char nf[50]="";
    strcat(nf,"../Residence/Freq2_");
    strcat(nf,fileName);
    strcat(nf,".txt");
    int i,j;
    f = fopen (nf,"w");
    if(f==NULL){
        printf("erro: verify permission in folder: ../Residence/ \n");
    }else{
		for(i=0;i<inst->nJobs;i++){
			for(j=0;j<inst->mAgents;j++){
            	fprintf(f,"x(%d,%d) = %d \n",i+1,j+1 , cont_similarity[i + j*inst->nJobs]);
            }
		}
		#ifdef PRINTALL
			printf("create frequency (v.2) output file sucessed : %s!\n",nf);		
		#endif
		fclose(f);
    }
    
}


int returnIndice(Solution *h_solution,EjectionChain *h_ejection, int block, int menor, int *h_long_list, int nJobs,int mAgents, int nThreads){
	int qnt_menor=0;
	int *v_pos_menor= (int*)malloc(sizeof(int)*nThreads);
	int *mod_cont = (int*)malloc(sizeof(int)*nThreads);
	int aux1 , aux2;
	int pos,i,j;
	for(i=0;i<nThreads;i++){
		v_pos_menor[i] = 0;	
	}
	for(i=0;i<nThreads;i++){
		mod_cont[i] = 0;
		if(menor == h_ejection->delta[i + block*nThreads]){
			qnt_menor++;
			v_pos_menor[qnt_menor-1] = i;
		}
	}

	if(qnt_menor==1){
		pos = v_pos_menor[0];
		free(mod_cont);
		free(v_pos_menor);
		return pos;
	}else{

		for(i=0;i<qnt_menor;i++){
			if(h_ejection->op[v_pos_menor[i] + block*nThreads]==1){
				mod_cont[i]+= h_long_list[h_ejection->pos[0 + v_pos_menor[i]*maxChain + block *maxChain*nThreads] + h_ejection->pos[1 + v_pos_menor[i]*maxChain + block *maxChain*nThreads]*nJobs];
				//printf("Shift with indice %d\n", mod_cont[i]);
			}else{
				//printf("ejection of size: %d \n", h_ejection->sizeChain[ v_pos_menor[i] + block*nThreads]);
				for(j=1;j<h_ejection->sizeChain[v_pos_menor[i] + block*nThreads];j++){
					aux1 = h_ejection->pos[j + v_pos_menor[i]*maxChain + block *maxChain*nThreads];
					aux2 = h_ejection->pos[(j -1) + v_pos_menor[i]*maxChain + block *maxChain*nThreads];
					mod_cont[i]+= h_long_list[aux2 + ((int)h_solution->s[aux1 + block*nJobs])*nJobs];
				}
				aux1 = h_ejection->pos[0 + v_pos_menor[i]*maxChain + block *maxChain*nThreads];
				aux2 = h_ejection->pos[(j-1) + v_pos_menor[i]*maxChain + block *maxChain*nThreads];
				mod_cont[i]+= h_long_list[aux2 + ((int)h_solution->s[aux1 + block*nJobs])*nJobs];
				//printf("ejection chain with indice: %d\n",mod_cont[i]);
			}
		}
		aux1 = mod_cont[0];
		pos = v_pos_menor[0];
		for(i=1;i<qnt_menor;i++){
			if(mod_cont[i]<aux1){
				aux1 = mod_cont[i];
				pos = v_pos_menor[i];
			}
		}
		free(mod_cont);
		free(v_pos_menor);
		return pos;
	}

}

