#include <iostream>

extern "C" {
#include "lp.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
}
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

typedef struct
{
    int nJobs;
    int mAgents;
    double *cost;
    double *resourceAgent;
    double *capacity;
} Instance;


int iReturn(int i, int j, int n, int m )
{
    return i*m+j;
}

Instance* allocationPointer(int nJobs,int mAgents)
{
    Instance *inst;
    size_t size_instance = sizeof(Instance)
                           + sizeof(double)*(nJobs*mAgents)
                           + sizeof(double)*(nJobs*mAgents)
                           + sizeof(double)*(mAgents);
    inst = (Instance*)malloc(size_instance);
    assert(inst!=NULL);
    memset(inst,0,size_instance);
    inst->cost = (double*)(inst+1);
    inst->resourceAgent = (double*)(inst->cost + (nJobs*mAgents));
    inst->capacity = (double*) (inst->resourceAgent + (nJobs*mAgents));
    return inst;
}

Instance* loadInstance(const char *fileName)
{
    FILE *arq;
    int aux_2;
    char ch;
    int m,n, cont=0, i,j,a;
    int aux;
    Instance *inst;
	
    char nf[30]="";
    strcat(nf,"../Instances/");
    strcat(nf,fileName);
    arq=fopen(nf, "r");
    if(arq==NULL)
    {
        printf("Erro, couldn't open the file: %s.\n",fileName);
    }
    else
    {
        a = fscanf(arq, "%d" , &m);
        a = fscanf(arq, "%d" , &n);
        inst = allocationPointer(n,m);
        inst->nJobs = n;
        inst->mAgents = m;
        for(i=0; i<m; i++)
        {
            for(j=0; j<n; j++)
            {
                a = fscanf(arq,"%d", &aux_2);
                inst->cost[iReturn(j,i,n,m)]= (double) aux_2;
            }
        }



        for(i=0; i<m; i++)
        {
            for(j=0; j<n; j++)
            {
                a = fscanf(arq,"%d", &aux);
                inst->resourceAgent[iReturn(j,i,n,m)]=(double) aux;
            }

        }


        for(j=0; j<m; j++)
        {
            a = fscanf(arq,"%d", &aux);
            inst->capacity[j]=(double) aux;

        }

    }
    fclose(arq);
    return inst;
}

void load_residence_sol(const char *fileName, int *residence ,int *sol, Instance *inst, int type)
{
    FILE *f;
    char nf[30]="";
    int i,j,aux;
    //char n[50];
    if(type == 0)
    {
        strcat(nf,"../Residence/Freq_");
    }
    else
    {
        strcat(nf,"../Residence/temp_");
    }
    strcat(nf,fileName);
    strcat(nf,".txt");
    f = fopen(nf, "r");
    if(f==NULL)
    {
        printf("Erro, couldn't open the file: %s.\n", nf);
    }
    else
    {
        while((fscanf(f,"x(%d,%d) = %d\n",&i,&j,&aux))!=EOF)
        {
            residence[i-1] = aux;
            sol[i-1] = j-1;
        }

    }
    fclose(f);
}


void load_m_reference_sol(const char *fileName, int *m_reference, Instance *inst)
{
    FILE *f;
    char nf[30]="";
    int i,j,aux;

    strcat(nf,"../Residence/Freq2_");
    strcat(nf,fileName);
    strcat(nf,".txt");
    f = fopen(nf, "r");
    if(f==NULL)
    {
        printf("Erro, couldn't open the file: %s .\n",nf);

    }
    else
    {

        while((fscanf(f,"x(%d,%d) = %d\n",&i,&j,&aux))!=EOF)
        {
            m_reference[(i-1) + (j-1)*inst->nJobs] = aux;
        }
    }
    fclose(f);
}


LinearProgramPtr geraLP(const char *fileName, Instance *inst)
{
    LinearProgramPtr lp = lp_create();
    int number_variables = inst->nJobs*inst->mAgents;
    double *c_variables;
    double *lb;
    double *ub;
    char *integer;
    char **name;
    int *indexes;
    double *cof;
    double *cof_temp;
    double *n_right;
    int *index_temp;
    int *index_temp2;
    char nome[15]="x";
    char n[15];
    strcpy(n,fileName);
    char aux[4];
    int i,j;
    //int *v_allocation;

    int position;

    c_variables = (double*)malloc(sizeof(double)*number_variables);
    lb = (double*)malloc(sizeof(double)*number_variables);
    ub = (double*)malloc(sizeof(double)*number_variables);
    integer = (char*)malloc(sizeof(char)*number_variables);
    index_temp = (int*)malloc(sizeof(int)*(inst->mAgents));
    index_temp2 = (int*)malloc(sizeof(int)*inst->nJobs);

    name = (char**)malloc(sizeof(char*)*number_variables);
    for(i=0; i<number_variables; i++)
    {
        name[i]=new char[15];
    }

    lp_set_parallel(lp,1);

    indexes = (int*)malloc(sizeof(int)*number_variables);
    cof = (double*)malloc(sizeof(double)*inst->nJobs);
    cof_temp = (double*)malloc(sizeof(double)*inst->mAgents);
    n_right = (double*)malloc(sizeof(double)*(inst->mAgents));
    for(i=0; i<number_variables; i++)
    {
        c_variables[i] = inst->cost[i];
        lb[i]=0;
        ub[i]=1;
        integer[i]=1;
        indexes[i]=i;
    }
    for(i=0; i<inst->mAgents; i++)
    {
        n_right[i] = inst->capacity[i];
    }
    for(i=0; i<inst->nJobs; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            strcpy(nome,"x(");
            sprintf(aux,"%d,",i+1);
            strcat(nome,aux);
            sprintf(aux,"%d)",j+1);
            strcat(nome,aux);
            strcpy(name[iReturn(i,j,inst->nJobs,inst->mAgents)],nome);
        }
    }

    /*Add function objetive*/
    lp_add_cols(lp,number_variables,c_variables,lb,ub,integer,name);
    for(i=0; i<inst->mAgents; i++)
    {
        strcpy(nome,"r");
        sprintf(aux,"%d",i+1);
        strcat(nome,aux);
        for(j=0; j<inst->nJobs; j++)
        {
            cof[j] = inst->resourceAgent[iReturn(j,i,inst->nJobs,inst->mAgents)];
            index_temp2[j] = indexes[iReturn(j,i,inst->nJobs,inst->mAgents)];
        }
        lp_add_row(lp,inst->nJobs,index_temp2,cof,nome,'L',n_right[i]);
    }

    for(i=0; i<inst->nJobs; i++)
    {
        strcpy(nome,"r");
        sprintf(aux,"%d",i+inst->mAgents+1);
        strcat(nome,aux);
        for(j=0; j<inst->mAgents; j++)
        {
            index_temp[j] = indexes[iReturn(i,j,inst->nJobs,inst->mAgents)];
            cof_temp[j]=1;
        }

        lp_add_row(lp,inst->mAgents,index_temp,cof_temp,nome,'E',1);
    }

    for(i=0; i<inst->nJobs; i++)
    {
        delete []name[i];
    }
    free(name);
    free(integer);
    free(lb);
    free(ub);
    free(c_variables);
    free(cof);
    free(cof_temp);
    free(n_right);
    free(indexes);
    free(index_temp);
    free(index_temp2);

    return lp;

}

LinearProgramPtr set_solution_lp(LinearProgramPtr lp, int *sol, Instance *inst)
{
    char **mip_name;
    //int *indexes = (int*)malloc(sizeof(int)*(inst->nJobs*inst->mAgents));
    int i,j;
    char aux[15];
    double *mip_coef = (double*)malloc(sizeof(double)*(inst->nJobs*inst->mAgents));
    mip_name = (char**)malloc(sizeof(char*)*(inst->nJobs*inst->mAgents));

    for(i=0; i<inst->nJobs; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            // indexes[i] = iReturn(i,sol[i],inst->nJobs,inst->mAgents);
            mip_name[iReturn(i,j,inst->nJobs,inst->mAgents)] = new char[15];
            sprintf(aux,"x(%d,%d)",i+1,j+1);
            strcpy(mip_name[iReturn(i,j,inst->nJobs,inst->mAgents)],aux);
            //printf("%s\n",mip_name[i]);
            if(j==sol[i]){
                mip_coef[iReturn(i,j,inst->nJobs,inst->mAgents)] = 1;
            }else{
                mip_coef[iReturn(i,j,inst->nJobs,inst->mAgents)] = 0;
            }

        }
    }
    //getchar();
    lp_load_mip_start(lp,(inst->nJobs*inst->mAgents),mip_name,mip_coef);
    //lp_load_mip_starti(lp,inst->nJobs,indexes,mip_coef);
   // lp_mipstart_debug(lp);
    for(i=0; i<inst->nJobs; i++)
    {
        delete []mip_name[i];
    }
    free(mip_name);
    free(mip_coef);
    return lp;
}

LinearProgramPtr fixed_variables(LinearProgramPtr lp, int *residence,int sizeFixHard, Instance *inst, int *sol)
{
    int *v_ordem =(int*)malloc(sizeof(int)*inst->nJobs);
    int *residence_aux = (int*)malloc(sizeof(int)*inst->nJobs);
    memcpy(residence_aux,residence,sizeof(int)*inst->nJobs);
    int i, j, aux;

    for(i=0; i<inst->nJobs; i++)
    {
        v_ordem[i] = i;
    }
    for( i = 0; i<inst->nJobs; i++ )
    {
        for( j = i + 1; j < inst->nJobs; j++ )
        {
            if ( residence_aux[i] < residence_aux[j])
            {
                aux = residence_aux[i];
                residence_aux[i] = residence_aux[j];
                residence_aux[j] = aux;
                aux=v_ordem[i];
                v_ordem[i]=v_ordem[j];
                v_ordem[j]=aux;
            }
        }
    }

    printf("Jobs Fixadas: \n");
    for(i=0; i<sizeFixHard; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            if(j == sol[v_ordem[i]])
            {
                lp_fix_col(lp,iReturn(v_ordem[i],j,inst->nJobs,inst->mAgents),1);
                printf("%d-",v_ordem[i]+1);
            }
            else
            {
                lp_fix_col(lp,iReturn(v_ordem[i],j,inst->nJobs,inst->mAgents),0);
            }


        }

    }
    printf("\n");

    free(v_ordem);
    free(residence_aux);


    return lp;
}

int return_objective(Instance *inst, int *sol)
{
    int cost_objective = 0;
    for(int i=0; i<inst->nJobs; i++)
    {
        cost_objective += inst->cost[iReturn(i ,sol[i],inst->nJobs,inst->mAgents)];
    }
    printf("Custo: %d\n", cost_objective);
    return cost_objective;
}

LinearProgramPtr include_constrain_canonic(LinearProgramPtr lp, Instance *inst, int value_obj)
{

    char nome[15];
    int i,j;

    double *cof =(double*)malloc(sizeof(double)*(inst->nJobs*inst->mAgents));
    int *index_temp = (int*)malloc(sizeof(int)*(inst->nJobs*inst->mAgents));
    strcpy(nome,"k1");
    for(i=0; i<inst->nJobs; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            cof[iReturn(i,j,inst->nJobs,inst->mAgents)] = inst->cost[iReturn(i,j,inst->nJobs,inst->mAgents)];
            index_temp[iReturn(i,j,inst->nJobs,inst->mAgents)] = iReturn(i,j,inst->nJobs,inst->mAgents);
        }
    }
    lp_add_row(lp,(inst->nJobs*inst->mAgents),index_temp,cof,nome,'L', value_obj);
    free(cof);
    free(index_temp);
    return lp;

}

LinearProgramPtr incluce_constrain_soft(LinearProgramPtr lp, Instance *inst, int *residence, int *sol, int sizeFixSoft)
{
    double *cof = (double*)malloc(sizeof(double)*inst->nJobs);
    double v_max, v_min, aux, soma = 0;
    int *index = (int*)malloc(sizeof(int)*inst->nJobs);
    int i;
    char nome[20] ="k2";
    v_max = v_min = residence[0];
    for(i = 1; i<inst->nJobs; i++)
    {
        if(residence[i]>v_max)
        {
            v_max = residence[i];
        }
        if(residence[i]<v_min)
        {
            v_min = residence[i];
        }
    }
    for(i = 0; i<inst->nJobs; i++)
    {
        aux = residence[i];
        cof[i] = (aux - v_min)/(v_max - v_min);
        soma += cof[i];
        index[i] = iReturn(i, sol[i],inst->nJobs,inst->mAgents);
        // printf("cof: %f, res: %d \n", cof[i],residence[i]);
    }
    soma -= sizeFixSoft;
    lp_add_row(lp,inst->nJobs,index,cof,nome,'G', soma);

    free(cof);
    free(index);
    return lp;
}


void update_res_ref(LinearProgramPtr lp, int *residence, int *m_reference, int *sol,Instance *inst,const char *n)
{
    //char **mip_name;
    //double *mip_coef;
    int i;
    int a1, a2;
    //Read File
    FILE *f = fopen(n,"r");
    char line[256];
    int nLine = 0;
    int nRCols = 0;
    char col[4][256];
    //mip_coef = (double*)malloc(sizeof(double)*inst->nJobs);
    //mip_name = (char**)malloc(sizeof(char*)*inst->nJobs);
    printf("nJobs %d \n",inst->nJobs);

    //for(i=0; i<inst->nJobs; i++)
    //{
    //    mip_name[i]=new char[10];
    //}

    //lp_read_mip_start(n,mip_name,mip_coef);

    while (fgets( line, 256, f ))
    {
        ++nLine;

        int nread = sscanf( line, "%s %s %s %s", col[0], col[1], col[2], col[3] );

        if (!nread)
            continue;

        if (strlen(col[0])&&isdigit(col[0][0])&&(nread>=3))
        {
            sscanf(col[1],"x(%d,%d)",&a1,&a2);
            //printf("x(%d,%d)\n", a1,a2);
            sol[a1-1] = a2-1;
            m_reference[iReturn(a1-1,a2-2,inst->nJobs,inst->mAgents)]++;
            residence[a1-1] = m_reference[iReturn(a1-1,a2-1,inst->nJobs,inst->mAgents)];
            ++nRCols;
        }    //freq[(a_ux-1)+(j-1)*inst->nJobs]++;
        //fprintf(f,"x(%d,%d) = %d \n",a_ux, j , freq[(a_ux-1) + (j-1)*inst->nJobs]);


    }
    fclose(f);
    //printf("Numero: %d\n", nRCols);
    //getchar();
    //for(i=0; i<inst->nJobs; i++)
    //{
    //    delete []mip_name[i];
    //}
    //free(mip_name);
    //free(mip_coef);
}

LinearProgramPtr free_variables(LinearProgramPtr lp, Instance *inst)
{

    for(int i=0; i<inst->nJobs; i++)
    {
        for(int j=0; j<inst->mAgents; j++)
        {
            lp_set_col_bounds(lp,iReturn(i,j,inst->nJobs,inst->mAgents),0,1);
        }
    }
    return lp;


}
void runSolver(const char *fileName, /*int sizeFixHard, */int sizeFixSoft, float time)
{
    int i,j, v_d = sizeFixSoft;
    int fo_antes, fo_depois;
    //int quantidade = sizeFixHard;
    char n[20];
    char n2[20];
    Instance* inst;
    inst = loadInstance(fileName);
    LinearProgramPtr lp = geraLP(fileName,inst);
    int *sol = (int*)malloc(sizeof(int)*inst->nJobs); //save solution
    int *residence = (int*)malloc(sizeof(int)*inst->nJobs);
    int *m_reference = (int*)malloc(sizeof(int)*(inst->nJobs*inst->mAgents));
    load_residence_sol(fileName,residence, sol,inst,0); // Load residence and sol
    load_m_reference_sol(fileName,m_reference,inst);//Load reference
    lp = set_solution_lp(lp,sol,inst);//Set MIP start
    fo_antes = return_objective(inst,sol);//Load Fo Initial
   //Set Parameter Time
    //lp_set_max_solutions(lp,5);//Set Parameter Solution
    strcpy(n,fileName);
    strcat(n,".lp");
    printf("%s\n",n);
    strcpy(n2,"../Solution/sol_");
    strcat(n2,fileName);
    strcat(n2,".txt");
    int res_retirada = inst->nJobs+inst->mAgents;

    int erro = 0;
	struct timeval inicio;
	struct timeval fim;
    float tmili = 0;
    gettimeofday(&inicio,NULL);
    while(/*(quantidade<=inst->nJobs)&&*/tmili<time*60000)
    {
        //lp = fixed_variables(lp,residence,inst->nJobs - quantidade,inst,sol);//Fixed variable
        printf("fo: %d\n",fo_antes);
        lp = include_constrain_canonic(lp, inst, fo_antes);
        lp = incluce_constrain_soft(lp,inst,residence,sol,sizeFixSoft);
        printf("1 -%s\n",n);
        printf("Time %f\n",tmili);
        lp_write_lp(lp,n);
        //printf("teste = %d\n",i);
        setNodeInfinity(lp);
        lp_set_max_seconds(lp,60);
        printf("time: %f\n",time*60-(int)(tmili/1000));
        getcallback(lp,time*60-(int)(tmili/1000));
        i=lp_optimize(lp);
        //lp_mipstart_debug(lp);
        fo_depois = lp_obj_value(lp);
        if(fo_depois<fo_antes)
        {
            printf("%s\n",n2);
            lp_write_sol(lp,n2);
            sizeFixSoft = v_d;
            //quantidade = sizeFixHard;
            update_res_ref(lp,residence,m_reference,sol,inst, n2);
        }
        else
        {
            sizeFixSoft +=v_d;
		
            //quantidade += sizeFixHard;
        }

        lp_remove_row(lp,res_retirada);
        lp_remove_row(lp,res_retirada+1);
        lp_write_lp(lp,n);
        //free_variables(lp,inst);
        lp = set_solution_lp(lp,sol,inst);
        //lp_write_lp(lp,n);

        //lp_mipstart_debug(lp);
        fo_antes = fo_depois;
         gettimeofday(&fim, NULL);
         tmili = 1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000;
         if(sizeFixSoft >inst->nJobs){
            break;
         }

    }

    gettimeofday(&fim, NULL);
    tmili = 1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000;
    printf("Final Time: %f\n",tmili);

    lp_free(&lp);
    remove(n);

    free(inst);
    free(sol);
    free(residence);
    free(m_reference);
}



int main(int argc,char *argv[])
{
    const char *fileName = argv[1];
    //int sizeFixHard = atoi(argv[2]);
    int sizeFixSoft = atoi(argv[2]);
    float time = atof(argv[3]);
    runSolver(fileName, /*sizeFixHard,*/ sizeFixSoft,time);
    printf("Final\n");
}

