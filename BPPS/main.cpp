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

#define TIMERERUN 10
#define SZMAXPATH 40

#define PRINTALL 0
#define PRINTDEFAULT 1
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

void freeInstance(Instance* inst){
    free(inst->cost);
    free(inst->resourceAgent);
    free(inst->capacity);
    free(inst);

}

Instance* loadInstance(const char *fileName)
{
    FILE *arq;
    int aux_2;
    char ch;
    int m,n, cont=0, i,j,a;
    int aux;
    Instance *inst;
	std::string nf = "../Instances/" + std::string(fileName);
    //char nf[30]="";
    //strcat(nf,"../Instances/");
    //strcat(nf,fileName);
    arq=fopen(nf.c_str(), "r");
    if(arq==NULL)
    {
        std::cout<<"Erro, couldn't open the file:"<<fileName<<std::endl;
        //printf("Erro, couldn't open the file: %s.\n",fileName);
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
    //char nf[30]="";
    std::string nf = "";
    int i,j,aux;
    //char n[50];
    if(type == 0)
    {
        nf = nf + std::string("../Residence/Freq_");
    }
    else
    {
        nf = nf + std::string("../Residence/temp_");
    }
    nf = nf + std::string(fileName) +std::string(".txt");
    // strcat(nf,fileName);
    // strcat(nf,".txt");
    f = fopen(nf.c_str(), "r");
    if(f==NULL)
    {
        std::cout<<"Erro, couldn't open the file:"<<nf<<std::endl;
        //printf("Erro, couldn't open the file: %s.\n", nf);
    }
    else
    {
        while((fscanf(f,"x(%d,%d) = %d\n",&i,&j,&aux))!=EOF)
        {
            residence[i-1] = aux; //increment 1 for occurrence.
            sol[i-1] = j-1;
        }

    }
    fclose(f);
}


void load_m_reference_sol(const char *fileName, int *m_reference, Instance *inst)
{
    FILE *f;
    std::string nf = "../Residence/Freq2_" +std::string(fileName) +std::string(".txt");
    int i,j,aux;

    // strcat(nf,"../Residence/Freq2_");
    // strcat(nf,fileName);
    // strcat(nf,".txt");
    f = fopen(nf.c_str(), "r");
    if(f==NULL)
    {
        std::cout<<"Erro, couldn't open the file:"<<nf<<std::endl;
        //printf("Erro, couldn't open the file: %s .\n",nf);

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
    double *c_variables, *lb, *ub;
    char *integer;
    char **name;
    int *indexes;
    double *cof, *cof_temp, *n_right;
    int *index_temp, *index_temp2;
   // std::string nome="x";
    std::string n(fileName);
    //strcpy(n,fileName);
    std::string aux;
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
        name[i] = new char[255];
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

            std::string nome = "x(" + std::to_string(i+1) + "," + std::to_string(j+1)+")";
            
            // strcpy(nome,"x(");
            // sprintf(aux,"%d,",i+1);
            // strcat(nome,aux);
            // sprintf(aux,"%d)",j+1);
            // //strcat(nome,aux);
            strcpy(name[iReturn(i,j,inst->nJobs,inst->mAgents)], nome.c_str());
            
        }
    }

    /*Add function objetive*/
    lp_add_cols(lp,number_variables,c_variables,lb,ub,integer,name);
    for(i=0; i<inst->mAgents; i++)
    {
        std::string nome = "r" + std::to_string(i+1);
        // strcpy(nome,"r");
        // sprintf(aux,"%d",i+1);
        // strcat(nome,aux);
        for(j=0; j<inst->nJobs; j++)
        {
            cof[j] = inst->resourceAgent[iReturn(j,i,inst->nJobs,inst->mAgents)];
            index_temp2[j] = indexes[iReturn(j,i,inst->nJobs,inst->mAgents)];
        }
        lp_add_row(lp,inst->nJobs,index_temp2,cof,nome.c_str(),'L',n_right[i]);
    }

    for(i=0; i<inst->nJobs; i++)
    {
        //strcpy(nome,"r");
        std::string nome = "r" + std::to_string(i+inst->mAgents+1);
        //sprintf(aux,"%d",i+inst->mAgents+1);
        //strcat(nome,aux);
        for(j=0; j<inst->mAgents; j++)
        {
            index_temp[j] = indexes[iReturn(i,j,inst->nJobs,inst->mAgents)];
            cof_temp[j]=1;
        }

        lp_add_row(lp,inst->mAgents,index_temp,cof_temp,nome.c_str(),'E',1);
    }

    for(i=0; i<number_variables; i++)
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
    char aux[SZMAXPATH];
    double *mip_coef = (double*)malloc(sizeof(double)*(inst->nJobs*inst->mAgents));
    mip_name = (char**)malloc(sizeof(char*)*(inst->nJobs*inst->mAgents));

    for(i=0; i<inst->nJobs; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            // indexes[i] = iReturn(i,sol[i],inst->nJobs,inst->mAgents);
            mip_name[iReturn(i,j,inst->nJobs,inst->mAgents)] = new char[SZMAXPATH];
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
    for(i=0; i<inst->nJobs*inst->mAgents; i++)
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
#if PRINTALL
    std::cout<<"Jobs Fixadas:"<<std::endl;
    //printf("Jobs Fixadas: \n");
#endif
    for(i=0; i<sizeFixHard; i++)
    {
        for(j=0; j<inst->mAgents; j++)
        {
            if(j == sol[v_ordem[i]])
            {
                lp_fix_col(lp,iReturn(v_ordem[i],j,inst->nJobs,inst->mAgents),1);
                #if PRINTALL
                    std::cout<<v_ordem[i]+1<<"-";
                    //printf("%d-",v_ordem[i]+1);
                #endif
            }
            else
            {
                lp_fix_col(lp,iReturn(v_ordem[i],j,inst->nJobs,inst->mAgents),0);
            }
        }

    }
    #if PRINTALL
        std::cout<<std::endl;
        //printf("\n");
    #endif

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
    return cost_objective;
}

LinearProgramPtr include_constrain_canonic(LinearProgramPtr lp, Instance *inst, int value_obj)
{

    char nome[SZMAXPATH];
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
    double v_max, v_min, aux, soma = 0.0;
    int *index = (int*)malloc(sizeof(int)*inst->nJobs);
    int i;
    char nome[SZMAXPATH] ="k2";
    v_max = v_min = float(residence[0]);
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
    int nonZeroCof = 0;
    for(i = 0; i<inst->nJobs; i++){
        aux = float(residence[i]);
        //printf("\naux:%f min: %f, max:%f", aux, v_min,v_max);
        double cofTemp = (aux - v_min)/(v_max - v_min);
        if(cofTemp!=0){
            cof[nonZeroCof] = (aux - v_min)/(v_max - v_min);
            soma += cof[nonZeroCof];
            index[nonZeroCof] = iReturn(i, sol[i],inst->nJobs,inst->mAgents);
            nonZeroCof++;
        }
        //printf("\ncof[%d]: %f, res: %d ", index[i], cof[i],residence[i]);
    }
    soma -= sizeFixSoft;
    lp_add_row(lp,nonZeroCof,index,cof,nome,'G', soma);  
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
    char line[255];
    int nLine = 0;
    int nRCols = 0;
    char col[4][255];
    //mip_coef = (double*)malloc(sizeof(double)*inst->nJobs);
    //mip_name = (char**)malloc(sizeof(char*)*inst->nJobs);
    //printf("nJobs %d \n",inst->nJobs);

    //for(i=0; i<inst->nJobs; i++)
    //{
    //    mip_name[i]=new char[10];
    //}

    //lp_read_mip_start(n,mip_name,mip_coef);

    while (fgets( line, 255, f ))
    {
        ++nLine;

        int nread = sscanf( line, "%s %s %s %s", col[0], col[1], col[2], col[3] );

        if (!nread)
            continue;

        if (strlen(col[0])&&isdigit(col[0][0])&&(nread>=3))
        {
            sscanf(col[1],"x(%d,%d)",&a1,&a2);
//            printf("x(%d,%d)\n", a1,a2);
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
    bool FixedAllFlag = false;
    int i,j, v_d = sizeFixSoft;
    int fo_ini, fo_antes, fo_depois;
    //int quantidade = sizeFixHard;
    char n[SZMAXPATH];
    char n2[SZMAXPATH];
    Instance* inst;
    inst = loadInstance(fileName);
    LinearProgramPtr lp = geraLP(fileName,inst);
    int *sol = (int*)malloc(sizeof(int)*inst->nJobs); //save solution
    int *residence = (int*)malloc(sizeof(int)*inst->nJobs);
    int *m_reference = (int*)malloc(sizeof(int)*(inst->nJobs*inst->mAgents));
    load_residence_sol(fileName,residence, sol,inst,0); // Load residence and sol
    load_m_reference_sol(fileName,m_reference,inst);//Load reference
    lp = set_solution_lp(lp,sol,inst);//Set MIP start
    fo_ini = fo_antes = return_objective(inst,sol);//Load Fo Initial
   //Set Parameter Time
    //lp_set_max_solutions(lp,5);//Set Parameter Solution
    strcpy(n,fileName);
    strcat(n,".lp");
    strcpy(n2,"../Solution/sol_");
    strcat(n2,fileName);
    strcat(n2,".txt");
    int res_retirada = inst->nJobs+inst->mAgents;

    int erro = 0;
	struct timeval inicio, inicioImp;
	struct timeval fim, fimImp;
    float tmili = 0, tImprovement = 0;
    gettimeofday(&inicio,NULL);
    gettimeofday(&inicioImp,NULL);
    long int countIte = 0;
    //while((/*(quantidade<=inst->nJobs)&&*/tmili<time*60000)&&(tImprovement<20000))
    while((tmili<time*60000)&&(tImprovement<20000))
    {
        //lp = fixed_variables(lp,residence,inst->nJobs - quantidade,inst,sol);//Fixed variable
        #if PRINTALL
            std::cout<<fo_antes<<"\t";
            //printf("fo: %d\n",fo_antes);
        #endif
        lp = include_constrain_canonic(lp, inst, fo_antes);
        lp = incluce_constrain_soft(lp,inst,residence,sol,sizeFixSoft);
        
        //printf("Time %f\n",tmili);
        lp_write_lp(lp,n);
        //printf("teste = %d\n",i);
        //setNodeInfinity(lp);
        setNodeLimit(lp,500);
        lp_set_max_seconds(lp,TIMERERUN);
        ///printf("time: %f\n",time*60-(int)(tmili/1000));
        getcallback(lp,time*60-(int)(tmili/1000));
        i=lp_optimize(lp);
        //lp_mipstart_debug(lp);
        fo_depois = lp_obj_value(lp);
        gettimeofday(&fim, NULL);
        tmili = 1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000;
        gettimeofday(&fimImp, NULL);
        tImprovement = 1000 * (fimImp.tv_sec - inicioImp.tv_sec) + (fimImp.tv_usec - inicioImp.tv_usec) / 1000;
        if(fo_depois<fo_antes)
        {
            //printf("%s\n",n2);
            lp_write_sol(lp,n2);
            sizeFixSoft = v_d;
            //quantidade = sizeFixHard;
            update_res_ref(lp,residence,m_reference,sol,inst, n2);
            tImprovement = 0;
            gettimeofday(&inicioImp, NULL);
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

        #if PRINTALL
            std::cout<<fo_depois<<"\t"<<tmili<<std::endl;
        #endif
        countIte++;
         if(sizeFixSoft >inst->nJobs){
            FixedAllFlag= true;
            break;
         }
 
 
    }

        
        if(FixedAllFlag){
            #if PRINTALL
                std::cout<<"Fixed all jobs (Opt Solution):";
            #endif
            #if PRINTDEFAULT   
                std::cout<<"1\t";
            #endif

        }
        #if PRINTDEFAULT
            else{
                std::cout<<"0\t";
            }
        #endif
    
    gettimeofday(&fim, NULL);
    tmili = 1000 * (fim.tv_sec - inicio.tv_sec) + (fim.tv_usec - inicio.tv_usec) / 1000;
    #if PRINTDEFAULT
        std::cout<<fo_ini<<"\t"<<fo_antes<<"\t"<<tmili<<"\t"<<countIte<<std::endl;
        //printf("Final Time: %f\n",tmili);
    #endif
    lp_free(&lp);
    
    //remove(n);
    

    //freeInstance(inst);
    free(sol);
    free(residence);
    free(m_reference);
    free(inst);
}



int main(int argc,char *argv[])
{
    const char *fileName = argv[1];
    //int sizeFixHard = atoi(argv[2]);
    int sizeFixSoft = atoi(argv[2]);
    float time = atof(argv[3]);
    runSolver(fileName, /*sizeFixHard,*/ sizeFixSoft,time);
    #if PRINTALL
        std::cout<<"Sucessed run"<<std::endl;
        //printf("Sucessed run\n");
    #endif
    return 0;
}

