/**
 * @file	mlproblem.h
 *
 * @brief	Handle a general optimization problem
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <math.h>
//#include "except.h"
//#include "gpulib/log.h"
#include "utils.h"
//#include "tsplib.h"

#include "gpulib/gpu.h"
//#include "mlproblem.h"
#include "gaproblem.h"
#include "gapinstance.hpp"


using namespace std;


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //


#define RLIB_FILE_SIGNATURE     "RLib: Traveling Salesman Problem (TSP)"


// ################################################################################ //
// ##                                                                            ## //
// ##                                GLOBAL VARIABLES                            ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS OFProblem                              ## //
// ##                                                                            ## //
// ################################################################################ //

void
GAProblem::free()
{   
    if(jobs) {

        for(uint i=0;i < nJobs;i++){
            // delete jobs[i];
            delete[] (jobs[i].costAgentsAllocation);
            delete[] jobs[i].resourceConsumed;
        }
        delete[] jobs;
    }
    jobs = NULL;
    nJobs = 0;
    if(agents) {

        for(uint i=0;i < mAgents;i++){
            delete[] (agents[i].jobsAllocated);
        }
        delete[] agents;
    }
    agents = NULL;
    mAgents = 0;
}


void GAProblem::allocData(uint nJ, uint mA)
{
    //int     **rows,*base;
    uint      i;
    free();
    nJobs = nJ;
    mAgents = mA;
    
    jobs = new GAPJobsData[nJobs];
    for(i =0; i< nJobs; i++){
        jobs[i].id = 0;
        jobs[i].idAgentAllocated = 0;
        jobs[i].costAgentsAllocation = new uint[mAgents];
        jobs[i].resourceConsumed = new uint[mAgents];
    }
    agents = new GAPAgentsData[mAgents];
    for(i =0; i< mAgents; i++){
        agents[i].id = 0;
        agents[i].capacity = 0;
        agents[i].resUsage = 0;
        agents[i].jobsAllocated = new bool[nJobs];
    }

}


void
GAProblem::load(const char *fname)
{
    stdgap::GapInstance inst;
    //GAPAgentsData           *agentInst;
    //GAPJobsData             *jobsInst;

    uint                     i,j;
    std::string fPathName =std::string("../Instances/") + std::string(fname);
    timeLoad = sysTimer();

    // dr = distRound ? 0.5F : 0.0F;
    // l4printf("Euclidean distance round: %0.2f\n",dr);
    inst.read(fPathName,"orlibrary");
    inst.build();
    allocData(inst.number_of_items(),inst.number_of_agents());
    for (j=0;j<nJobs;j++){
        jobs[j].id = j;
        for(i=0;i<mAgents;i++){
            jobs[j].costAgentsAllocation[i] = inst.cost(j,i);
            jobs[j].resourceConsumed[i] = inst.weight(j,i);
        }
    }
    for (i=0;i<mAgents;i++){
        agents[i].id = i;
        agents[i].capacity = inst.capacity(i);
        agents[i].resUsage = 0;
        for(j=0;j<nJobs;j++){
            agents[i].jobsAllocated[j]=false;
        }
    }
    timeLoad = sysTimer() - timeLoad;
}

void GAProblem::showInstance(){
    std::cout<<"nJobs:"<<nJobs<<std::endl;
    std::cout<<"mAgents:"<<mAgents<<std::endl;
    std::cout<<"timeLoad:"<<timeLoad<<std::endl;
}

void
GAProblem::save(const char *fname)
{
    std::cout<<"Not implemented!"<<std::endl;
    // ofstream  fout;
    // char      bname[128];
    // uint      i,j;

    // strcpy(bname,fname);
    // stripext(bname);

    // fout.open(fname);

    // fout << RLIB_FILE_SIGNATURE << '\n';
    // fout << bname << '\n';
    // fout << size << ' ' << 0 << '\n';
    // for(i=0;i < size;i++) {
    //     fout << i;
    //     for(j=0;j < i;j++)
    //         fout << ' ' << clients[i].weight[j];
    //     fout << '\n';
    // }
    // fout.close();
}

//MLSolution *
//MLProblem::createSolution()
//{
//    return NULL;
//}
