
#include "gapinstance.hpp"
#include "gaproblem.h"
#include <iostream>
#include <string>
#include "log.h"
#include "gpulib/gpu.h"
#include "except.h"
#include "gsaExperiments.hpp"
//using namespace stdgap;
using namespace std;


/*!
 * Initialize environment.
 */
void
envInit(int deviceNumber = 0)
{
    //int     error;

    // Initializes log system
    logInit();

    cudaDeviceProp  prop;
    int             count;

    // Set thread GPU
    cudaSetDevice(deviceNumber);

    // Detect CUDA driver and GPU devices
    switch(cudaGetDeviceCount(&count)) {
    case cudaSuccess:
        for(int d=0;d < count;d++) {
            if(cudaGetDeviceProperties(&prop,d) == cudaSuccess) {
                if(prop.major < 2)
                    WARNING("Device '%s' is not suitable to this application. Device capability %d.%d < 2.0\n",
                            prop.name,prop.major,prop.minor);
            }
            ////lprintf("GPU ok!\n");
        }
        break;
    case cudaErrorNoDevice:
        WARNING("No GPU Devices detected.");
        break;
    case cudaErrorInsufficientDriver:
        WARNING("No CUDA driver installed.");
        break;
    default:
        EXCEPTION("Unknown error detecting GPU devices.");
    }
}

int main(int argc, char **argv)
{
	envInit();
    std::vector<std::string> all_args;
    if (argc > 1) {
        all_args.assign(argv + 1, argv + argc);
    }
    std::string filename ="../Instances/" + all_args[0];

	bool costTour = true;
	bool distRound = false;
	bool coordShift = false;
	//string instance_path = "./instances/01_berlin52.tsp";
	//string instance_path = "./instances/02_kroD100.tsp";
	//string instance_path = "./instances/03_pr226.tsp";
	//string instance_path = "./instances/04_lin318.tsp";
	//string instance_path = "./instances/05_TRP-S500-R1.tsp";
	//string instance_path = "./instances/06_d657.tsp";
	//string instance_path = "./instances/07_rat784.tsp";
	//string instance_path = "./instances/08_TRP-S1000-R1.tsp";

	//string instance_path = "./instances/08_TRP-S1000-R1.tsp";

	GAProblem problem(costTour, distRound, coordShift);
	problem.load(filename.c_str());
    problem.showInstance();


	//int seed = 500; // 0: random
//	WAMCAExperiment exper(problem, seed);
//	exper.runWAMCA2016();


    printf(">>>> BUILT AT %s %s\n",__DATE__,__TIME__);

    ////lprintf("finished successfully\n");

    return 0;
}

//  int main(int argc, char **argv)
//  {
//     std::vector<std::string> all_args;
//     if (argc > 1) {
//         all_args.assign(argv + 1, argv + argc);
//     }
//     std::string fileName ="../Instances/" + all_args[0];
//     std::cout<<fileName<<std::endl;
//     GapInstance inst;
//     inst.read(fileName,"orlibrary");
//     inst.build();

//     int nJ = inst.number_of_items();
//     int mA = inst.number_of_agents();
//     std::cout<<"nJobs:"<<nJ<<std::endl;
//     std::cout<<"mAgents:"<<mA<<std::endl;
//     for(int i=0;i<nJ;i++){
//         for (int j=0;j<mA;j++){
//             std::cout<<"custo de alocar a job "<<i+1<<" ao agent "<<j+1<<": "<<inst.cost(i,j)<<std::endl;
//             std::cout<<"quantidade de recurso consumida ao alocar o job "<<i+1<<" ao agent "<<j+1<<": "<<inst.weight(i,j)<<std::endl;
            
//         }
//         std::cout<<"Agent com maior custo de alocar o job "<<i+1<<": "<<inst.item(i).maximum_cost_agent_id<<std::endl;
//         std::cout<<"Agent com menor custo de alocar o job "<<i+1<<": "<<inst.item(i).minimum_cost_agent_id<<std::endl;
//         std::cout<<"Agent com maior consumo de recurso de alocar o job "<<i+1<<": "<<inst.item(i).maximum_weight_agent_id<<std::endl;
//         std::cout<<"Agent com maior consumo de recurso de alocar o job "<<i+1<<": "<<inst.item(i).minimum_weight_agent_id<<std::endl;
//     }
//     for (int j=0;j<mA;j++){
//         std::cout<<"Capacidade do agent "<<j+1<<": "<<inst.capacity(j)<<std::endl;
//     }
//     std::cout<<"Total Cost: "<<inst.total_cost()<<std::endl;
//     std::cout<<"Maximum Cost: "<<inst.maximum_cost()<<std::endl;
//     std::cout<<"Maximum Weight: "<<inst.maximum_weight()<<std::endl;
    
//     return 0;
//  }
