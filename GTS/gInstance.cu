#include "gInstance.cuh"

Instance* createGPUInstance(const Instance* h_instance, TnJobs nJobs, TmAgents mAgents){
	//printf("begin createGPUInstance \n");
	size_t size_instance = sizeof(Instance)
				+ sizeof(Tcost) * (nJobs*mAgents)      //cost
				+ sizeof(TresourcesAgent) * (nJobs*mAgents)    //resourcesAgent
				+ sizeof(Tcapacity) * (mAgents); // capacity

	Instance* h_inst_gpu = (Instance*) malloc (size_instance);
	memcpy(h_inst_gpu,h_instance, size_instance);
	Instance* d_inst;
	gpuMalloc((void**)&d_inst, size_instance);
	//printf("malloc ok\n");
	//getchar();
	gpuMemset(d_inst,0,size_instance);
	//printf("menset ok\n");
	//getchar();

	h_inst_gpu->cost = (Tcost*)(d_inst +1);
	h_inst_gpu->resourcesAgent = (TresourcesAgent*) (h_inst_gpu->cost + (nJobs*mAgents));
	h_inst_gpu->capacity = (Tcapacity*) (h_inst_gpu->resourcesAgent + (nJobs*mAgents) );

	//printf("adjusting GPU pointers\n");
	//getchar();

	h_inst_gpu->nJobs = nJobs;
	h_inst_gpu->mAgents = mAgents;

	gpuMemcpy(d_inst, h_inst_gpu,size_instance, cudaMemcpyHostToDevice);
	return d_inst;


}
