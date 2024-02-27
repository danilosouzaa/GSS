#include "gInstance.cuh"

Instance* createGPUInstance(Instance* h_instance, TnJobs nJobs, TmAgents mAgents){
	//printf("begin createGPUInstance \n");
	size_t size_instance = sizeof(Instance)
				+ sizeof(Tcost) * (nJobs*mAgents)      //cost d_ji
				+ sizeof(TresourcesAgent) * (nJobs*mAgents)    //resourcesAgent r_ji
				+ sizeof(Tcapacity) * (mAgents); // capacity b_i

	// Instance* d_inst_gpu = (Instance*) malloc (size_instance);
	// memcpy(h_inst_gpu,h_instance, size_instance);
	Instance* d_inst;
	gpuMalloc((void**)&d_inst, size_instance);
	gpuMemset(d_inst,0,size_instance);
	h_instance->cost = (Tcost*)(d_inst +1);
	h_instance->resourcesAgent = (TresourcesAgent*) (h_instance->cost + (nJobs*mAgents));
	h_instance->capacity = (Tcapacity*) (h_instance->resourcesAgent + (nJobs*mAgents) );
	h_instance->nJobs = nJobs;
	h_instance->mAgents = mAgents;

	gpuMemcpy(d_inst, h_instance,size_instance, cudaMemcpyHostToDevice);
	return d_inst;


}
