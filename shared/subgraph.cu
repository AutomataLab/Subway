#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>
#include <stdlib.h>

template <class E>
Subgraph<E>::Subgraph(uint num_nodes, ull num_edges)
{
    cudaProfilerStart();
    cudaError_t error;
    cudaDeviceProp dev;
    int deviceID;
    cudaGetDevice(&deviceID);
    error = cudaGetDeviceProperties(&dev, deviceID);
    if(error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaProfilerStop();

    std::cout << "num_edges: " << num_edges << std::endl;
    std::cout << "device memory(bytes): " << dev.totalGlobalMem << std::endl;

    float estimated_gpu_memory_size = float(num_nodes) * 20 * 4;
    if (dev.totalGlobalMem > estimated_gpu_memory_size ){
       max_partition_size = 0.9 * (dev.totalGlobalMem - estimated_gpu_memory_size) / sizeof(E);
    }else {
        std::cout << "no sufficient memory" << std::endl;
        exit(-1);
    }
    //if(max_partition_size > DIST_INFINITY)
    //  max_partition_size = DIST_INFINITY;
    std::cout << "Max Partition Size: " << max_partition_size << std::endl;
    this->num_nodes = num_nodes;
    this->num_edges = num_edges;

    ull m = num_nodes;
    gpuErrorcheck(cudaMallocHost(&activeNodes, m * sizeof(uint)));
    gpuErrorcheck(cudaMallocHost(&activeNodesPointer, (m+1) * sizeof(ull)));
    gpuErrorcheck(cudaMallocHost(&activeEdgeList, num_edges * sizeof(E)));

    gpuErrorcheck(cudaMalloc(&d_activeNodes, m * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_activeNodesPointer, (m+1) * sizeof(ull)));
    gpuErrorcheck(cudaMalloc(&d_activeEdgeList, (max_partition_size) * sizeof(E)));
    std::cout << "subgraph .." << std::endl;
}

template class Subgraph<OutEdge>;
template class Subgraph<OutEdgeWeighted>;

// For initialization with one active node
//unsigned int numActiveNodes = 1;
//subgraph.activeNodes[0] = SOURCE_NODE;
//for(unsigned int i=graph.nodePointer[SOURCE_NODE], j=0; i<graph.nodePointer[SOURCE_NODE] + graph.outDegree[SOURCE_NODE]; i++, j++)
//    subgraph.activeEdgeList[j] = graph.edgeList[i];
//subgraph.activeNodesPointer[0] = 0;
//subgraph.activeNodesPointer[1] = graph.outDegree[SOURCE_NODE];
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodes, subgraph.activeNodes, numActiveNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer, subgraph.activeNodesPointer, (numActiveNodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));


