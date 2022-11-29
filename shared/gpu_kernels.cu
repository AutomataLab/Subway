#include "gpu_kernels.cuh"
#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include "subgraph.cuh"


__global__ void bfs_kernel(uint numNodes,
                           uint from,
                           ull numPartitionedEdges,
                           uint *activeNodes,
                           ull *activeNodesPointer,
                           OutEdge *edgeList,
                           uint *outDegree,
                           uint *value,
                           bool *label1,
                           bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = value[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        uint finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            //finalDist = sourceWeight + edgeList[i].w8;
            finalDist = sourceWeight + 1;
            if(finalDist < value[edgeList[i].end])
            {
                atomicMin(&value[edgeList[i].end] , finalDist);

                //*finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}

__global__ void cc_kernel(uint numNodes,
                          uint from,
                          ull numPartitionedEdges,
                          uint *activeNodes,
                          ull *activeNodesPointer,
                          OutEdge *edgeList,
                          uint *outDegree,
                          uint *dist,
                          //bool *finished,
                          bool *label1,
                          bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        //unsigned int finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            //finalDist = sourceWeight + edgeList[i].w8;
            if(sourceWeight < dist[edgeList[i].end])
            {
                atomicMin(&dist[edgeList[i].end] , sourceWeight);

                //*finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}


__global__ void sssp_kernel(uint numNodes,
                            uint from,
                            ull numPartitionedEdges,
                            uint *activeNodes,
                            ull *activeNodesPointer,
                            OutEdgeWeighted *edgeList,
                            uint *outDegree,
                            uint *dist,
                            //bool *finished,
                            bool *label1,
                            bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;

        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        uint finalDist;
        for(ull i=thisFrom; i < thisTo; i++)
        {
            finalDist = sourceWeight + edgeList[i].w8;
            if(finalDist < dist[edgeList[i].end])
            {
                atomicMin(&dist[edgeList[i].end] , finalDist);

                //*finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}

__global__ void sswp_kernel(uint numNodes,
                            uint from,
                            ull numPartitionedEdges,
                            uint *activeNodes,
                            ull *activeNodesPointer,
                            OutEdgeWeighted *edgeList,
                            uint *outDegree,
                            uint *dist,
                            //bool *finished,
                            bool *label1,
                            bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        uint finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            finalDist = min(sourceWeight, edgeList[i].w8);
            if(finalDist > dist[edgeList[i].end])
            {
                atomicMax(&dist[edgeList[i].end] , finalDist);

                //*finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}

__global__ void pr_kernel(uint numNodes,
                          uint from,
                          ull numPartitionedEdges,
                          uint *activeNodes,
                          ull *activeNodesPointer,
                          OutEdge *edgeList,
                          uint *outDegree,
                          float *dist,
                          float *delta,
                          //bool *finished,
                          float acc)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        uint degree = outDegree[id];
        float thisDelta = delta[id];

        if(thisDelta > acc)
        {
            dist[id] += thisDelta;
            if(degree != 0)
            {
                //*finished = false;
                float sourcePR = ((float) thisDelta / degree) * 0.85;

                ull thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
                ull thisto = thisfrom + degree;
                for(ull i=thisfrom; i<thisto; i++)
                {
                    atomicAdd(&delta[edgeList[i].end], sourcePR);
                }
            }
            atomicAdd(&delta[id], -thisDelta);
        }
    }
}


__global__ void bfs_async(uint numNodes,
                          uint from,
                          ull numPartitionedEdges,
                          uint *activeNodes,
                          ull *activeNodesPointer,
                          OutEdge *edgeList,
                          uint *outDegree,
                          uint *dist,
                          bool *finished,
                          bool *label1,
                          bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        uint finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            //finalDist = sourceWeight + edgeList[i].w8;
            finalDist = sourceWeight + 1;
            if(finalDist < dist[edgeList[i].end])
            {
                atomicMin(&dist[edgeList[i].end] , finalDist);

                *finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}

__global__ void sssp_async(uint numNodes,
                           uint from,
                           ull numPartitionedEdges,
                           uint *activeNodes,
                           ull *activeNodesPointer,
                           OutEdgeWeighted *edgeList,
                           uint *outDegree,
                           uint *dist,
                           bool *finished,
                           bool *label1,
                           bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        uint finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            finalDist = sourceWeight + edgeList[i].w8;
            if(finalDist < dist[edgeList[i].end])
            {
                atomicMin(&dist[edgeList[i].end] , finalDist);

                *finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}

__global__ void sswp_async(uint numNodes,
                           uint from,
                           ull numPartitionedEdges,
                           uint *activeNodes,
                           ull *activeNodesPointer,
                           OutEdgeWeighted *edgeList,
                           uint *outDegree,
                           uint *dist,
                           bool *finished,
                           bool *label1,
                           bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;
        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        uint thisTo = thisFrom + degree;
        uint finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            finalDist = min(sourceWeight, edgeList[i].w8);
            if(finalDist > dist[edgeList[i].end])
            {
                atomicMax(&dist[edgeList[i].end] , finalDist);

                *finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}


__global__ void cc_async(uint numNodes,
                         uint from,
                         ull numPartitionedEdges,
                         uint *activeNodes,
                         ull *activeNodesPointer,
                         OutEdge *edgeList,
                         uint *outDegree,
                         uint *dist,
                         bool *finished,
                         bool *label1,
                         bool *label2)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        if(label1[id] == false)
            return;
        label1[id] = false;

        uint sourceWeight = dist[id];

        ull thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
        uint degree = outDegree[id];
        ull thisTo = thisFrom + degree;
        //printf("******* %i\n", thisFrom);
        //unsigned int finalDist;
        for(ull i=thisFrom; i<thisTo; i++)
        {
            //finalDist = sourceWeight + edgeList[i].w8;
            if(sourceWeight < dist[edgeList[i].end])
            {
                atomicMin(&dist[edgeList[i].end] , sourceWeight);

                *finished = false;
                //label1[edgeList[i].end] = true;

                label2[edgeList[i].end] = true;
            }
        }
    }
}


__global__ void pr_async(uint numNodes,
                         uint from,
                         ull numPartitionedEdges,
                         uint *activeNodes,
                         ull *activeNodesPointer,
                         OutEdge *edgeList,
                         uint *outDegree,
                         float *dist,
                         float *delta,
                         bool *finished,
                         float acc)
{
    uint tId = blockDim.x * blockIdx.x + threadIdx.x;

    if(tId < numNodes)
    {
        uint id = activeNodes[from + tId];
        uint degree = outDegree[id];
        float thisDelta = delta[id];

        if(thisDelta > acc)
        {
            dist[id] += thisDelta;
            if(degree != 0)
            {
                *finished = false;
                float sourcePR = ((float) thisDelta / degree) * 0.85;

                ull thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
                ull thisto = thisfrom + degree;
                for(ull i=thisfrom; i<thisto; i++)
                {
                    atomicAdd(&delta[edgeList[i].end], sourcePR);
                }
            }
            atomicAdd(&delta[id], -thisDelta);
        }
    }
}



__global__ void clearLabel(uint * activeNodes, bool *label, uint size, uint from)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size)
    {
        label[activeNodes[id+from]] = false;
    }
}

__global__ void mixLabels(uint * activeNodes, bool *label1, bool *label2, uint size, uint from)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size){
        uint nID = activeNodes[id+from];
        label1[nID] = label1[nID] || label2[nID];
        label2[nID] = false;
    }
}

__global__ void moveUpLabels(uint * activeNodes, bool *label1, bool *label2, uint size, uint from)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    uint nID;
    if(id < size){
        nID = activeNodes[id+from];
        label1[nID] = label2[nID];
        label2[nID] = false;
    }
}

