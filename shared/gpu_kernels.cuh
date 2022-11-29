#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"


__global__ void bfs_kernel(uint numNodes,
                           uint from,
                           ull  numPartitionedEdges,
                           uint *activeNodes,
                           ull *activeNodesPointer,
                           OutEdge *edgeList,
                           uint *outDegree,
                           uint *value,
                           //bool *finished,
                           bool *label1,
                           bool *label2);

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
                          bool *label2);

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
                            bool *label2);

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
                            bool *label2);

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
                         float acc);

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
                          bool *label2);

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
                           bool *label2);


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
                           bool *label2);


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
                         bool *label2);

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
                         float acc);

__global__ void clearLabel(uint * activeNodes, bool *label, uint size, uint from);

__global__ void mixLabels(uint * activeNodes, bool *label1, bool *label2, uint size, uint from);

__global__ void moveUpLabels(uint * activeNodes, bool *label1, bool *label2, uint size, uint from);
