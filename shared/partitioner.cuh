#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH


#include "globals.hpp"
#include "subgraph.cuh"

template <class E>
class Partitioner
{
private:

public:
    uint numPartitions;
    vector<uint> fromNode;
    vector<ull> fromEdge;
    vector<uint> partitionNodeSize;
    vector<ull> partitionEdgeSize;
    Partitioner();
    void partition(Subgraph<E> &subgraph, uint numActiveNodes);
    void reset();
};

#endif    //    PARTITIONER_CUH



