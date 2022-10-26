#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP


#include "globals.hpp"


template <class E>
class Subgraph
{
private:

public:
    uint num_nodes;
    ull num_edges;
    uint numActiveNodes;
    uint *activeNodes;
    ull *activeNodesPointer;
    E *activeEdgeList;
    uint *d_activeNodes;
    ull *d_activeNodesPointer;
    E *d_activeEdgeList;
    ull max_partition_size;
    Subgraph(uint num_nodes, ull num_edges);
};

#endif    //    SUBGRAPH_HPP
