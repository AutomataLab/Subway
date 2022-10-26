#ifndef SUBGRAPH_GENERATOR_HPP
#define SUBGRAPH_GENERATOR_HPP


#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>

template <class E>
class SubgraphGenerator
{
private:

public:
    uint *activeNodesLabeling;
    uint *activeNodesDegree;
    uint *prefixLabeling;
    ull *prefixSumDegrees;
    uint *d_activeNodesLabeling;
    uint *d_activeNodesDegree;
    uint *d_prefixLabeling;
    ull *d_prefixSumDegrees;
    SubgraphGenerator(Graph<E> &graph);
    SubgraphGenerator(GraphPR<E> &graph);
    void generate(Graph<E> &graph, Subgraph<E> &subgraph);
    void generate(GraphPR<E> &graph, Subgraph<E> &subgraph, float acc);
};

#endif    //    SUBGRAPH_GENERATOR_HPP



