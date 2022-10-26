#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"


int main(int argc, char** argv)
{
    cudaFree(0);

    ArgumentParser arguments(argc, argv, false, false);
    Timer timer;
    timer.Start();
    Graph<OutEdge> graph(arguments.input, false);
    graph.ReadGraph();
    float readtime = timer.Finish();
    cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

    for(uint i=0; i < graph.num_nodes; i++)
    {
        graph.value[i] = i;
        graph.label1[i] = false;
        graph.label2[i] = true;
    }

    ull n = graph.num_nodes;
    gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, n * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, n * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, n * sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, n * sizeof(bool), cudaMemcpyHostToDevice));
    Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);

    SubgraphGenerator<OutEdge> subgen(graph);

    subgen.generate(graph, subgraph);

    Partitioner<OutEdge> partitioner;
    timer.Start();
    uint itr = 0;


    while (subgraph.numActiveNodes>0)
    {
        std::cout << "number_of_active_nodes: " << subgraph.numActiveNodes << std::endl;
        itr++;
        partitioner.partition(subgraph, subgraph.numActiveNodes);


        std::cout << "number of partitions: " << partitioner.numPartitions << std::endl;
        gpuErrorcheck( cudaPeekAtLastError() );


        // a super iteration
        for(int i=0; i<partitioner.numPartitions; i++)
        {
            std::cout << "partition: " << i << std::endl;
            std::cout << "partition #edges: " << partitioner.partitionEdgeSize[i] << std::endl;
            std::cout << "partition node size: " << partitioner.partitionNodeSize[i] << std::endl;
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();

            moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
            cc_kernel<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
                                                    partitioner.fromNode[i],
                                                    partitioner.fromEdge[i],
                                                    subgraph.d_activeNodes,
                                                    subgraph.d_activeNodesPointer,
                                                    subgraph.d_activeEdgeList,
                                                    graph.d_outDegree,
                                                    graph.d_value, 
                                                    //d_finished,
                                                    graph.d_label1,
                                                    graph.d_label2);

            cudaDeviceSynchronize();
            gpuErrorcheck( cudaPeekAtLastError() );
        }
        subgen.generate(graph, subgraph);
    }
    float runtime = timer.Finish();
    cout << "Processing finished in " << runtime/1000 << " (s).\n";
    cout << "Number of iterations = " << itr << endl;
    gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, n*sizeof(uint), cudaMemcpyDeviceToHost));
    utilities::PrintResults(graph.value, min(30, graph.num_nodes));
    if(arguments.hasOutput)
        utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

