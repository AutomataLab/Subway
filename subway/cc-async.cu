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

    ArgumentParser arguments(argc, argv, true, false);
    Timer timer;
    timer.Start();
    Graph<OutEdge> graph(arguments.input, false);
    graph.ReadGraph();
    float readtime = timer.Finish();
    cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
    for(uint i=0; i<graph.num_nodes; i++)
    {
        graph.value[i] = i;
        graph.label1[i] = true;
        graph.label2[i] = false;
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
    uint gItr = 0;
    bool finished;
    bool *d_finished;
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    while (subgraph.numActiveNodes>0)
    {
        gItr++;
        partitioner.partition(subgraph, subgraph.numActiveNodes);
        // a super iteration
        for(int i=0; i<partitioner.numPartitions; i++)
        {
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();

            //moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
            mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
            uint itr = 0;
            do
            {
                itr++;
                finished = true;
                gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
                cc_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
                                                        partitioner.fromNode[i],
                                                        partitioner.fromEdge[i],
                                                        subgraph.d_activeNodes,
                                                        subgraph.d_activeNodesPointer,
                                                        subgraph.d_activeEdgeList,
                                                        graph.d_outDegree,
                                                        graph.d_value, 
                                                        d_finished,
                                                        (itr%2==1) ? graph.d_label1 : graph.d_label2,
                                                        (itr%2==1) ? graph.d_label2 : graph.d_label1);

                cudaDeviceSynchronize();
                gpuErrorcheck( cudaPeekAtLastError() );
                gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
            }while(!(finished));
            cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
        }
        subgen.generate(graph, subgraph);
    }
    float runtime = timer.Finish();
    cout << "Processing finished in " << runtime/1000 << " (s).\n";
    gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, n*sizeof(uint), cudaMemcpyDeviceToHost));
    utilities::PrintResults(graph.value, min(30, graph.num_nodes));
    if(arguments.hasOutput)
        utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

