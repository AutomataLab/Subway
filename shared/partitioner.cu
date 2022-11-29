#include "partitioner.cuh"
#include "gpu_error_check.cuh"

template <class E>
Partitioner<E>::Partitioner()
{
    reset();
}

template <class E>
void Partitioner<E>::partition(Subgraph<E> &subgraph, uint numActiveNodes)
{
    reset();
    uint from, to;
    uint left, right, mid;
    ull partitionSize;
    uint numNodesInPartition;
    ull numPartitionedEdges;
    bool foundTo;
    ull accurCount;
    from = 0;
    to = numActiveNodes; // last in pointers
    numPartitionedEdges = 0;
    do
    {
        left = from;
        right = numActiveNodes;

        std::cout << "#active nodes: " << numActiveNodes << std::endl;
        std::cout << "left: " << left << "    right: " << right << std::endl;
        std::cout << "pointer to left: " << subgraph.activeNodesPointer[left] << "    pointer to right: " << subgraph.activeNodesPointer[right] << std::endl;

        partitionSize = subgraph.activeNodesPointer[right] - subgraph.activeNodesPointer[left];
        //std::cout << "partitionSize: " << partitionSize << std::endl;

        if(partitionSize <= subgraph.max_partition_size)
        {
            to = right;
        }
        else
        {
            foundTo = false;
            accurCount = 10;
            while(foundTo==false || accurCount>0)
            {
                mid = (left + right)/2;
                partitionSize = subgraph.activeNodesPointer[mid] - subgraph.activeNodesPointer[from];
                if(foundTo == true)
                    accurCount--;
                if(partitionSize <= subgraph.max_partition_size)
                {
                    left = mid;
                    to = mid;
                    foundTo = true;
                }
                else
                {
                    right = mid;  
                }
            }
            if(to == numActiveNodes)
            {
                cout << "Error in Partitioning...\n";
                exit(-1);
            }

        }

        partitionSize = subgraph.activeNodesPointer[to] - subgraph.activeNodesPointer[from];
        numNodesInPartition = to - from;

        //std::cout << "from: " << from << "   to: " << to << std::endl;
        //std::cout << "#nodes in P: " << numNodesInPartition << "    #edges in P: " << partitionSize << std::endl;
        fromNode.push_back(from);
        fromEdge.push_back(numPartitionedEdges);
        partitionNodeSize.push_back(numNodesInPartition);
        partitionEdgeSize.push_back(partitionSize);
        from = to;
        numPartitionedEdges += partitionSize;
    } while (to != numActiveNodes);
    numPartitions = fromNode.size();
}

template <class E>
void Partitioner<E>::reset()
{
    fromNode.clear();
    fromEdge.clear();
    partitionNodeSize.clear();
    partitionEdgeSize.clear();
    numPartitions = 0;
}

template class Partitioner<OutEdge>;
template class Partitioner<OutEdgeWeighted>;
