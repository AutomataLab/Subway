#include "graph.cuh"
#include "gpu_error_check.cuh"

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
    this->graphFilePath = graphFilePath;
    this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{
    cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
    this->graphFormat = GetFileExtension(graphFilePath);
    if(graphFormat == "bcsr" || graphFormat == "bwcsr")
    {
        ifstream infile (graphFilePath, ios::in | ios::binary);
        infile.read ((char*)&num_nodes, sizeof(uint));
        infile.read ((char*)&num_edges, sizeof(ull));
        nodePointer = new ull[num_nodes+1];
        gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
        ull num_node = num_nodes;
        infile.read ((char*)nodePointer, (num_node + 1 ) * sizeof(ull) );
        infile.read ((char*)edgeList, num_edges * sizeof(E) );
        std::cout << nodePointer[num_nodes] << std::endl;
    }
    else
    {
        cout << "The graph format is not supported!\n";
        exit(-1);
    }
    outDegree  = new uint[num_nodes];
    for(uint i=1; i<num_nodes; i++)
        outDegree[i-1] = nodePointer[i] - nodePointer[i-1];

    outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
    label1 = new bool[num_nodes];
    label2 = new bool[num_nodes];
    value  = new uint[num_nodes];

    ull n = num_nodes;
    gpuErrorcheck(cudaMalloc(&d_outDegree, n * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_value, n * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_label1, n * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_label2, n * sizeof(bool)));
    cout << "Done reading.\n";
    cout << "Number of nodes = " << num_nodes << endl;
    cout << "Number of edges = " << num_edges << endl;
}

//--------------------------------------

template <class E>
GraphPR<E>::GraphPR(string graphFilePath, bool isWeighted)
{
    this->graphFilePath = graphFilePath;
    this->isWeighted = isWeighted;
}

template <class E>
string GraphPR<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphPR<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void GraphPR<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void GraphPR<E>::ReadGraph()
{

    cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
    this->graphFormat = GetFileExtension(graphFilePath);
    if(graphFormat == "bcsr" || graphFormat == "bwcsr")
    {
        ifstream infile (graphFilePath, ios::in | ios::binary);
        infile.read ((char*)&num_nodes, sizeof(uint));
        infile.read ((char*)&num_edges, sizeof(ull));
        nodePointer = new ull[num_nodes+1];
        gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
        ull n = num_nodes;
        infile.read ((char*)nodePointer, (n+1) * sizeof(ull));
        infile.read ((char*)edgeList, num_edges * sizeof(E));
        nodePointer[num_nodes] = num_edges;
    }
    else if(graphFormat == "el" || graphFormat == "wel")
    {
        ifstream infile;
        infile.open(graphFilePath);
        stringstream ss;
        uint max = 0;
        string line;
        ull edgeCounter = 0;
        if(isWeighted)
        {
            vector<EdgeWeighted> edges;
            EdgeWeighted newEdge;
            while(getline( infile, line ))
            {
                ss.str("");
                ss.clear();
                ss << line;
                ss >> newEdge.source;
                ss >> newEdge.end;
                ss >> newEdge.w8;
                edges.push_back(newEdge);
                edgeCounter++;
                if(max < newEdge.source)
                    max = newEdge.source;
                if(max < newEdge.end)
                    max = newEdge.end;
            }
            infile.close();
            num_nodes = max + 1;
            num_edges = edgeCounter;
            nodePointer = new ull[num_nodes+1];
            gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
            uint *degree = new uint[num_nodes];
            for(uint i=0; i<num_nodes; i++)
                degree[i] = 0;
            for(ull i=0; i<num_edges; i++)
                degree[edges[i].source]++;
            ull counter=0;
            for(uint i=0; i<num_nodes; i++)
            {
                nodePointer[i] = counter;
                counter = counter + degree[i];
            }
            nodePointer[num_nodes] = num_edges;
            uint *outDegreeCounter  = new uint[num_nodes];
            for(uint i = 0; i < num_nodes; i++){
                 outDegreeCounter[i] = 0;
            }
            ull location;  
            for(ull i=0; i<num_edges; i++)
            {
                location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
                edgeList[location].end = edges[i].end;
                if(isWeighted)
                    AssignW8(edges[i].w8, location);
                    //edgeList[location].w8 = edges[i].w8;
                outDegreeCounter[edges[i].source]++;  
            }
            edges.clear();
            delete[] degree;
            delete[] outDegreeCounter;
        }
        else
        {
            vector<Edge> edges;
            Edge newEdge;
            while(getline( infile, line ))
            {
                ss.str("");
                ss.clear();
                ss << line;
                ss >> newEdge.source;
                ss >> newEdge.end;
                edges.push_back(newEdge);
                edgeCounter++;
                if(max < newEdge.source)
                    max = newEdge.source;
                if(max < newEdge.end)
                    max = newEdge.end;
            }
            infile.close();
            num_nodes = max + 1;
            num_edges = edgeCounter;
            nodePointer = new ull[num_nodes+1];
            gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
            uint *degree = new uint[num_nodes];
            for(uint i=0; i<num_nodes; i++)
                degree[i] = 0;
            for(ull i=0; i<num_edges; i++)
                degree[edges[i].source]++;
            ull counter=0;
            for(uint i=0; i<num_nodes; i++)
            {
                nodePointer[i] = counter;
                counter = counter + degree[i];
            }
            nodePointer[num_nodes] = num_edges;
            uint *outDegreeCounter  = new uint[num_nodes];
            for (uint i = 0; i < num_nodes; i++){
                 outDegreeCounter[i] = 0;
            }
            ull location;  
            for(ull i=0; i<num_edges; i++)
            {
                location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
                edgeList[location].end = edges[i].end;
                //if(isWeighted)
                //    edgeList[location].w8 = edges[i].w8;
                outDegreeCounter[edges[i].source]++;  
            }
            edges.clear();
            delete[] degree;
            delete[] outDegreeCounter;
        }
    }
    else
    {
        cout << "The graph format is not supported!\n";
        exit(-1);
    }
    outDegree  = new uint[num_nodes]();
    for(uint i=1; i<num_nodes; i++)
        outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
    outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];

    std::cout << "last nodePointer: " << nodePointer[num_nodes] << std::endl;

    value  = new float[num_nodes];
    delta  = new float[num_nodes];
    gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
    cout << "Done reading.\n";
    cout << "Number of nodes = " << num_nodes << endl;
    cout << "Number of edges = " << num_edges << endl;

}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;

template class GraphPR<OutEdge>;
template class GraphPR<OutEdgeWeighted>;
