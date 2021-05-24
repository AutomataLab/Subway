#ifndef GRAPH_CUH
#define GRAPH_CUH


#include "globals.hpp"

template <class E>
class Graph
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	uint num_nodes;
	uint num_edges;
	uint *nodePointer;
	E *edgeList;
	uint *outDegree;
	bool *label1;
	bool *label2;
	uint *value;
	uint *d_outDegree;
	uint *d_value;
	bool *d_label1;
	bool *d_label2;
	string graphFormat;
    Graph(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReadGraph();
};

template <class E>
class GraphPR
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	uint num_nodes;
	uint num_edges;
	uint *nodePointer;
	E *edgeList;
	uint *outDegree;
	float *value;
	float *delta;
	uint *d_outDegree;
	float *d_value;
	float *d_delta;
	string graphFormat;
    GraphPR(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReadGraph();
};

#endif	//	GRAPH_CUH



