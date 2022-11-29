#include "../shared/globals.hpp"
#include <string>
#include <iostream>

bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

void save_edge_data_to_csr(const std::string& output_filename)
{
    uint max = 0;
    ull edgeCounter = 0;
    vector<Edge> edges;
    Edge newEdge;

    std::string delim = "\t";
    for (std::string line; std::getline(std::cin, line);)
    {
        auto start = 0;
        auto end = line.find(delim);
        newEdge.source = static_cast<uint>(std::stoul(line.substr(start, end - start)));
        start = end + delim.length();
        newEdge.end = static_cast<uint>(std::stoul(line.substr(start)));
        edges.push_back(newEdge);
        edgeCounter++;
        if(max < newEdge.source)
              max = newEdge.source;
        if(max < newEdge.end)
              max = newEdge.end;
    }
    uint num_nodes = max + 1;
    ull num_edges = edgeCounter;
    ull *nodePointer = new ull[num_nodes+1];
    OutEdge *edgeList = new OutEdge[num_edges];
    //out degree
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
    for(ull i=0; i < num_edges; i++)
    {
        if ( i % 20000 == 0)
        {
            std::cout << i << std::endl;
        }
        ull location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
        edgeList[location].end = edges[i].end;
        outDegreeCounter[edges[i].source]++;
    }
    edges.clear();
    delete[] degree;
    delete[] outDegreeCounter;

    std::ofstream outfile(output_filename, std::ofstream::binary);
    outfile.write((char*)&num_nodes, sizeof(uint));
    outfile.write((char*)&num_edges, sizeof(ull));
    ull n = num_nodes;
    outfile.write ((char*)nodePointer, (n+1) * sizeof(ull));
    outfile.write ((char*)edgeList, num_edges * sizeof(OutEdge) );
    outfile.close();
}


void parseLine(std::string& line, EdgeWeighted& edge, const std::string& delimiter)
{
   auto start = 0;
   auto end = line.find(delimiter);
   edge.source = static_cast<uint>(std::stoul(line.substr(start, end - start)));
   start = end + delimiter.length();
   end = line.find(delimiter, start);
   edge.end = static_cast<uint>(std::stoul(line.substr(start, end - start)));
   start = end + delimiter.length();
   edge.w8  =  static_cast<uint>(std::stoul(line.substr(start)));
}

void save_weighted_edge_data_to_csr(const std::string& output_filename)
{
   uint max = 0;
   string line;
   ull edgeCounter = 0;
		
   vector<EdgeWeighted> edges;
   EdgeWeighted newEdge;
   std::string delim = "\t";
   for (std::string line; std::getline(std::cin, line);)
   { 
      parseLine(line, newEdge, delim);
      edges.push_back(newEdge);
      edgeCounter++;			
      if(max < newEdge.source)
	  max = newEdge.source;
      if(max < newEdge.end)
	 max = newEdge.end;				
   }			
   uint num_nodes = max + 1;
   ull num_edges = edgeCounter;
   ull *nodePointer = new ull[num_nodes+1];
   OutEdgeWeighted *edgeList = new OutEdgeWeighted[num_edges];
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
   for(ull i=0; i<num_edges; i++)
   {
      ull location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
      edgeList[location].end = edges[i].end;
      edgeList[location].w8 = edges[i].w8;
      outDegreeCounter[edges[i].source]++;  
   }
   edges.clear();
   delete[] degree;
   delete[] outDegreeCounter;

   ull n = num_nodes;   
   std::ofstream outfile(output_filename, std::ofstream::binary);
   outfile.write((char*)&num_nodes, sizeof(uint));
   outfile.write((char*)&num_edges, sizeof(ull));
   outfile.write ((char*)nodePointer, (n + 1) * sizeof(ull));
   outfile.write ((char*)edgeList, num_edges * sizeof(OutEdgeWeighted));
   outfile.close();
}

int main(int argc, char** argv)
{
   if (argc != 3)
   {
      std::cout << "\n usage : cat data_file|converter_stdin file_type output_filename\n";
      std::cout << "file_type: el or wel" << std::endl;
      std::cout << "wel: weighted edge data (src TAB dst TAB weight(uint)" << std::endl;
      std::cout << "el: edge data without weight (src TAB dst)" << std::endl;
      exit(1);
   }
   std::string output_filename(argv[2]);
   std::string file_type(argv[1]);	
   if(file_type == "el")
   {  
      save_edge_data_to_csr(output_filename);
   }
   else if (file_type == "wel")
   {
     save_weighted_edge_data_to_csr(output_filename);
   }
   return 0;
}
