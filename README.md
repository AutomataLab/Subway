## Subway
Subway is an out-of-GPU-memory graph processing framework.

Subway provides a highly cost-effective solution to extracting a subgraph that only consists of the edges of active vertices. This allows it to transfer only the active parts of the graph from CPU to GPU, thus dramatically reduces the volume of data transfer. The benefits from the data transfer reduction outweigh the costs of subgraph generation in (almost) all iterations of graph processing, bringing in substantial overall performance improvements. Moreover, it supports asynchronous processing between the loaded subgraph in GPU and the rest of the graph in host memory, which tends to decrease the number of global iterations, thus can further reduce the data transfer.

#### Compilation

To compile Subway, just run make in the root directory. The only requrements are g++ and CUDA toolkit.

#### Input graph formats

Subway accepts edge-list (.el) and weighted edge-list (.wel) graph formats, as well as the binary serialized pre-built CSR graph representation (.bcsr and .bwcsr). It is highly recommended to convert edge-list format graph files to the binary format (using tools/converter). Reading binary formats is faster and more space efficient.

Subway is sensitive to graph file extension. A weighted edge-list graph file has to end with .wel. The followings are two graph file examples.

Graph.el ("SOURCE DESTINATION" for each edge in each line):
```
0 1
0 3
2 3
1 2
```

Graph.wel ("SOURCE DESTINATION WEIGHT" for each edge in each line):
```
0 1 26
0 3 33
2 3 40
1 2 10
```

To convert these graph files to the binary format, run the following commands in the root folder:
```
tools/converter path_to_Graph.el
tools/converter path_to_Graph.wel
```

The first command converts Graph.el to the binary CSR format and generates a binary graph file with .bcsr extension under the same directory as the original file. The second command converts Graph.wel to a weighted binary graph file with .bwcsr extension.

#### Running applications in Subway
The applications take a graph as input as well as some optional arguments. For example:

```
$ ./sssp-async --input path-to-input-graph
$ ./sssp-async --input path-to-input-graph --source 10
```

For applications that run on weighted graphs, like SSSP, the input must be weighted (.bwcsr or .wel) and for applications that run on unweighted graphs, like BFS, the input must be unweighted (.bcsr or .el).

#### Publications:

[EUROSYS'20] Amir Hossein Nodehi Sabet, Zhijia Zhao, and Rajiv Gupta. [Subway: minimizing data transfer during out-of-GPU-memory graph processing](https://dl.acm.org/doi/abs/10.1145/3342195.3387537). In Proceedings of the Fifteenth European Conference on Computer Systems.

[ASPLOS'18] Amir Hossein Nodehi Sabet, Junqiao Qiu, and Zhijia Zhao. [Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing](https://dl.acm.org/doi/10.1145/3173162.3173180). In Proceedings of the Twenty-Third International Conference on Architectural Support for Programming Languages and Operating Systems.


