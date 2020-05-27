## Subway
Subway is an Out-of-GPU-Memory Graph Processing frameworks for GPU platforms.
Subway provides a highly cost-effective solution to extracting a subgraph that only consists of the edges of active vertices. As a consequence, the volume of data transfer between CPU and GPU is dramatically reduced. The benefits from data transfer reduction outweigh the costs of subgraph generation in (almost) all iterations of graph processing, bringing in substantial overall performance improvements.

#### Compilation

To compile Subway, just run make in the root directory. The only requrements are g++ and CUDA toolkit.

#### Input graph formats

Subway accepts edge-list (.el) and weighted edge-list (.wel) graph formats, as well as the binary serialized pre-built CSR graph representation (.bcsr and .bwcsr). It is extremely recommended to convert edge-list format grph files to binary format (using tools/converter) and run the applications on the binary files. Reading binary formats is faster and space efficient.

Subway is sensitive to graph file extension. A weighted edge-list graph file has to finish with .wel characters. The followings are two graph file examples.

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

To convert these graph files to binary format, run the following commands in the root:
```
tools/converter path_to_Graph.el
tools/converter path_to_Graph.wel
```

The first command converts Graph.el to the binary CSR format and makes the binary graph file with .bcsr extension into the same directory as the original file. The second command converts Graph.wel and makes the weighted binary graph file with .bwcsr extension.

#### Running applications in Subway
The applications take the input graph as input as well as some optional arguments. For example:

```
$ ./sssp-async --input path-to-input-graph
$ ./sssp-async --input path-to-input-graph --source 10
```

For applications which run on the weighted graphs, like sssp, the input must be weighted (.bwcsr or .wel) and for applications which run on the unweighted graphs, like bfs, the input must be unweighted (.bcsr or .el).

#### Publications:

[EUROSYS'20] Amir Hossein Nodehi Sabet, Zhijia Zhao, and Rajiv Gupta. Subway: minimizing data transfer during out-of-GPU-memory graph processing. In Proceedings of the Fifteenth European Conference on Computer Systems.

[ASPLOS'18] Amir Hossein Nodehi Sabet, Junqiao Qiu, and Zhijia Zhao. Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing. In Proceedings of the Twenty-Third International Conference on Architectural Support for Programming Languages and Operating Systems.


