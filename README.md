### cuda_bellman_ford
Compile:
```Bash
$ nvcc -std=c++11 -o cuda_bellman_ford cuda_bellman_ford.cu
```
Run:
```Bash
$ ./cuda_bellman_ford <intput file> <number of blocks per grid> <number of threads
per block>
```


## Input and output files

The input file will be in following format:
1. The first line is an integer N, the number of vertices in the input graph.
2. The following lines are an N*N adjacency matrix mat, one line per row. The entry in row v and column w, mat[v][w], is the distance (weight) from vertex v to vertex w. All distances are integers. If there is no edge joining vertex v and w, mat[v][w] will be 1000000 to represent infinity.

The vertex labels are non-negative, consecutive integers, for an input graph with N
vertices, the vertices will be labeled by 0, 1, 2, …, N-1. We always use vertex 0
as the source vertex.

The output file consists the distances from vertex 0 to all vertices, in the increasing order of the vertex label (vertex 0, 1, 2, … and so on), one distance per line. If there are at least one negative cycle (the sum of the weights of the cycle is
negative in the graph), the program will set variable has_negative_cycle to true and print "FOUND NEGATIVE CYCLE!" as there will be no shortest path.


