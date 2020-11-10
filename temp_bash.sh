nvcc -std=c++11 -o cuda_bellman_ford cuda_bellman_ford.cu
./cuda_bellman_ford test.txt 10 10
