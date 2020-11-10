/*
 * This is a CUDA version of bellman_ford algorithm
 */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
int N; //number of vertices
int *mat; // the adjacency matrix

void abort_with_error_message(string msg) {
	std::cerr << msg << endl;
	abort();
}

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) {
	return x * n + y;
}

int read_file(string filename) {
	std::ifstream inputf(filename, std::ifstream::in);
	if (!inputf.good()) {
		abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
	}
	inputf >> N;
  // Assume a N by N matrix
	mat = (int *) malloc(N * N * sizeof(int));
  // need to change this to COO format
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
      //save value into mat
			inputf >> mat[convert_dimension_2D_1D(i, j, N)];
		}
	return 0;
}

int print_result(bool has_negative_cycle, int *dist) {
	std::ofstream outputf("output.txt", std::ofstream::out);
	if (!has_negative_cycle) {
		for (int i = 0; i < N; i++) {
			if (dist[i] > INF)
				dist[i] = INF;
			outputf << dist[i] << '\n';
		}
		outputf.flush();
	} else {
		outputf << "FOUND NEGATIVE CYCLE!" << endl;
	}
	outputf.close();
	return 0;
}
}


__global__ void bellman_ford_one_iter(int n, int *d_mat, int *d_dist, bool *d_has_next, int iter_num){
	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int elementSkip = blockDim.x * gridDim.x;

	if(global_tid >= n) return;

	for(int u = 0 ; u < n ; u ++){
		for(int v = global_tid; v < n; v+= elementSkip){
			int weight = d_mat[u * n + v];
			if(weight < INF){
				int new_dist = d_dist[u] + weight;
				if(new_dist < d_dist[v]){
					d_dist[v] = new_dist;
					*d_has_next = true;
				}
			}
		}
	}

}

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
	dim3 blocks(blocksPerGrid);
	dim3 threads(threadsPerBlock);

	int iter_num = 0;
	int *d_mat, *d_dist;
	bool *d_has_next, h_has_next;

  // allocate space on GPU
	cudaMalloc(&d_mat, sizeof(int) * n * n);
	cudaMalloc(&d_dist, sizeof(int) *n);
	cudaMalloc(&d_has_next, sizeof(bool));


	*has_negative_cycle = false;
  // set elements to INF in dist array
	for(int i = 0 ; i < n; i ++){
		dist[i] = INF;
	}
  
	dist[0] = 0;
  // copy memory from cpu to gpu
	cudaMemcpy(d_mat, mat, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int) * n, cudaMemcpyHostToDevice);

	for(;;){
		h_has_next = false;
    // copy memory back from gpu to cpu
		cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

		bellman_ford_one_iter<<<blocks, threads>>>(n, d_mat, d_dist, d_has_next, iter_num);
    // copy memory back from gpu to cpu
		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

		iter_num++;
		if(iter_num >= n-1){
			*has_negative_cycle = true;
			break;
		}
		if(!h_has_next){
			break;
		}

	}
	if(! *has_negative_cycle){
		cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
	}
  // free memory on gpu
	cudaFree(d_mat);
	cudaFree(d_dist);
	cudaFree(d_has_next);
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
	}
	if (argc <= 3) {
		utils::abort_with_error_message("blocksPerGrid or threadsPerBlock WAS NOT FOUND!");
	}

	string filename = argv[1];
	int blockPerGrid = atoi(argv[2]);
	int threadsPerBlock = atoi(argv[3]);

	int *dist;
	bool has_negative_cycle = false;

  // read file
	assert(utils::read_file(filename) == 0);
  // malloc and init to zero 
	dist = (int *) calloc(sizeof(int), utils::N);


	//time counter
	timeval start_wall_time_t, end_wall_time_t;
	float ms_wall;
	cudaDeviceReset();
	//start timer
	gettimeofday(&start_wall_time_t, nullptr);
	//bellman-ford algorithm
	bellman_ford(blockPerGrid, threadsPerBlock, utils::N, utils::mat, dist, &has_negative_cycle);
	//end timer
	gettimeofday(&end_wall_time_t, nullptr);
	ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
			+ end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

	std::cerr.setf(std::ios::fixed);
	std::cerr << std::setprecision(6) << "Time(s): " << (ms_wall/1000.0) << endl;
	utils::print_result(has_negative_cycle, dist);
	free(dist);
	free(utils::mat);

	return 0;
}
