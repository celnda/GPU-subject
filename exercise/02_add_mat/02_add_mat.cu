/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 03.09.2021 (revised)
 Description	: second excercise
 				: allocation, transfer & deallocation of device resources
				: add two matrix together 
				: host & device function
				: timing methods
 ============================================================================
 */


#include <stddef.h>
#include <stdio.h>
#include "cuda_runtime_api.h"
#include "err_check.h"
#include "02_utilities.h"

/* === The program section === */

/* == the host code == */
__host__ __device__ void element_add(unsigned int ind,float *matA, float *matB, float *matC)
/*
 * the device and host function for elementwise addition into matrixes
 */
{
	matC[ind] = matA[ind] + matB[ind]; // the very addition
}

void matrix_add_c (unsigned int size_x, unsigned int size_y, float *matA, float *matB, float *matC)
/*
 * the CPU implementation of matrix addition
 * the matrixes are in flattened form -> index as vectors
 *   								  -> beware row/coll order
 */
{
	//TODO - implement
	//TASK - write CPU implementation of matrix elementwise addition
	//     - use the prepared element_add() function	

}

/* == the device code == */
__global__ void matrix_add_g (unsigned int size_x, unsigned int size_y, float *matA, float *matB, float *matC)
/*
 * The kernel for addition of two matrixes
 * ! elementwise addition with size input size_x, size_y
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> index as vectors
 *   									-> beware row/coll order
 */
{
	//TODO - implement
	//TASK - transform the CPU into GPU implementation
	//     - use the prepared element_add() function		
	//     - do not forget the flatened format used here

}

int main()
/*
 * main function executing the kernel call 
 */
{
	/* Parameter section */
	const unsigned int size_x = 1;
	const unsigned int size_y = 10;
	size_t size_n = size_x*size_y*sizeof(float);

	const long int seed = 123456789; // to get same random matrix - good for debugging 

	unsigned int thread_count_x = 32; // number of threads used per block in x
	unsigned int thread_count_y = 32; // number of threads used per block in y
	
	int result_of_check; //for number of wrong calculation in comparison
	
	// host variables
	float *matA = NULL;
	float *matB = NULL;
	float *matC = NULL;
	float *matC_dev;

	//NOTE - observe the notation prefix d_ used for device variables
	//NOTE - device global memory variables are always pointers 
	// device variables
	float *d_matA;
	float *d_matB;
	float *d_matC;

	// timing
	// NOTE - notice two types of timing CPU specific time
	clock_t cpu_start, cpu_stop;
	// NOTE - notice GPU event base timing 
	//      - standard time counts the kernell launch and response time on top
	cudaEvent_t gpu_start, gpu_stop;
	float cpu_time, gpu_time;

	/* Allocation section */
	matA = (float*) malloc(size_n);
	matB = (float*) malloc(size_n);
	matC = (float*) malloc(size_n);
	matC_dev = (float*) malloc(size_n);


	/* Initialisation section */
	srand(seed); // initialize random generator with seed
	get_rnd_mat(size_x,size_y,matA);
//	display_matrix(size_x, size_y, matA); //DEBUG
	get_rnd_mat(size_x,size_y,matB);
//	display_matrix(size_x, size_y, matB); //DEBUG

	// the device memory allocation
	//NOTE - observe the format of device allocation
	//     - in particular the generic void pointer to where the memory should be allocated
	//     - this is consequence of memory being on the device
	cudaSafeCall(cudaMalloc((void**)&d_matA, size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matC, size_n));

	// the data move host->device
	//NOTE - matrices have to be moved to device
	//     - notice the format: TARGET, SOURCE, SIZE, DIRECTION(type of transfer)
	//     - direction is cuda keyword - for more info consult documentation 
	cudaSafeCall(cudaMemcpy(d_matA, matA, size_n, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_matB, matB, size_n, cudaMemcpyHostToDevice));
	
	// create timing events
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));

	/* Kernel call section */
	//NOTE - creation of the 2D grid for kernell call
	//     - 2D tread block correspond to 2D grid
	//TASK - understand what what actual size we are requesting
	//     - does it overflow ?
	//     - does it require precise multiple of thread count?
	dim3 thread_dim(thread_count_x,thread_count_y);
	dim3 block_dim(ceil((float)size_x/thread_count_x),
		           ceil((float)size_y/thread_count_y));

	cudaSafeCall(cudaEventRecord(gpu_start, 0));

	matrix_add_g<<<block_dim,thread_dim>>>(size_x, size_y, d_matA, d_matB, d_matC);
	
	cudaSafeCall(cudaEventRecord(gpu_stop, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop)); //transfer of the the timing data
	cudaSafeKernell(); // for synchronization not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time,gpu_start,gpu_stop));
	
	// the data move device->host
	//TASK - write copy of the result memory
	

	// event cleaup
	//NOTE - GPU events have to be destoyed to prevent GPU memory leaks
	cudaSafeCall(cudaEventDestroy(gpu_start));
	cudaSafeCall(cudaEventDestroy(gpu_stop));

	/* CPU section */
	cpu_start = clock();
	matrix_add_c(size_x, size_y, matA, matB, matC);
	cpu_stop = clock();

	cpu_time = 1000*(cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC); //convert to ms
	/* execution statistics section */
	printf("*** level of parallelization ***\n");
	printf("*** matrix: %d,%d ***\n", size_x, size_y);
	printf("*** block: %d,%d ***\n", block_dim.x, block_dim.y);
	printf("*** thread: %d,%d ***\n", thread_dim.x, thread_dim.y);
	printf("*** time measurement ***\n");
	printf("*** CPU: %f ms\n",cpu_time);
	printf("*** GPU: %f ms\n",gpu_time);
	printf("*** speedup: %f \n",cpu_time/gpu_time); 

	/* control section */
	// display_matrix(size_x, size_y, matC); //DEBUG
	// display_matrix(size_x, size_y, matC_dev); //DEBUG
	
	//TASK - for full check of smaller matrices use the full variant
	// result_of_check = check_result_full(size_x, size_y, matC, matC_dev, true);
	// printf("%d\n", result_of_check );
	
	result_of_check = check_result_sample(size_x, size_y, matC, matC_dev, true);
	printf("%d\n", result_of_check );

	/* Cleanup section */
	//BUG - deallocate all memory (CPU,GPU)
	free(matA);
	free(matB);
	free(matC);
	free(matC_dev);
	//NOTE - the memory allocated with cudaMalloc has to be freed with cudaFree
	//TASK review what all has to be deallocated, is it all ?
	cudaSafeCall(cudaFree(d_matA));
	cudaSafeCall(cudaFree(d_matB));
	cudaSafeCall(cudaFree(d_matC));

	return result_of_check;
}

