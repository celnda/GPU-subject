	/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 06.09.2021 (revised)
 Description	: third excercise
 				: transposition of matrix - Out of place
				: bandwidth measurement
				: 
 ============================================================================
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"
#include "err_check.h"
#include "03_utilities.h"

#define CUDA_ERROR_CHECK

// the controll of implementation technique
#define NAIVE 0
#define BLOCK 1
#define SHARED 2
#define CACHED 3

#ifndef TRAN_VER
#define TRAN_VER NAIVE // <- modify this to change operation
#endif
// TILE_SIZE is only valid for tiled
#define TILE_SIZE 16 //  <- modify this to change tiled approach size
#define ILP 4   // number of the work ech thread calculate

// global parameter for tran_ver reporting
const char* tran_ver_name[] = {"naive", "block", "shared", "cached"};

// constant memory declaration
__constant__ unsigned int d_A_size_x;
__constant__ unsigned int d_A_size_y;

/* === The program section === */
/* == the host code == */
void matrix_transpose_c (unsigned int A_size_x, unsigned int A_size_y, float *matA, float *matB)
/*
 * the CPU implementation of matrix transposition
 * the matrixes are in flattened form -> index as vectors
 *   								  -> beware row/coll order
 */
{
	unsigned int i,j;

	for (i = 0; i < A_size_x; i++)
	{ // walk through rows of A matrix
		for (j = 0; j < A_size_y; j++)
		{ // walk through collumns of A matrix
			matB[j*A_size_x + i] = matA[i*A_size_y + j]; 
		}
	}
}
/* == the device code == */

__device__ void submat_transpose_g_naive(unsigned int id_i,unsigned int id_j,float *matA, float *matB)
/*
 * the device function for result elemnet matrix transposition
 * expected the 2d call grid
 * general version with most uncoalesced( esp. matB writes) memory pattern 
 */
{	
	//TODO - implement the naive transpose matrix obeying the function header and description

}

__device__ void submat_transpose_g_block(unsigned int id_i,unsigned int id_j,float *matA, float *matB)
/*
 * the device function for result elemnet matrix transposition
 * expected the 2d call grid with the y dimension is TILE_SIZE times smaller
 * tiled schema -> more execution per thread (in collumn direction)
 * general version with most uncoalesced( esp. matB writes) memory pattern 
 */
{	
	//TODO - implement the tiled transpose matrix obeying the function header and description
	//     - do not forget to unroll the loops

}

__device__ void submat_transpose_g_shared(unsigned int id_i, unsigned int id_j, float *tmp_A_submat,float *matA, float *matB)
/*
 * the device function for result block matrix transposition
 * expected the 2d thread grid
 * temporary storage for coalesced reads
 */
{		
	//TODO - implement the shared version of transpose matrix obeying the function header and description
	//	   - do not forget to properly synchronize
	//     - utilize the loading transpose into shared memory for coalescing the writes

}

#if TRAN_VER == CACHED

texture<float, cudaTextureType1D, cudaReadModeElementType> texRef;

__device__ void submat_transpose_g_cached(unsigned int id_i, unsigned int id_j, float *tmp_A_submat, float *matB)
/*
 * the device function for result block matrix transposition
 * expected the 2d thread grid
 * reading is cached from texture memory
 * temporary storage for coalesced reads
 */
{	
	//TODO - modify previous shared to the cached version of transpose matrix obeying the function header and description
	//     - consult cuda for propper operation with texture memory in texRef defined above

}
#endif

__global__ void matrix_transpose_g (float *matA, float *matB)
/*
 * The kernel for matrix matrix multiplication
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
#if TRAN_VER == NAIVE
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	
#elif TRAN_VER == BLOCK
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	id_y *= ILP; // to account for the ILP of each thread
#elif TRAN_VER == SHARED
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
#elif TRAN_VER == CACHED
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
	
#endif
	//TASK - try the padded/nonpadded version and notice the changes required in this kernell
	//TODO - implement these required changes      

#if TRAN_VER == NAIVE
	submat_transpose_g_naive (id_x, id_y, matA, matB);
#elif TRAN_VER == BLOCK
	submat_transpose_g_block (id_x, id_y, matA, matB);
#elif TRAN_VER == SHARED
	submat_transpose_g_shared(id_x, id_y, tmp_A_submat, matA, matB);
#elif TRAN_VER == CACHED
	submat_transpose_g_cached(id_x, id_y, tmp_A_submat, matB);
#endif
}


__global__ void matrix_copy_g (float *matA, float *matB)
/*
 * The kernel for pure matrix copy
 * 	 expected to be called as two dimensional grid
 *	 used for effective bandwidth measurement
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
	unsigned int id_i =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_j =  blockDim.y * blockIdx.y + threadIdx.y; 
	
	matB[id_i*d_A_size_y + id_j] = matA[id_i*d_A_size_y + id_j];
}

/* == the main call code == */

int main( int argc, char *argv[] )
/*
 * main function executing the kernell call and GPU properties output
 */
{	
	//NOTE - rudimentary input argument parsing to handle different type of matrices
	unsigned int tmp_x, tmp_y;
	if (argc == 1) // DEFAULT CASE
	{
		tmp_x = 16;
		tmp_y = 4;
	}
	
	if (argc == 2) // SQUARE CASE
	{
		tmp_x = strtol(argv[1], NULL, 10);
		tmp_y = strtol(argv[1], NULL, 10);
	}
	
	if (argc == 3) //RECTANGLE CASE
	{
		tmp_x = strtol(argv[1], NULL, 10);
		tmp_y = strtol(argv[2], NULL, 10);
	}
	else
	{
		printf("Error: wrong number of parameters: %d\n",argc);
		printf("   accepted cases are: 0 = default\n");
		printf("                       1 = square matrix of int('arg1') size\n");
		printf("                       2 = rectangle matrix of int('arg1')*int('arg2') size\n");
		exit(-1);
	}
	
	// parameters section
	const long int seed = 123456789;

	//BUG does not handle negative numbers input
	const unsigned int A_size_x = tmp_x;
	const unsigned int A_size_y = tmp_y;

	const unsigned int B_size_x = A_size_y;
	const unsigned int B_size_y = A_size_x;
		
	const unsigned int A_size_p_x = TILE_SIZE*(int)((A_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int A_size_p_y = TILE_SIZE*(int)((A_size_y + TILE_SIZE -1)/TILE_SIZE);
	
	const unsigned int B_size_p_x = TILE_SIZE*(int)((B_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int B_size_p_y = TILE_SIZE*(int)((B_size_y + TILE_SIZE -1)/TILE_SIZE);
	
	size_t A_size_n = A_size_p_x*A_size_p_y*sizeof(float);
	size_t B_size_n = B_size_p_x*B_size_p_y*sizeof(float);
		
	// host variables
	float *matA = NULL;
	float *matB = NULL;
	float *matB_dev;
	// device variables
	float *d_matA;
	float *d_matB;
	float *d_matB_b;
	
	// timing
	clock_t cpu_start, cpu_stop;
	cudaEvent_t gpu_start, gpu_start_b, gpu_stop, gpu_stop_b;
	float cpu_time, gpu_time, gpu_time_b;

	/* initialization section */	
	srand(seed);
	
	// init the rest: matB, matB_dev
	matA = (float*) malloc(A_size_n);
	matB = (float*) malloc(B_size_n);
	matB_dev = (float*) malloc(B_size_n);

	//TASK - study the display_matrix function to see how the data are shown
	//       is it row-wise or column-wise

	//TASK - try the padded and non-padded version and see the differences 
// 	get_rnd_mat   (A_size_x, A_size_y, matA);	
// 	get_zero_mat  (B_size_x, B_size_y, matB);
	
	get_rnd_mat_padd (A_size_x, A_size_y, A_size_p_x, A_size_p_y, matA);	
	get_zero_mat     (B_size_p_x, B_size_p_y, matB);	

// 	display_matrix (A_size_p_x, A_size_p_y, matA);
// 	display_matrix (B_size_p_x, B_size_p_y, matB);

	// the device init
	cudaSafeCall(cudaMalloc((void**)&d_matA, A_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, B_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB_b, A_size_n)); // for the bandwidth check
	
	// constant memory move
	//NOTE - see the constant memory utilization here
	//     - notice the similarities to memcpy below 	
	//     - notice where and how the constant memory is declared
	//     - beware that the source location has to be pointer
	//     - realize that GPU constant does not need to be compile constant
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_x, &A_size_p_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_y, &A_size_p_y, sizeof(unsigned int)));
	
	// the data move host->device
	cudaSafeCall(cudaMemcpy(d_matA, matA, A_size_n, cudaMemcpyHostToDevice));
	
	// cached variant initialisation
#if TRAN_VER == CACHED
	//TASK - look up what cudaCreateChannelDesc does 
	//     - think how this could help in this context
	
	// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* cuArray;
	cudaMalloc(&cuArray, A_size_n);

	cudaMemcpy(cuArray, matA, A_size_n, cudaMemcpyHostToDevice);
	//TASK - with which memory we are operating precisely
	//     - study the usage of this type of memory
	//     - look at the benefits from already open cuda documentatiton
	//     - notice the differences between this and global memory
	texRef.addressMode[0]	= cudaAddressModeClamp;
	texRef.addressMode[1]	= cudaAddressModeClamp;
	texRef.filterMode		= cudaFilterModePoint;
	texRef.normalized		= false;
	
	size_t offset = 0;
	//TASK - what does this operation ensure
	//     - is it mandatory ?
	cudaBindTexture(&offset, texRef, cuArray, A_size_n);	
	
#endif
	
	// timing
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));
	
	cudaSafeCall(cudaEventCreate(&gpu_start_b));
	cudaSafeCall(cudaEventCreate(&gpu_stop_b));
	
	/* Kernell call section */
	//NOTE - see the difference in the problem division into threadblocks/grid
	//     - be able to explain why BLOCK case is separated
#if TRAN_VER == NAIVE || TRAN_VER == SHARED || TRAN_VER == CACHED
	dim3 thread_dim(TILE_SIZE,TILE_SIZE);
	dim3 block_dim(A_size_p_x/TILE_SIZE, A_size_p_y/TILE_SIZE);
#else  
	//TASK - which dimension is being targeted by ILP and why
	//     - notice the dependency created between the call and the implementation
	dim3 thread_dim(TILE_SIZE,TILE_SIZE);	
	dim3 block_dim(A_size_p_x/TILE_SIZE, ceil((float)A_size_p_y/TILE_SIZE/ILP)); 
#endif
		
	/* to get the device initialisation in case that is not measured*/
	matrix_copy_g<<<block_dim,thread_dim>>>(d_matA, d_matB_b);
	cudaDeviceSynchronize();
	
	/* the copy kernel */
	//NOTE - this is illustation of the base bandwidth benchmark shown here
	//     - baseline is created by pure copying of data from one place in global memory to other
	//TASK - predict if the transpose versions will have worse/same/better bandwidth
	//     - compare these predictions accross the implemented cases
	cudaSafeCall(cudaEventRecord(gpu_start_b, 0));
	
	matrix_copy_g<<<block_dim,thread_dim>>>(d_matA, d_matB_b);

	//TASK - consult the CUDA documentation if the synchronization is needed here or it is just slowdown
	// cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop_b, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop_b));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time_b,gpu_start_b,gpu_stop_b));
	
	/* the transpose kernel */
	cudaSafeCall(cudaEventRecord(gpu_start, 0));
	
#if TRAN_VER == NAIVE || TRAN_VER == SHARED || TRAN_VER == CACHED || TRAN_VER == BLOCK
	matrix_transpose_g<<<block_dim,thread_dim>>>(d_matA, d_matB);
#else
	printf(" !!! Unsuported version option: %d !!! \n",TRAN_VER);
#endif

	//TASK - consult the CUDA documentation if the synchronization is needed here or it is just slowdown
	// cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time,gpu_start,gpu_stop));

	// the data move device->host
	cudaSafeCall(cudaMemcpy(matB_dev, d_matB, B_size_n, cudaMemcpyDeviceToHost));

	// event cleaup
	cudaSafeCall(cudaEventDestroy(gpu_start));
	cudaSafeCall(cudaEventDestroy(gpu_start_b));
	cudaSafeCall(cudaEventDestroy(gpu_stop));
	cudaSafeCall(cudaEventDestroy(gpu_stop_b));
	
	/* CPU section */
	cpu_start = clock();

	matrix_transpose_c(A_size_p_x, A_size_p_y, matA, matB);

	cpu_stop = clock();

	cpu_time = 1000*(cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC); // in ms

	/* execution statistics section */
	printf("*** COMPILED VERSION: %s\n",tran_ver_name[TRAN_VER]);
	printf("*** level of parallelization ***\n");
	printf("*** matrix: %d,%d ***\n", A_size_x, A_size_y);
	printf("*** padded: %d,%d ***\n", A_size_p_x, A_size_p_y);
	printf("*** block: %d,%d ***\n", block_dim.x, block_dim.y);
	printf("*** thread: %d,%d ***\n", thread_dim.x, thread_dim.y);
	
	printf("*** time measurement ***\n");
	printf("*** CPU: %f ms\n",cpu_time);
	printf("*** GPU_bench: %f ms\n",gpu_time_b);
	printf("*** GPU: %f ms\n",gpu_time);	
	printf("*** speedup: %f \n",cpu_time/gpu_time);
	//TASK - devise the conversion from output value to desired GB
	printf("*** bandwidth_bench: %f GB\n",2*A_size_n/gpu_time_b/ ??? ); 
	printf("*** bandwidth: %f GB\n",2*A_size_n/gpu_time/ ??? ); 
	//BONUS TASK - implement comparison with maximally achevable bandwidth for used Hardware
	//NOTE - look which properties can be used as shown in first task about hello world
	//TASK - think about why badwith benchmark is used as oposed to the theoretical value.

	/* controll section */
	// const char* str_format = "%16.12f";
	const char* str_format = "%f, ";

	// display_matrix(A_size_p_x, A_size_p_y, matA, str_format);
	// display_matrix(B_size_p_x, B_size_p_y, matB, str_format);
	// display_matrix(B_size_p_x, B_size_p_y, matB_dev, str_format);	

	int result;
	// result= check_result(B_size_x, B_size_y, matB, matB_dev, false); // sampling
	// result= check_result_full(B_size_x, B_size_y, matB, matB_dev, true); // full check for non padded matrix
	result= check_result_full_padd(B_size_x, B_size_y, B_size_p_x, B_size_p_y, matB, matB_dev, false); // full check for padded matrix
	
	printf("Mismatching element count %d!\n", result);
	
	/* celanup section */
	// TODO - consult the documentation whether and how texture memory is cleaned up - implement your finding
	free(matA);
	free(matB);
	free(matB_dev);
	cudaFree(d_matA);
	cudaFree(d_matB);
	cudaFree(d_matB_b);

	return result;
}
