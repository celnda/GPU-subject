/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 03.09.2021 (revised)
 Description	: fisrt excercise
				: verification of the CUDA installation
				: err checking, basic querry for GPU info
 ============================================================================
 */
 
#include <stddef.h>
#include <stdio.h>
#include "err_check.h"

/* === The program section === */
__global__ void say_hello()
/*
 * The kernel for print of Hello World message
 * expected to be called with 1D grid
 */
{
	// NOTE - look at the thread identification by the indexes
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	printf("Hello world from thread with index %d! \n", idx);
}

int main()
/*
 * main function executing the kernell call and GPU properties output
 */
{	
	const int thread_count = 16; // number of threads used per block
	const int block_count = 2; // number of blocks used 
	
	// NOTE - basic query for information about the device present
	int available_device; // the device number ! expect only single GPU coprocessor presence to work as intended

	// NOTE - notice the cudaSafeCall wrap around cuda function call - this reports runtime errors during call
	// TASK - have a look at the err_check.h for what cudaSafeCall is
	cudaDeviceProp device_prop; 
	cudaSafeCall( cudaGetDevice(&available_device)); //get the device count
	cudaSafeCall( cudaGetDeviceProperties(&device_prop, available_device)); // get the last device properties
	
	/* Device infromation print section */
	//TASK - improved the print section with regards size recalculation 
	printf("*** Hello world from %s coprocessor ***\n", device_prop.name);
	printf("*** SM: %i, Gmem: %d GB, Smem/B: %d kB, Cmem: %d kB ***\n",(int)device_prop.multiProcessorCount
																	  ,(int)device_prop.totalGlobalMem/2^30
																	  ,(int)device_prop.sharedMemPerBlock/1024
																	  ,(int)device_prop.totalConstMem/1024);
		
	/* Kernell call section */
	printf("\n");
	printf("The hello kernel is invoked with following <<< %d, %d >>> \n", block_count, thread_count);

	say_hello<<<block_count,thread_count>>>();
	
	// NOTE - this is required to wait for print to come from GPU
	// TASK - comment it and execute to observe lack of Hello world
	cudaDeviceSynchronize(); 

	// NOTE - Checking that no error occured during kernel launch
	cudaSafeKernell();
	
	return 0;	
}