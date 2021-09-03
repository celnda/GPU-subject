/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 03.09.2021 (revised)
 Description	: support code
				: error checking capability
				: alternative to Nvidia error checking 
				: more info at https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
 ============================================================================
 */

#ifndef ERR_CHECK_H_
#define ERR_CHECK_H_

//========== INCLUDE ==========
#include <stdio.h>
//========== DECLARE ==========

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

// Define macro function calls for checking
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaSafeKernell()    __cudaCheckError( __FILE__, __LINE__ )

//========== FUNCTIONALITY ==========
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i at %s: %i of type: %s\n",
                 err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i at %s: %i of type: %s\n",
        		err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i with sync at %s: %i of type: %s\n",
                 err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif /* ERR_CHECK_H_ */
