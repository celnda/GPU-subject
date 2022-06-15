/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 03.09.2021 (revised)
 Description	: support code for 03 exercise
				: utilities for matrix generation, display, check
 ============================================================================
 */
#include <stdio.h>
#include "utilities.h"

/* = the utilities code = */
void get_zero_mat(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * fill the given array with zeros
 * used for the final multiplication matrix initialisation in case of multiple calculation steps
 */
{
	unsigned int i;
	// check the correctness - should be faster
	// mat = memset(0,size_x*size_y*sizeof(float));
	
	for (i = 0; i < size_x*size_y; i++)
	{ // operate on single row		
		mat[i] = 0.0 ;		
	}
	return;
}

void get_rnd_mat(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * fill the given array with random numbers
 *  random generator spread is set with RAND_WIDTH and RAND_SHIFT
 */
{
	unsigned int i,j;

	for (i = 0; i < size_y; i++)
	{ // operate on single row
		for (j = 0; j < size_x; j++)
		{ // walk through rows
			mat[j + i*size_x] = RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
		}
	}
	return;
}

void get_rnd_mat_padd(unsigned int size_x, unsigned int size_y, unsigned int size_p_x, unsigned int size_p_y, float *mat)
/*
 * fill the given array with random numbers and padd the the rest <size_p_x size_p_y with zeros
 *  random generator spread is set with RAND_WIDTH and RAND_SHIFT
 */
{
	unsigned int i,j;

	for (i = 0; i < size_p_y; i++)
	{ // walk through rows
		if (i < size_y)
		{
			for (j = 0; j < size_p_x; j++)
			{ // walk through collumns
				if (j < size_x)
				{
					mat[j + i*size_p_x] = RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
				}
				else
				{
					mat[j + i*size_p_x] = 0.0;
				}
			}
		}
		else
		{
			for (j = 0; j < size_p_x; j++)
			{
				mat[j + i*size_p_x] = 0.0;
			}
		}
	}
	return;
}

void display_matrix(unsigned int size_x, unsigned int size_y, float *mat, const char* print_format)
/*
 * utility for displaying small matrixes
 * accepts float type format specifier as optional input
 */
{
	unsigned int i,j;
	for (i = 0; i < size_y; i++)
	{ // operate on single row
		printf("( ");
		for (j = 0; j < size_x; j++)
		{ // walk through rows
			printf(print_format, mat[j + i*size_x]);
		}
		printf(")\n");
	}
	printf("\n");
	return;
}

//TODO - implement
// void display_matrix_padd(???)
// {
// 	//TASK - create a function that would only display relevant nonzero element of original matrix
//         - omit display of the padding
//         - use it in main file for more concentrated display
// }

/* = the comparator code = */
int compare_at_index(unsigned int index, unsigned int size_x, unsigned int size_y, float* mat_a, float* mat_b, bool output)
/*
 * perform comparison of two matrix values at index
 * write formated message about the comparison if output=true
 * 											   otherwise only at nonequal values
 */
{
	float cmp;
	
	cmp = abs(mat_a[index] - mat_b[index]); // calculate absolute difference
	
	if (cmp > cmp_eps) // BUG - beware the epsilon differences
	{
		printf("Difference in CPU/GPU comparison at index:%d [x=%d y=%d] \n abs(A[x,y]-B[x,y]) = %16.12f \n", index, index%size_x, index/size_x, cmp );
		return 1;
	}
	else if (output)
	{
		printf("Comparison at [x=%4d, y=%4d] D=%16.12f H=%16.12f \n",index%size_x,index/size_x,mat_a[index], mat_b[index]);
	}
	return 0;
}

int check_result_full(unsigned int size_x, unsigned int size_y, float *mat_host, float *mat_dev, bool output)
/*
 *	perform full check of all indexes of matrix
 *  clear code for teaching purposes not for speed (no vectorization)
 *  for large matrixes this is slow
 */
{
	unsigned int i_x,i_y;
	unsigned int status; // count number of mistakes 

	status = 0;
	printf(" *** Sampling full %d elements from matrix: ***\n", size_x*size_y);
	for (i_y = 0; i_y < size_x*size_y; i_y += size_x)
	{
		for (i_x = 0; i_x < size_x; i_x++)
		{
			status += compare_at_index(i_x+i_y, size_x, size_y, mat_host, mat_dev, output);
		}
		if (output)
		{
			printf(" *** -------------------------------------- ***\n");
		}
	}	
	return status;
}

int check_result_full_padd(unsigned int size_x, unsigned int size_y, unsigned int size_p_x, unsigned int size_p_y, float *mat_host, float*mat_dev, bool output)
/*
 *	perform full check of all indexes of matrix
 *  clear code for teaching purposes not for speed (no vectorization)
 *  works on the padded matrix of size_p_x*size_p_y 
 *  	where numbers expected nonzero for size_x*size_y 
 *  for large matrixes this is slow
 */
{
	unsigned int i_x,i_y;
	unsigned int status; // count number of mistakes 

	status = 0;
	printf(" *** Sampling full %d elements from matrix: ***\n", size_x*size_y);
	//BUG - beware of padding employed for flattened matrix
	//    - the zero region add to the total dimension 
	for (i_y = 0; i_y < size_y; i_y++)
	{
		for (i_x = 0; i_x < size_x; i_x++)
		{
			status += compare_at_index(i_x+i_y*size_p_x, size_p_x, size_p_y, mat_host, mat_dev, output);
		}
		if (output)
		{
			printf(" *** -------------------------------------- ***\n");
		}
	}	
	return status;
	
}

int check_result_sample(unsigned int size_x, unsigned int size_y, float *mat_host, float *mat_dev, bool output)
/*
 * the verification function for result checking
 * 	samples both matrixes no matter how big
 * 			-> the problematic corner values are definitely sampled
 * 			-> the middle part is partially sampled (proportionally to size)
 * 	! beware oversampling for smaller matrixes
 * 	default output is used for printing
 */
{
	unsigned int i, tmp_idx, status;
	const unsigned int idx_max_cnt = 8+log2(1.0*size_x*size_y); //sampling quantity
	unsigned int sample_idx[idx_max_cnt]; // the sampled indexes
	int corners, middles;

	/* sample the corner and sides in middle*/
	i = 0; 
	sample_idx[i++] = 0; //top left corner always present
	
	//TASK - understand how the corners are calculated in the flattend matrix format	
	if (size_x == 1)
	{
		if (size_y ==1 ) // 1*1 scalar
		{ 
			corners = 1;
			middles = 0;
		}
		else // 1*y vector
		{
			sample_idx[i++] = size_x*size_y - 1; 
			corners = 2;

			sample_idx[i++] = size_y/2;
			middles = 1;
		}
	}
	else
	{
		if (size_y ==1 ) //x*1 vector
		{
			sample_idx[i++] = size_x*size_y - 1; 
			corners = 2;

			sample_idx[i++] = size_x/2;
			middles = 1;
		}
		else // x*y matrix
		{
			sample_idx[i++] = size_x - 1;
			sample_idx[i++] = (size_y - 1)*size_x;
			sample_idx[i++] = size_x*size_y - 1; 
			corners = 4;

			sample_idx[4] = size_x/2;
			sample_idx[5] = size_y/2 *size_x;
			sample_idx[6] = size_y/2*(size_x+1);
			sample_idx[7] = size_x*size_y - size_x/2;
			middles = 4;
		}
	}	
	
	/* sample inner portion of matrix */
	// if (size_x>2 && size_y>2)
	// {
		for (; i < idx_max_cnt; i++)
		{
			tmp_idx = (int)((size_x-1)*(rand()/(float)(RAND_MAX)));			// random x index of size_x
			tmp_idx += (int)((size_y-1)*(rand()/(float)(RAND_MAX)))*size_x; // and random y index of size_y
			sample_idx[i] = tmp_idx;
		}
	// }

	/* print header and perform the comparison*/

	if (output) printf(" *** Sampling %d elements from matrix: ***\n     %d corners, %d middle sides, %d insides\n", idx_max_cnt, corners, middles, idx_max_cnt - corners - middles);
	
	status = 0;
	for (i = 0; i < corners; i++)
	{
		status += compare_at_index(sample_idx[i], size_x, size_y, mat_host, mat_dev, output);
	}
	if (output) printf(" *** --------------------------------- ***\n");

	for (; i < corners+middles; i++)
	{
		status += compare_at_index(sample_idx[i], size_x, size_y, mat_host, mat_dev, output);
	}
	if (output) printf(" *** --------------------------------- ***\n");

	for (; i < idx_max_cnt; i++)
	{
		status += compare_at_index(sample_idx[i], size_x, size_y, mat_host, mat_dev, output);
	}
	return status;
}
