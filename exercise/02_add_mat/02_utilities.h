/*
 ============================================================================
 Name			: Parallel programming in CUDA
 Author			: David Celny
 Date			: 03.09.2021 (revised)
 Description	: support code for 02 exercise
				: utilities for matrix generation, display, check
 ============================================================================
 */

#ifndef UTILITIES_02_H_
#define UTILITIES_02_H_

//========== INCLUDE ==========

//========== DECLARE ==========

#define RND_WIDTH 10
#define RND_SHIFT 5
//========== FORWARD ==========
const float cmp_eps = 1e-11;

//========== FUNCTIONALITY ==========
void get_rnd_mat(unsigned int size_x, unsigned int size_y, float *mat);
void display_matrix(unsigned int size_x, unsigned int size_y, float *mat);

int check_result_full(unsigned int size_x, unsigned int size_y, float *mat_host, float*mat_dev, bool output=false);
int check_result_sample(unsigned int size_x, unsigned int size_y, float *mat_host, float*mat_dev, bool output=false);

#endif /* UTILITIES_02_H_ */