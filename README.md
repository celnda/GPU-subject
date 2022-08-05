# GPU-subject
> “Courage will now be your best defence against the storm that is at hand-—that and such hope as I bring.” Gandalf the White

Exercises to the GPU programming course. Here you can practise what you have heard about during the lectures. The codes are hevily commented with points of interest and aditional information. Guide for the code is given in [general_guide](exercise/general_guide.md). 

The exercises will be condensed into 5 sessions:
* [hello world](exercise/01_hello_world/hello_world_readme.md) - test the instalation setup
	* introduce error checking, device querry 
* [add mat](exercise/02_add_mat/add_mat_readme.md) - perform elementwise sum of two vectors and matrices
	* introduce basic memory operation on GPU, kernell launch with 2D grid, timing of GPU code
* [transpose mat](exercise/03_tanspose_mat/transpose_mat_readme.md) - perform matrix transposition using different types of memory
	* introduce memory operation with shared memory, show how texture memory is used,  example of Instruction Level Paralelism