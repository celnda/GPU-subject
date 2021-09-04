# Add mat guide
The program is intended to show more elaborate operation with GPU. On the simple problem of elementwise addition of matrixes the basic concepts of memory allocation, deallocation and transfer are shown. The program is supplemented by utilities to ease matrix generation and consequent verification. The task is completed with example of timing the perfomance of CPU and GPU implementation comparison.

## Memory allocation
Beware this section is very important. Memory management is fundamental knowledge and also knowledge where most of the mistakes in C are made. With new memory space on GPU the amount of memory management is doubled. Therefore several guidelines are to be followed for sucessful compilation and execution of the code. 
* section the code on CPU and GPU parts (smaller project with sections, larger in files)
* when allocating CPU part allocate GPU part as well to see if anything is missing
* immediately after allocation write deallocation and keep both updated
* all device variables are subsripted **d_** not to be dereferenced from GPU
	* this would return invalid pointer as such a operation is not permitted in this kontext
* have the GPU allocated sizes meaningful and prepared - they are used in memory transfers
	* prevents mistakes when only part of result array is transfered due to the misleading names

## Kernell launch parameters
From the previous exercise this time 2D kernell is being called. Notice the subtle differences here as now dim3(?,?) is used for the construction of 2D launch parameters. Notice that both thread block and grid use the same way.
Corresponding can be also done for 1D and of course for 3D case.
* beware what grid is being used with what kernell
	* notice the comment stating with which grid the kernel expect to be launched (1D,2D,3D)
* beware of the size of the problem and launch bounds in kernel
	* overflow threads has to be handled accordingly
	* prevent mistakes from unwanted data race and mishandling of data

## Timing
When it comes to GPU programming benchmarking becomes integral part of the developement. Because performance is the ultimate motivation for the GPU programing benchmarking is performed earlier than usual. When it comes to simple type of benchmark one has to be aware of the role hardware plays in it. Due to the changes to the hardware by manufacturers it is not immediately certain how the specific program will work on different hardware. 
More rigorous aproach include profiling the aplication which in essence time portion/whole program multiple times and output the gathered statistics. For GPU additional data are often queried about bandwith and other memory related features.
In presented example simple one pass timing is performed to illustrate the concept. Notice the duality of timing shown here. CPU time is obtained throughout the *clock()* functionality while gpu time rely on inbuilt events. Use of the events is not the only way how to measure time on GPU and clock can be use to some extent as well. Look on the internet for the alternatves.

## Try out
During this task you will be introduced into multiple of core concept of GPU programming. It is therefore recomended to spend your time looking around and trying different configurations and try to understand underlying behaviour. Because of this not much code needs to be writen. 
1. **familiarize** yourself with the code structure
	* be able to navigate yourself in the code section and understand what the sections do
	* look at the used utilities and understand how they work
	* go through the tasks and see if you can answer them
1. look at the cuda specifics calls - in particular **allocation data move, timing and deallocation**
1. fill the CPU and GPU elementwise adition functionality
	* into prepared functions
	* using prepared *element_add()* function 
1. do not forget to write **copy back** of the GPU result
1. **measure the times for diferent size of matrixes used**
	1. compare CPU, GPU times on entities with 2^5 ... 2^15 elements
	1. make a graf of both timings over increasing matrix size
	1. at what size does the GPU overtakes the CPU ?
		* you can change the compared sizes and comment on the trends you can see

Based on the lectures what are your recomendation for improving the performance on GPU?