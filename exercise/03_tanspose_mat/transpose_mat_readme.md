# Transpose mat guide
The program is intended to show multiple ways of optimization on GPU. On the problem of matrix transpose the example of blocked,shared memory and texture memory usage are illustrated. The program is supplemented by utilities to ease matrix generation, visualization and consequent verification. The task is completed with example of timing the perfomance of CPU and GPU implementation comparison and additionall bandwidth utilization.

## Higher level of abstraction
In CUDA C level programming it is often the case that level of abstraction is not enough for desired purpose of speed. It is the issue that lot is unknown during compile time but the efficitient execution demands knowledge of such situation. The solution to this issue rather than dynamic detection of inputed values is the use of macros. Macros are an effective/dangerous tool to influence generation of code that is later compiled. In this example you will see usage of macros to bypass the need for conditions within the computational code. Be aware that often neglected value of macro can lead to disastrous errors and long debugging sessions. It is recommended to verify the expected situation with the code generated after macros are processed. 

## Shared memory
For the purpose of coalesced memory writes and reads even for tasks that do not inherently have this pattern one can use shared memory. Shared memory in general enables communication of threads within block. As such it is prone to data race condition and require carefull handling. For access pattern be aware that data banks serialize accesses. This memory is therefore most efficient when each thread access different data bank. It is not mandatory for the accesses to be coalesced in this case. Because of common occurence of broadcast pattern shared memory also offers fast distribution of memory to all treads. 
* this type of memory is stil about order of magnitude slower as registers
	* synchronization barriers or thread fences add to the slowdown
	* atomic operation can be employed in same way as for global memory
* allocation/deallocation of shared memory is handled in third parameter in kernell launch
	* be aware of the limit of shared memory **requested** for the card
		* it can be overriden by the value required in kernell

## Texture memory
Steming from the original purpose of GPU texture memory cater to the task of manipulating pixel data. In cuda it is also posible to leverage the properties of texture memory for computation purposes of non pixel type values. In such case additional programator care has to be given for texture memory management. From hardware perspective texture memory has separate prefetcher and can contribute to bandwith saturation. Additionally texture memory has smaller cacheline width of 48b. This is the reason for utilization of texture memory for tasks with scatter memory access that cant be leveraged by shared memory usage. In case of texture interesting option is implicit interpolation and texture boundary operation. These option often do not hold much use for calculation and can be found in documentation if the need arise.

## ILP
Advanced optimization concept that capture both badwith utilization and computation strain of units is expressed in instruction level parallelism. ILP for short shift the balance between the two by using the computational units over multiple elements of the computational grid. Doing so can help for large enough tasks by freeing the computational power.
As explained at the lecture this can only work until bandwith is saturated and there is clear cut limit how far ILP can help. As such ILP is a technique employed further in the optimization only if the situation fullfils the condition of large task the is computationally underperfomed. In this way we can shift the balance of memory and computation to order of about 100 operations per memory load.

## Try out
During this task you will be introduced into multiple of core concept of GPU programming. It is therefore recomended to spend your time looking around and trying different configurations and try to understand underlying behaviour. Testing the different optimization cases is essential for understanding.
1. **familiarize** yourself with the code structure
	* be able to navigate yourself in the code section and understand what the sections do
	* look at the used utilities and understand how they work
	* go through the tasks and see if you can answer them
1. look at the cuda specifics calls - in particular **new types of memory management, macro use**
1. fill the CPU and GPU elementwise adition functionality
	* into prepared functions
	* using prepared *element_add()* function 
1. do not forget to write **copy back** of the GPU result
1. **measure the times for diferent size of matrixes used for the implemented transpose algorithm variants**
	1. compare CPU, GPU times for the transpose variants with 2^5 ... 2^15 elements
	1. make a graf of both timings over increasing matrix size
	1. at what size does the GPU overtakes the CPU ?
		* you can change the compared sizes and comment on the trends you can see

Based on the lectures and your current experience what are your recomendation for improving the performance on GPU?