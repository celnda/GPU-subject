# Hello world guide
The program is intended to test that cuda has been sucessfuly enabled on the system. In adition to this code illustrate two other features. Namely it is error checking and gpu information query. 

## Error checking
As all C/C++ programmers know runtime errors do happen and with extensions added by CUDA this extends as well. To understand the concept of error checking one has to understand that runtime error handling is not free and every messages parsing from GPU to CPU cost a synchronization between the devices. This can be partially remedied by employing a buffer behind the scenes but even though some slowdown will be incurred.

For this reason runtime error checking is/has to be  build on top of cuda calls to enable error detection with full knowledge of the costs. CUDA toolkit offer its own implementation of runtime error detection described in greater detail [here](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html). In this code simpler (but perfectly usable) version has been created. 

The method used here consist of two cases. Function calls and kernell calls. Function calls check if the function called return an error code and this code is displayed by clever use of macro wrapper. This way is fine and because the calls are issued by CPU this does not slow excution significantly. One can tweak the output to personal liking if desired.
It is strongly advised to always use these checks on cuda specific functions (*cuda\*()*) as it incurrs negligible cost and discovers all issues from wrong initialization to faulty memory copies.

The other case of kernel call is more complicated because the errors here hapen on the GPU (device). And kernels are asynchronous with regard to CPU. As such the a synchronization needs to take place for CPU to realize that problem occured, hence imposed slowdown. 
Be aware that in this category fall all kinds of wron kernel launch parameters and issues withnin kernel implementation. It is strongly recomended to have these check after all kernels under development or in DEBUG configuration. Otherwise issues are silently being ignored.

## Information query
Another feature is the programatical way how to gather properties of used device (GPU). It is useful to adjust the code settings according to the used hardware. Therefore information about available memories and other parameters can be crucial in development of truly efficient software operating on multiple devices. 

The rudimentary version shown in the code handle information query from a single GPU present on the machine. The query is limited to memory only but more information can be gathered than this. For more detail refer to the [documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html).
For programming details in action one can browse the source code of deviceQuerry from nvidia toolkit samples under 1_Utilities.

## Try out
To look closer to sheduling and execution of kernels try executing the program with different thread_count and block_count parameters. Notice where the bounds lie for your device (try i.e. 10 000 threads per block). Notice order of output (try more threads than warp in total in single block). 
> **BEWARE:** The prints from kernel are ordered for warp size - that **does not** mean that regular calculations are as well.

Predict what will be the difference between *say_hello<<<64,1>>>()* and *say_hello<<<1,64>>>()*. Verify your answers.