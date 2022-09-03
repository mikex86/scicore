#ifndef STD_KERNEL_CUH
#define STD_KERNEL_CUH

#include <stdint.cuh>

#define KERNEL_EXPORT extern "C" __global__
#define KERNEL_TEMPLATE __device__ __forceinline__

#endif // STD_KERNEL_CUH