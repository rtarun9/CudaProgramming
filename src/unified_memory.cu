// Unified memory makes it very easy to allocate and access data that can be used
// from both the cpu and gpu.
// In GPU Api programming terms, it is the region of memory that is not as fast as 
// only host / only device, but can be accessed by both.

#include <stdio.h>
#include <stdlib.h>

__global__ void multiply_by_factor(int* array, int factor)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    array[index] = array[index] * factor;
}

int main()
{
    const int N = 32;
    int *array = NULL;

    cudaMallocManaged((void**)&array, sizeof(int) * N);

    for (int i = 0; i < N; i++)
    {
        array[i] = i;
    }

    multiply_by_factor<<<1, N>>>(array, 2);

    // Before accesing the array (ptr) on host side, wait for the gpu to finish.
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        printf("array[%d] = %d.\n", i, array[i]);
    }

    return 0;
}