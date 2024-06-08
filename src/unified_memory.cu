// Unified memory makes it very easy to allocate and access data that can be used
// from both the cpu and gpu.
// In GPU Api programming terms, it is the region of memory that is not as fast as 
// only host / only device, but can be accessed by both.
// HOW IT WORKS:
// In modern GPU architectures, memory may not be allocated when cudaMallocManaged is called().
// The memory & page table entries are not created until the data is accessed by GPU or CPU.
// If data is on CPU and then required by GPU, page table entries must be migrated, which has slight overhead.
// Pre-fetching can hint the driver to explicitely migrate memory before use.

#include <stdio.h>
#include <stdlib.h>

__global__ void multiply_by_factor(int* input, int* output, int factor)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    output[index] = input[index] * factor;
}

__global__ void fill_data(int* input)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    input[index] = index;
}

int main()
{
    const int N = 32;
    int* input_array{nullptr};
    int* output_array{nullptr};

    cudaMallocManaged(&input_array, sizeof(int) * N);
    cudaMallocManaged(&output_array, sizeof(int) * N);

    fill_data<<<1, N>>>(input_array);

    // Nwo, if twe access this memory from the CPU, then when kernel is launched we have to move data to GPU (since page fault will happen).
    // To prevent this, fill in data on the GPU directly. For arrays where no data is to be filled, you can prefetch and hint the driver that 
    // the memory will PROBABLY be required for use by GPU.
    #if 0
    for (int i = 0; i < N; i++)
    {
        array[i] = i;
    }
    #endif

    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaMemPrefetchAsync(&output_array, sizeof(int) * N, device_id);

    multiply_by_factor<<<1, N>>>(input_array, output_array, 2);

    // Before accesing the array (ptr) on host side, wait for the gpu to finish.
    // In regular (non - unified) memory access, stuff like memcpy is essentially a memory barrier, which is why this device synchronize was not required.
    cudaDeviceSynchronize();


    // When the data is being read / written to by CPU, the driver will automagically handle the barriers and other cases for us.
    for (int i = 0; i < N; i++)
    {
        printf("array[%d] = %d.\n", i, output_array[i]);
    }

    return 0;
}