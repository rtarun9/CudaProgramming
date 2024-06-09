// Simple approach : Have log2(n) passes. Each pass you launch the SAME number of threads. 
// If thread index % pow(2, number_of_pass) == 0, add current with thread index + 2 * (number_of_pass - 1).

#include <stdlib.h>
#include <stdio.h>
#include <random>

#define NUM_ELEMENTS (int)pow(2, 11)

__global__
void sum_reduction(int* array, int phase_number)
{
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = 1 << phase_number;

    if (thread_id < NUM_ELEMENTS)
    {
        if (thread_id % stride == 0)
        {
            array[thread_id] += array[thread_id + stride / 2];
        }
    }
}

int main()
{
    int* host_array = (int*)malloc(NUM_ELEMENTS * sizeof(int));
    int* device_array = nullptr;

    // Generate values for host array.
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = rand();
    }

    cudaMalloc(&device_array, sizeof(int) * NUM_ELEMENTS);
    cudaMemcpy(device_array, host_array, sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

    const dim3 thread_group_dim = dim3(64, 1, 1);
    const dim3 thread_grid_dim = dim3((int)ceil(NUM_ELEMENTS / (float)thread_group_dim.x), 1, 1);

    const int num_passes = ceilf(log2f(NUM_ELEMENTS));
    printf("Number of passes :: %d\n", num_passes);

    for (int i = 1; i <= num_passes; i++)
    {
        sum_reduction<<<thread_grid_dim, thread_group_dim>>>(device_array, i);
        cudaDeviceSynchronize();
    }

    int computed_result = 0;
    cudaMemcpy(&computed_result, device_array, sizeof(int) * 1, cudaMemcpyDeviceToHost);


    size_t actual_res = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) actual_res += host_array[i];

    printf("computed result %d\n", computed_result);
    printf("actual result %zd\n", actual_res);

    free(host_array);
    cudaFree(device_array);
}