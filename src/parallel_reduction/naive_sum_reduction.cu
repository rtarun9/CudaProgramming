// Simple approach : Have log2(n) passes. Each pass you launch the SAME number of threads. 
// If thread index % pow(2, number_of_pass) == 0, add current with thread index + 2 * (number_of_pass - 1).

#include <stdlib.h>
#include <stdio.h>

#define NUM_ELEMENTS (int)pow(2, 11)

__global__
void sum_reduction(int* array, int phase_number)
{
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id < NUM_ELEMENTS)
    {
        if (thread_id % (int)pow(2, phase_number) == 0)
        {
            array[thread_id] += array[thread_id + (int)pow(2, phase_number - 1)];
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
        host_array[i] = i * 3;
    }

    cudaMalloc(&device_array, sizeof(int) * NUM_ELEMENTS);
    cudaMemcpy(device_array, host_array, sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

    const dim3 thread_group_dim = dim3(32, 1, 1);
    const dim3 thread_grid_dim = dim3((int)ceil(NUM_ELEMENTS / (float)thread_group_dim.x), 1, 1);


    const int num_passes = ceilf(log2f(NUM_ELEMENTS));

    for (int i = 1; i <= num_passes; i++)
    {
        sum_reduction<<<thread_grid_dim, thread_group_dim>>>(device_array, i);
    }

    // Find the actual result.
    size_t res = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        res += host_array[i];
    }

    int computed_result = 0;
    cudaMemcpy(&computed_result, device_array, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    int* temp = (int*)malloc(NUM_ELEMENTS * sizeof(int));
    cudaMemcpy(temp, device_array, sizeof(int) * NUM_ELEMENTS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        printf("%d %d\n", host_array[i], temp[i]);
    }

    printf("Actual result : %zd computed result %d\n", res, computed_result);

    free(host_array);
    cudaFree(device_array);
}