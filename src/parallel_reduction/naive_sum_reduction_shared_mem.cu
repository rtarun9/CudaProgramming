// Simple approach : Have 2 pass.
// One where each thread group finds the sum, and second where all thread groups find the collective sum.

#include <stdlib.h>
#include <stdio.h>
#include <random>

#define NUM_ELEMENTS (1 << 14)

__global__
void sum_reduction(int* input_array, int* output_array, int phase_count_start, int phase_count_end)
{
    __shared__ int smem[128 /* thread group dim*/];

    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    smem[threadIdx.x] = input_array[thread_id];
    __syncthreads();

    if (thread_id < NUM_ELEMENTS)
    {
        for (int i = phase_count_start; i <= phase_count_end; i++)
        {
            const int stride = 1 << i;

            if (thread_id % stride == 0)
            {
                smem[threadIdx.x] += smem[threadIdx.x + stride / 2];
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0)
    {
        printf("blockIdx.x %d has value %d\n", blockIdx.x, smem[threadIdx.x]);
        output_array[blockIdx.x] = smem[threadIdx.x];
    }
}

int main()
{
    int* host_array = (int*)malloc(NUM_ELEMENTS * sizeof(int));
    int* device_input_array = nullptr;
    int* device_output_array = nullptr;

    // Generate values for host array.
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = 1;
    }

    cudaMalloc(&device_input_array, sizeof(int) * NUM_ELEMENTS);
    cudaMalloc(&device_output_array, sizeof(int) * (int)ceilf(log2f(NUM_ELEMENTS)));

    cudaMemcpy(device_input_array, host_array, sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

    const dim3 thread_group_dim = dim3(128, 1, 1);
    const dim3 thread_grid_dim = dim3((int)ceil(NUM_ELEMENTS / (float)thread_group_dim.x), 1, 1);

    if (thread_grid_dim.x > thread_group_dim.x)
    {
        printf("This would cause program to not work correctly, as the 2 kernel launches will have different block dim");
    }

    // For first pass, num_passes = log(thread_group_dim)
    const int num_passes = ceilf(log2f(thread_group_dim.x));

    // First pass, each thread group finds its sum and stores in the array.
    sum_reduction<<<thread_grid_dim, thread_group_dim>>>(device_input_array, device_output_array, 1, num_passes);
    // Second pass, find the sum of each thread block.
    sum_reduction<<<1, thread_grid_dim>>>(device_output_array, device_output_array, 1, (int)ceilf(log2f(thread_grid_dim.x)));

    cudaDeviceSynchronize();

    size_t actual_res = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) actual_res += host_array[i];

    int computed_result = 0;
    cudaMemcpy(&computed_result, device_output_array, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("computed result %d\n", computed_result);
    printf("actual result %zd\n", actual_res);

    free(host_array);
    cudaFree(device_input_array);
    cudaFree(device_output_array);
}