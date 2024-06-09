
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

    // Whats the issue with the previous (reduced divergence) algorithm?
    // Notice that the way we access memory is... non sequential.
    // In phase 1, you can index 0 accessing 0 and 1, index 1 accessing 2 and 3, etc.
    // This is fine for the most part. But, in later strides, you may have cases where
    // index 0 is accesssing 0 and 32, index 1 is accessing 64 and 96, etc.
    // the issue is there is a chance these elements (index 32 * X where X = 0, 1, 2, ..) lie in the same bank.
    // Think of memory bank like cache, where you have Y (say, 32 slots) and elements with address % Y == 0 lie in same bank.
    // In case of bank conflicts, the memory accesses become SEQUENTIAL.
    // How to prevent this? If memory access is linear for all threads (say index 0 accesses 0 and 8, index 1 access 1 adn 9, etc),
    // you have lot less bank conflicts.
    if (thread_id < NUM_ELEMENTS)
    {
        int stride = blockDim.x / 2;
        for (int i = phase_count_start; i <= phase_count_end; i++)
        {
            // Even though there is this if statement, divergence in the warp is still not too high.
            if (threadIdx.x < stride)
            {
                // the stride keeps reducing in this case.
                smem[threadIdx.x] += smem[threadIdx.x + stride];
                __syncthreads();
                stride /= 2;
            }
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
        host_array[i] = rand();
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

    printf("Parallel sum reduction with reduced bank conflicts.\n");
    printf("computed result %d\n", computed_result);
    printf("actual result %zd\n", actual_res);

    free(host_array);
    cudaFree(device_input_array);
    cudaFree(device_output_array);
}