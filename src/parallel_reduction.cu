#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 65536
#define THREAD_GROUP_DIM 256

// Attempts to fix the above logic in 2 ways:
// (i) Code is such that threads 1, 3, 5, ... will not be idle / masked away,
// instead if you take array size as 8, thread index 0, 1, 2, 3 will calculator 
// array elem 0+1, 2+3, 4+5, 6+7 and so on. Due to this, in the case that
// block size > warp size, fewer thread diverges will occur.
// (ii) Expensive modulo operator not used.
__global__ void parallel_reduce_addition_interleaved_addressing(int *array)
{
    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    shared_arr[threadIdx.x] = array[index];

    __syncthreads();

    // Go through all phases.
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * threadIdx.x;
        if (i < blockDim.x)
        {
            shared_arr[i] += shared_arr[i + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        array[blockIdx.x] = shared_arr[0];
}

// Problems with this logic:
// (i) Modulo operator is very expensive
// (ii) Lot of inactive threads in warp + lot of divergence (ESPECIALLY when block size > warp size).
__global__ void parallel_reduce_addition_naive(int *array)
{
    // Idea : Consider the array
    // 1 2 3 4 5 6 7 8. We will have log2(NUM_ELEMENTS) phases.
    // In phase one, we add element[i+_a_value] to element[i].
    // so, we have:
    // 3 _ 7 _ 11 _ 15_
    // In phase two, we add element[i - 2] to element[i].
    // 10 _ _ _ 26 _ _ _
    // Finally, we add element[i - 4] to element[i].
    // 36 _ _ _ _ _ _ _.
    // For reduction of number of idle threads,
    // multiple kernel launches are made.

    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    shared_arr[threadIdx.x] = array[index];

    __syncthreads();

    // Go through all phases.
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
        {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + s];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
        array[blockIdx.x] = shared_arr[0];
}


__global__ void parallel_reduce_max_naive(int *array)
{
    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    shared_arr[threadIdx.x] = array[index];

    __syncthreads();

    // Go through all phases.
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
        {
            shared_arr[threadIdx.x] = max(shared_arr[threadIdx.x + s], shared_arr[threadIdx.x]);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
        array[blockIdx.x] = shared_arr[0];
}

int main()
{
    int *host_array = (int *)(malloc(NUM_ELEMENTS * sizeof(int)));
    int *device_array = NULL;

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = 1;
    }

    cudaMalloc((void **)&device_array, NUM_ELEMENTS * sizeof(int));
    cudaMemcpy(device_array, host_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

    const dim3 grid_dim(NUM_ELEMENTS / THREAD_GROUP_DIM, 1, 1);
    const dim3 thread_group_dim(THREAD_GROUP_DIM, 1, 1);

    // Now, if there are N blocks, we need to run the algorithm again to find the sum
    // of elements 0, 32, 64, ... and so on.
    parallel_reduce_addition_interleaved_addressing<<<grid_dim, thread_group_dim>>>(device_array);
    parallel_reduce_addition_interleaved_addressing<<<1, thread_group_dim>>>(device_array);
    
    printf("[For verification] : Sum : %ll.\n", (long long)NUM_ELEMENTS);
    cudaMemcpy(host_array, device_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("Sum of elements :  %d.\n", host_array[0]);

    int max_element = INT_MIN;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = rand() % 100000;
        //printf("[%d] => %d.\n", i, host_array[i]);
        max_element = max(max_element, host_array[i]);
    }

    printf("[For verfiication] : Max element : %d.\n", max_element);

    cudaMemcpy(device_array, host_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    parallel_reduce_max_naive<<<grid_dim, thread_group_dim>>>(device_array);
    parallel_reduce_max_naive<<<1, thread_group_dim>>>(device_array);

    cudaMemcpy(host_array, device_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    printf("\n\n");
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        //printf("[%d] => %d.\n", i, host_array[i]);
    }
    printf("Max of elements :  %d.\n", host_array[0]);

    free(host_array);
    cudaFree(device_array);
}