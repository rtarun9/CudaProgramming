#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 65536
#define THREAD_GROUP_DIM 256

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

// Attempts to fix the above logic in 2 ways:
// (i) Code is such that threads 1, 3, 5, ... will not be idle / masked away,
// instead if you take array size as 8, thread index 0, 1, 2, 3 will calculator 
// array elem 0+1, 2+3, 4+5, 6+7 and so on. Due to this, in the case that
// block size > warp size, fewer thread diverges will occur.
// (ii) Expensive modulo operator not used.

// Problem this approach has:
// (i) Potential frequent memory bank conflicts.
// This is because we still hop around a lot in memory to read data from shared memory.
// If we somehow made sequential reads / writes into shared memory, memory bank conflicts would
// be largely reduced. (This is a issue since bank conflicts make memory reads sequential rather then concurrent).
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

// Sequential shared memory reads / writes to reduce bank conflicts.

// Problem : In the first loop iteration, we aren't even using half the threads in the block (they are only used for loading into shared memory, but thats it).
__global__ void parallel_reduce_addition_no_conflicts(int *array)
{
    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    shared_arr[threadIdx.x] = array[index];

    __syncthreads();

    // Go through all phases.
    for (int s = blockDim.x / 2; s > 0; s = s >> 1)
    {
        if (threadIdx.x < s)
        {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + s];
        }
        
        __syncthreads();
    }

    if (threadIdx.x == 0)
        array[blockIdx.x] = shared_arr[0];
}

// Sequential shared memory reads / writes to reduce bank conflicts.

// Problem with this approach : 
// We dont even use half of the threads in the first iteration :(
// But, we need them with current solution because we use all threads to load
// individual memory into shared memory.
// Solution : While loading data from global mem to shared mem, do the first add of reduction
// that time itself, and reduce number of blocks required by 2.

// Visual explanation of the old index logic. Consider a block do have 4 threads, and we have 2 blocks in total.
// Old method : 
// Block 1 : T0 T1 T2 T3    (shared memory : {arr0, arr1, arr2, arr3})
// Block 2 : T4 T5 T6 T7    (shared memory : {arr4 arr5 arr6, arr7})
// Now, in the new logic, we have 4 threads per block but half the num of blocks launched.
// index = threadIdx.x. + blockIdx.x * blockDim.x * 2.
// Block 1 : T0 T1 T2 T3    (shared memory : {arr0 + arr4, arr1 + arr5, arr2 + arr6, arr3 + arr7})
// Same example, but say we originally had 4 blocks, and now only 2.
// Block 1 : T0 T1 T2 T3    (shared memory : {arr0 + arr4, arr1 + arr5, arr2 + arr6, arr3 + arr7})
// Block 2 : T8 T9 T10 T11  (shared memory : {arr8 + arr12, arr9 + arr13, arr10 + arr14, arr11 + arr15}).

// Problem with this solution : number of active threads reduces in each loop iteration.
// when s (the 'stride') is <= 32, there is a single warp left, and that __syncthreads() call is redundent.
// We also dont need the threadIdx.x < s. Essentially in later stages of phase iterations, most threads do wasted checks and no real work.
__global__ void parallel_reduce_addition_reduce_idle_threads(int *array, int *output)
{
    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * 2 * blockDim.x;  
    if (index < NUM_ELEMENTS)
    {
        shared_arr[threadIdx.x] = array[index] + array[index + blockDim.x];
    }
    else
    {
        shared_arr[threadIdx.x] = 0;
    }
    

    __syncthreads();

    // Go through all phases.
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + s];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = shared_arr[0];
}

__device__ void warp_reduce(volatile int* array, int threadId)
{
    array[threadId] += array[threadId + 32];
    array[threadId] += array[threadId + 16];
    array[threadId] += array[threadId + 8];
    array[threadId] += array[threadId + 4];
    array[threadId] += array[threadId + 2];
    array[threadId] += array[threadId + 1];

}

__global__ void parallel_reduce_addition_with_warp_reduce(int *array, int *output)
{
    __shared__ int shared_arr[THREAD_GROUP_DIM];

    int index = threadIdx.x + blockIdx.x * 2 * blockDim.x;  

    shared_arr[threadIdx.x] = array[index] + array[index + blockDim.x];

    __syncthreads();

    // Go through all phases.
    for (int s = blockDim.x / 2; s > 32; s = s >> 1)
    {
        if (threadIdx.x < s)
        {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + s];
        }

        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        // the warp_reduce MUST have volatile before int *shared_arr.
        // We dont want any caching going on (which the compiler may do as optimization).
        // Marking as volatile will prevent this. 
        warp_reduce(shared_arr, threadIdx.x);
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = shared_arr[0];
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
    int *device_output = NULL;

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = 1;
    }

    cudaMalloc((void **)&device_array, NUM_ELEMENTS * sizeof(int));
    cudaMalloc((void **)&device_output, NUM_ELEMENTS * sizeof(int));
    cudaMemcpy(device_array, host_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

    int grid_dim = (NUM_ELEMENTS / (2 * THREAD_GROUP_DIM));
    dim3 thread_group_dim(THREAD_GROUP_DIM, 1, 1);

    // Now, if there are N blocks, we need to run the algorithm again to find the sum
    // of elements 0, 32, 64, ... and so on.
    parallel_reduce_addition_with_warp_reduce<<<grid_dim, thread_group_dim>>>(device_array, device_output);
    parallel_reduce_addition_with_warp_reduce<<<1, thread_group_dim>>>(device_output, device_output);
    
    printf("[For verification] : Sum : %d.\n", NUM_ELEMENTS);
    cudaMemcpy(host_array, device_output, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

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
    grid_dim *= 2;
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