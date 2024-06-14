#include <stdio.h>
#include <stdlib.h>

// The blelloch algorithm is a O(n) work algorithm with O(log n) step complexity.
// 2 pass algorithm. First the reduce pass:
// Take the indices where (t + 1) % 2 == 0 and add with previous one.
// Then, (t + 1) % 4 == 0 and add with element 2 places before this.
// Then, do the same for (t + 1) % 8 == 0 and so on.

// Downsweep pass.
// Consider 2 elements a and b.
// Downsweep operator produces 2 outputs, the left element is just 'b', and the right is
// a + b.
// The threads which participate are the OPPOSITE of what occured in the reduction step.

// Example:
// a    b   c   d
//      (a+b)   (d+c)
//              (a+b+c+d)
//                 0
//       0    c  (a+b)
// 0    a   (a+b) (a+b+c)

#define NUM_ELEMENTS 1024 * 1024
#define BLOCK_DIM 1024

// Use shared memory and find the scan (prefix sum) for a small portion of the overall input array.
__global__ void blelloch_scan(int* input_output_array, int* per_block_scan_output)
{
    __shared__ int smem[BLOCK_DIM];

    const int tx = threadIdx.x + blockIdx.x * blockDim.x;

    smem[threadIdx.x] = input_output_array[tx];
    __syncthreads();

    // Reduction step.
    for (int i = 2; i < BLOCK_DIM; i *= 2)
    {
        int val_to_add_to_smem= 0;
        if ((threadIdx.x+ 1) % i == 0)
        {
            val_to_add_to_smem = smem[threadIdx.x - i / 2];
        }

        smem[threadIdx.x] += val_to_add_to_smem;
        __syncthreads();
    }

    smem[BLOCK_DIM - 1] = 0;
    __syncthreads();

    // Downsweep step.
    for (int i = BLOCK_DIM; i > 0; i = i / 2)
    {
        if (threadIdx.x > 0 && (threadIdx.x + 1) %  i  == 0)
        {
            int left = smem[threadIdx.x - i / 2];
            int right = smem[threadIdx.x];

            smem[threadIdx.x] = left + right;
            smem[threadIdx.x - i / 2] = right;
        }

        __syncthreads();
    }


    if (per_block_scan_output && threadIdx.x == blockDim.x - 1)
    {
        per_block_scan_output[blockIdx.x] = input_output_array[tx] + smem[threadIdx.x];
    }

    input_output_array[tx] = smem[threadIdx.x];
}

__global__ void add_array_with_per_block_scan_result(int* input_output_array, int* per_block_scan_output)
{
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x > 0)
    {
        input_output_array[tx] += per_block_scan_output[blockIdx.x];
    }
}

int main()
{
    constexpr size_t BYTES = NUM_ELEMENTS * sizeof(int);

    // Allocate and setup host side buffers.
    int* host_input_array = (int*)malloc(BYTES);
    int* host_output_array = (int*)malloc(BYTES);

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_input_array[i] = 1;
    }

    // Allocate and setup device side buffers.
    int* device_input_output_array = nullptr;
    cudaMalloc(&device_input_output_array, BYTES);
    cudaMemcpy(device_input_output_array, host_input_array, BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    const dim3 block_dim(BLOCK_DIM, 1, 1);
    const dim3 grid_dim((NUM_ELEMENTS + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);

    int* device_per_block_scan_output_array = nullptr;
    cudaMalloc(&device_per_block_scan_output_array, sizeof(int) * grid_dim.x);

    blelloch_scan<<<grid_dim, block_dim>>>(device_input_output_array, device_per_block_scan_output_array);

    // Now find the scan value Of the per block scans done in the previous step :wow:
    // This DOES mean that number of blocks launched in previous step = number of elements in device_per_block_scan_output_array
    // must be less than or equal to maximum number of threads that can be processed in a block (1024 in this case).
    {
        const dim3 per_block_scan_grid_dim((grid_dim.x + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
        blelloch_scan<<<per_block_scan_grid_dim, block_dim>>>(device_per_block_scan_output_array, nullptr);
    }

    // Now, add each element to the coresponding value of device_per_block_scan_output_array.
    add_array_with_per_block_scan_result<<<grid_dim, block_dim>>>(device_input_output_array, device_per_block_scan_output_array);

    cudaMemcpy(host_output_array, device_input_output_array, BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    bool success = true;
    int scan_result = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        if (scan_result != host_output_array[i])
        {
            printf("ERROR at index : %d. Got %d but expected %d.\n", i, host_output_array[i], scan_result);
            success = false;
        }
        scan_result += host_input_array[i];
    }

    if (success)
    {
        printf("Algorithm was succesfull\n");
    }

    free(host_input_array);
    free(host_output_array);

    cudaFree(device_input_output_array);
}