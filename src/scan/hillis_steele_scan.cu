// Multi thread block hillis steele scan algorithm.
// This is a 2 phase algorithm. First, each block computes its local scan (using shared memory).
// Each block stores the value of its local final element to a differeant array. 
// This value is added to each element of threads block with index i + 1.
// NOTE : 
// Inorder to keep this algorithm limited to 3 passes, the MAXIMUM number of elements that can processed for now is 
// 1024 * 1024.

#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 1024 * 1024
#define BLOCK_DIM 1024

// Use shared memory and find the scan (prefix sum) for a small portion of the overall input array.
__global__ void hillis_steele_local_scan(int* input_output_array, int* per_block_scan_output)
{
    __shared__ int smem[BLOCK_DIM];

    const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= NUM_ELEMENTS)
    {
        return;
    }

    smem[threadIdx.x] = input_output_array[tx];
    __syncthreads();

    for (int stride = 1; stride < BLOCK_DIM; stride *= 2)
    {
        int val_to_add_to_smem = 0;
        if (threadIdx.x >= stride)
        {
            val_to_add_to_smem = smem[threadIdx.x - stride];
        }

        __syncthreads();

        smem[threadIdx.x] += val_to_add_to_smem;

        __syncthreads();
    }

    input_output_array[tx] = smem[threadIdx.x];

    if (per_block_scan_output && threadIdx.x == blockDim.x - 1)
    {
        per_block_scan_output[blockIdx.x] = smem[threadIdx.x];
    }
}

__global__ void add_array_with_per_block_scan_result(int* input_output_array, int* per_block_scan_output)
{
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x >= 1)
    {
        input_output_array[tx] += per_block_scan_output[blockIdx.x - 1];
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

    // Lauch kernel to compute per block local scan.
    hillis_steele_local_scan<<<grid_dim, block_dim>>>(device_input_output_array, device_per_block_scan_output_array);

    // Now find the scan value Of the per block scans done in the previous step :wow:
    // This DOES mean that number of blocks launched in previous step = number of elements in device_per_block_scan_output_array
    // must be less than or equal to maximum number of threads that can be processed in a block (1024 in this case).
    {
        const dim3 per_block_scan_grid_dim((grid_dim.x + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
        hillis_steele_local_scan<<<per_block_scan_grid_dim, block_dim>>>(device_per_block_scan_output_array, nullptr);
    }

    // Now, add each element to the coresponding value of device_per_block_scan_output_array.
    add_array_with_per_block_scan_result<<<grid_dim, block_dim>>>(device_input_output_array, device_per_block_scan_output_array);

    cudaMemcpy(host_output_array, device_input_output_array, BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    bool success = true;
    int scan_result = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        scan_result += host_input_array[i];
        if (scan_result != host_output_array[i])
        {
            printf("ERROR at index : %d. Got %d but expected %d.\n", i, host_output_array[i], scan_result);
            success = false;
        }
    }

    if (success)
    {
        printf("Algorithm was succesfull.");
    }

    free(host_input_array);
    free(host_output_array);

    cudaFree(device_input_output_array);
    cudaFree(device_per_block_scan_output_array);
}