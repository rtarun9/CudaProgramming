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
//        0       (a+b)
//  0     a   (a+b)(a+b+c)

#define NUM_ELEMENTS 1024
#define BLOCK_DIM 1024

__global__ void blelloch_scan(int* input_output_array)
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

    input_output_array[tx] = smem[threadIdx.x];
}

int main()
{
    constexpr size_t BYTES = NUM_ELEMENTS * sizeof(int);

    // Allocate and setup host side buffers.
    int* host_input_array = (int*)malloc(BYTES);
    int* host_output_array = (int*)malloc(BYTES);

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_input_array[i] = i;
    }

    // Allocate and setup device side buffers.
    int* device_input_output_array = nullptr;
    cudaMalloc(&device_input_output_array, BYTES);
    cudaMemcpy(device_input_output_array, host_input_array, BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    const dim3 block_dim(BLOCK_DIM, 1, 1);
    const dim3 grid_dim((NUM_ELEMENTS + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);

    blelloch_scan<<<grid_dim, block_dim>>>(device_input_output_array);

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
        printf("Algorithm was succesfull.");
        printf("Output : \n");
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            printf("%d ", host_output_array[i]);
        }
    }

    free(host_input_array);
    free(host_output_array);

    cudaFree(device_input_output_array);
}