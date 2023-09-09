#include "common.h"

// What is Scan? (or parallel prefix sum):
// Consider a array arr with elements [a0, a1, a2, ..., an]
// The (exclusive) scan for this array given a associative operator (O) is:
// [I, a0, a0 O a1, a0 O a1 O a2, .... a0 O a1 O ..... an-2]
// This is called exclusive because for any index i in the result array, element arr[i] is not present
// (i.e the result at index i is the result of applying O on all elements prior to it).
// In inclusive scan, we dont have the first element (identity), and the resulting array becomes:
// [a0, a0 O a1, a0 O a1 O a2, ....].
// Exclusive and Inclusive scans can be generated from each other (for I -> E we have to right shift by one element and include the identity at the start,
// while for E -> I we need to left shfit and require the input array to apply O on the last element of input with the original last element of exclusive scan.

// The example here uses + as the associative operator.
void sequential_scan_inclusive(int* arr, int* output, int size)
{
    output[0] = arr[0];

    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + arr[i];
    }
}

void sequential_scan_exclusive(int* arr, int* output, int size, int identity)
{
    output[0] = identity;
    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + arr[i - 1];
    }
}

// Brute force parallel implementation (the We don't have a for loop from 1 to n, but instead from 1 to threadIdx.x).
__global__
void parallel_scan_inclusive_brute_force(int* arr, int* output, int size)
{
    if (threadIdx.x >= size)
    {
        return;
    }

    output[threadIdx.x] = arr[threadIdx.x];

    int res = 0;
    for (int i = 0; i <= threadIdx.x; i++)
    {
        res += arr[i];
    }

    output[threadIdx.x] = res;
}

// Naive parallel scan method (as described here : https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
// The idea is to have a set of 'phases', where the number of phases = log base 2 (input size).
// In phase 0, each thread set the value of output[threadId.x.] to input[threadIdx.x].
// In phase 1, Each thread (except the first) applies the operator on itself and element at previous element.
// In phase 2, the same thing happens, but only elements with threadId >= (2 ^ p - 1)  (p = phase number).
// So, in any phase X, participating threads have id's >= (2 ^ X) - 1, and they do X[i] = x[i] + x[i - (2^X - 1)].
// Note that this is not work efficient as in sequential implementation (on CPU) number of additions is N, while here is it N (log N).
__global__
void parallel_scan_inclusive_naive(int* arr, int* output, int size)
{
    if (threadIdx.x >= size)
    {
        return;
    }

    // Phase '0'.
    output[threadIdx.x] = arr[threadIdx.x];

    for (int phase = 1; phase <= ceilf(__log2f(size)); phase++)
    {
        if (threadIdx.x >= pow(2, phase - 1))
        {
            output[threadIdx.x] = output[threadIdx.x] + output[int(threadIdx.x - __powf(2, phase - 1))];
            __syncthreads();
        }
    }
}

int main()
{
    int* u_input_array = NULL;
    int* u_output_array = NULL;

    const int SIZE = 96;
    
    cudaMallocManaged((void**)&u_input_array, sizeof(int) * SIZE);
    cudaMallocManaged((void**)&u_output_array, sizeof(int) * SIZE);

    // Fill in array with simple data.
    fill_with_increment(u_input_array, SIZE, 1);
    printf("Input array : \n");
    print_array(u_input_array, SIZE);

    sequential_scan_exclusive(u_input_array, u_output_array, SIZE, 0);
    printf("Exclusive sequential scan\n");
    print_array(u_output_array, SIZE);

    sequential_scan_inclusive(u_input_array, u_output_array, SIZE);
    printf("Inclusive sequential scan\n");
    print_array(u_output_array, SIZE);

    int num_threads = SIZE;
    int num_blocks = 1;

    parallel_scan_inclusive_brute_force<<<num_blocks, num_threads>>>(u_input_array, u_output_array, SIZE);
    cudaDeviceSynchronize();

    printf("Inclusive Brute force parallel scan\n");
    print_array(u_output_array, SIZE);

    parallel_scan_inclusive_naive<<<num_blocks, num_threads>>>(u_input_array, u_output_array, SIZE);
    cudaDeviceSynchronize();

    printf("Inclusive Naive parallel scan\n");
    print_array(u_output_array, SIZE);

    return 0;
}