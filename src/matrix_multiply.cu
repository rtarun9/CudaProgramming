#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIMENSION 32

// Idea on how to approach.
// Consider the matrix to be 32x32.
// 1 2 3 4 5 .... 31
// 32 33 ......   63
// ........     1023

// We can solve the problem using a single block on the block grid consisting of 32 x 32 (1024) threads.
// So, blockDim = (1, 1, 1), and threadGroupDim = (32, 32, 1)

__global__ void matrix_multiply(const int *const matrix_a, const int *const matrix_b, int *const matrix_sum_result)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    const int index = i + j * MATRIX_DIMENSION;

    int accumulator = 0;

    for (int k = 0; k < MATRIX_DIMENSION; k++)
    {
        // Copy data from global memory to registers.
        int a = matrix_a[k + i * MATRIX_DIMENSION];
        int b = matrix_b[k * MATRIX_DIMENSION + j];

        accumulator += a * b;
    }

    matrix_sum_result[index] = accumulator;
}

int main()
{
    // Setup host side data / buffers.
    int *host_matrix_a = NULL;
    int *host_matrix_b = NULL;
    int *host_matrix_product_result = NULL;

    host_matrix_a = (int *)malloc(sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION);
    host_matrix_b = (int *)malloc(sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION);
    host_matrix_product_result = (int *)calloc(MATRIX_DIMENSION * MATRIX_DIMENSION, sizeof(int));

    for (int i = 0; i < MATRIX_DIMENSION * MATRIX_DIMENSION; i++)
    {
        host_matrix_a[i] = i;
        host_matrix_b[i] = i;
    }

    // Setup device side data / buffers.
    int *device_matrix_a = NULL;
    int *device_matrix_b = NULL;
    int *device_matrix_product_result = NULL;

    cudaMalloc((void **)&device_matrix_a, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));
    cudaMalloc((void **)&device_matrix_b, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));
    cudaMalloc((void **)&device_matrix_product_result, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));

    // Copy data from host buffers to device buffers.
    cudaMemcpy(device_matrix_a, host_matrix_a, sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, host_matrix_b, sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Invoke the kernel to perform processing on the GPU.
    const dim3 thread_group_dim(32, 32, 1);
    const dim3 block_dim(MATRIX_DIMENSION / thread_group_dim.x, MATRIX_DIMENSION / thread_group_dim.y, 1);

    matrix_multiply<<<block_dim, thread_group_dim>>>(device_matrix_a, device_matrix_b, device_matrix_product_result);

    // Copy the result data from device to host memory.
    cudaMemcpy(host_matrix_product_result, device_matrix_product_result, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // Display results via console.
    printf("matrix A :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_matrix_a[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    printf("\nmatrix B :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_matrix_b[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    printf("\n matrix product :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_matrix_product_result[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    printf("\n Expected resume :: \n");

    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            int sum = 0;
            for (int k = 0; k < MATRIX_DIMENSION; k++)
            {
                sum += host_matrix_a[k + i * MATRIX_DIMENSION] * host_matrix_b[k * MATRIX_DIMENSION + j];
            }

            printf("%d ", sum);
            if (sum != host_matrix_product_result[i + j * MATRIX_DIMENSION])
            {
                printf("ERROR at index %d %d.\n", i, j);
                return -1;
            }
        }
        printf("\n");
    }

    // Free allocated memory.
    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_product_result);

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_product_result);

    printf("%s", "All calculations were correct!\n");

    return 0;
}