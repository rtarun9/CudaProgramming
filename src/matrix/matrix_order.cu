// Experiment to see the CPU / GPU order of matrix.

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIMENSION 4

__global__ void matrix_order(int *const matrix)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    const int index = i + j * MATRIX_DIMENSION;

    matrix[index] = index;
}

int main()
{
    // Setup host side data / buffers.
    int *host_matrix = NULL;
    int *host_matrix_result = NULL;

    host_matrix = (int *)malloc(sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION);
    host_matrix_result = (int *)malloc(sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION);

    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            host_matrix[i + j * MATRIX_DIMENSION] = i + j * MATRIX_DIMENSION;
        }
    }

    // Setup device side data / buffers.
    int *device_matrix = NULL;

    cudaMalloc((void **)&device_matrix, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));

    // Copy data from host buffers to device buffers.
    cudaMemcpy(device_matrix, host_matrix, sizeof(int) * MATRIX_DIMENSION * MATRIX_DIMENSION, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Invoke the kernel to perform processing on the GPU.
    const dim3 thread_group_dim(32, 32, 1);
    const dim3 block_dim(MATRIX_DIMENSION / thread_group_dim.x, MATRIX_DIMENSION / thread_group_dim.y, 1);

    matrix_order<<<block_dim, thread_group_dim>>>(device_matrix);

    // Copy the result data from device to host memory.
    cudaMemcpy(host_matrix_result, device_matrix, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // Display results via console.
    printf("matrix host :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_matrix[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    printf("\nmatrix devcie :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_matrix_result[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    // Free allocated memory.
    free(host_matrix);
    free(host_matrix_result);
    
    cudaFree(device_matrix);

    return 0;
}