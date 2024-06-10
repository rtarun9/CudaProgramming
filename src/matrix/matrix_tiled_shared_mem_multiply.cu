// Optimized version of matrix multiplication using cache tiling.

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIMENSION 16
#define TILE_DIMENSION 2

// Each thread block has a dimension of TILE_DIMENSION * TIME_DIMENSION.
// Number of phases : Number of iterations required to compute the result for the current tile.
// Typically it will be MATRIX_DIMENSION / TILE_DIMENSION.
// Say the tile dimension is 4x4.
// Each thread in the tile will load 2 elements from A and B in shared memory.
// By doing so, block will load TILE_DIMENSION * TIME_DIMENSION chunk of matrix A and B in shared memory.
// Final matrix product can be broken down into sub matrix products. Here, to multiply A and B,
// we multiply the sub matrices / chunks of A and B.
// After loading, each thread computes the product / solution AT THAT location / index.
// Continue this for all phases, and store in the resulant matrix.
__global__ void matrix_multiply_tile_based(const int *const matrix_a, const int *const matrix_b, int *const matrix_product_result)
{
    __shared__ int shared_mem_chunk_a[TILE_DIMENSION][TILE_DIMENSION];
    __shared__ int shared_mem_chunk_b[TILE_DIMENSION][TILE_DIMENSION];

    int accumulator = 0;

    const int row = threadIdx.x + TILE_DIMENSION * blockIdx.x;
    const int column = threadIdx.y + TILE_DIMENSION * blockIdx.y;

    // Number of phases / iterations required to compute the matrix product for a single element.
    for (int i = 0; i < MATRIX_DIMENSION / TILE_DIMENSION; i++)
    {
        // Each thread loads 2 values into shared memory (one for chunk_a, one for chunk_b).

        shared_mem_chunk_a[threadIdx.x][threadIdx.y] = matrix_a[row * MATRIX_DIMENSION + i * TILE_DIMENSION + threadIdx.y];
        shared_mem_chunk_b[threadIdx.x][threadIdx.y] = matrix_b[(i * TILE_DIMENSION + threadIdx.x) * MATRIX_DIMENSION + column];

        __syncthreads();

        // Each thread in block computes sub-result.

        for (int k = 0; k < TILE_DIMENSION; k++)
        {
            accumulator += shared_mem_chunk_a[threadIdx.x][k] * shared_mem_chunk_b[k][threadIdx.y];
        }

        __syncthreads();
    }

    matrix_product_result[row + column * MATRIX_DIMENSION] = accumulator;
}

bool check_results(int* host_matrix_a, int *host_matrix_b, int *host_result_matrix)
{
    printf("\n matrix product :: \n");
    for (int i = 0; i < MATRIX_DIMENSION; i++)
    {
        for (int j = 0; j < MATRIX_DIMENSION; j++)
        {
            printf("%d ", host_result_matrix[i + j * MATRIX_DIMENSION]);
        }
        printf("\n");
    }

    printf("\n Expected result :: \n");

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
            if (sum != host_result_matrix[i + j * MATRIX_DIMENSION])
            {
                printf("ERROR at index %d %d.\n", i, j);
                return false;
            }
        }
        printf("\n");
    }

    return true;
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
        host_matrix_b[i] = (i/3);
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
    const dim3 thread_group_dim(TILE_DIMENSION, TILE_DIMENSION, 1);
    const dim3 block_dim(MATRIX_DIMENSION / thread_group_dim.x, MATRIX_DIMENSION / thread_group_dim.y, 1);

    matrix_multiply_tile_based<<<block_dim, thread_group_dim>>>(device_matrix_a, device_matrix_b, device_matrix_product_result);

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

    if (!check_results(host_matrix_a, host_matrix_b, host_matrix_product_result))
    {
        printf("Error in computation!\n");
    }
    else
    {
        printf("%s", "All calculations were correct!\n");
    }

    // Free allocated memory.
    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_product_result);

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_product_result);

    return 0;
}