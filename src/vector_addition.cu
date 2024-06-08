#include <stdio.h>
#include <stdlib.h>

#define VECTOR_DIMENSION 32 * 32 * 11 * 9 * 33

__global__ 
void vector_add(const int* const vector_a, const int* const vector_b, int *const vector_sum_result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < VECTOR_DIMENSION)
    {
        // Copy data from global memory to registers.
        int a = vector_a[i];
        int b = vector_b[i];

        vector_sum_result[i] = a + b;
    }
}

int main()
{
    // Setup host side data / buffers.
    int *host_vector_a = NULL;
    int *host_vector_b = NULL;
    int *host_vector_sum_result = NULL;

    host_vector_a = (int*)malloc(sizeof(int) * VECTOR_DIMENSION);
    host_vector_b = (int*)malloc(sizeof(int) * VECTOR_DIMENSION);
    host_vector_sum_result = (int*)calloc(VECTOR_DIMENSION, sizeof(int));

    for (int i = 0; i < VECTOR_DIMENSION; i++)
    {
        host_vector_a[i] = i;
        host_vector_b[i] = i * 2;
    }

    // Setup device side data / buffers.
    int *device_vector_a = NULL;
    int *device_vector_b = NULL;
    int *device_vector_sum_result = NULL;

    cudaMalloc((void**)&device_vector_a, VECTOR_DIMENSION * sizeof(int));
    cudaMalloc((void**)&device_vector_b, VECTOR_DIMENSION * sizeof(int));
    cudaMalloc((void**)&device_vector_sum_result, VECTOR_DIMENSION * sizeof(int));

    // Copy data from host buffers to device buffers.
    cudaMemcpy(device_vector_a, host_vector_a, sizeof(int) * VECTOR_DIMENSION, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, sizeof(int) * VECTOR_DIMENSION, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Invoke the kernel to perform processing on the GPU.
    const dim3 thread_group_dim(64, 1, 1);
    const dim3 block_dim((VECTOR_DIMENSION + thread_group_dim.x - 1) / thread_group_dim.x, 1, 1);

    vector_add<<<block_dim, thread_group_dim>>>(device_vector_a, device_vector_b, device_vector_sum_result);

    // Copy the result data from device to host memory.
    cudaMemcpy(host_vector_sum_result, device_vector_sum_result, VECTOR_DIMENSION * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // Display results via console.
    #if 0
    printf("Vector A :: \n");
    for (int i = 0; i < VECTOR_DIMENSION; i++)
    {
        printf("%d ", host_vector_a[i]);
    }

    printf("\nVector B :: \n");
    for (int i = 0; i < VECTOR_DIMENSION; i++)
    {
        printf("%d ", host_vector_b[i]);
    }
    #endif

    bool success = true;
    for (int i = 0; i < VECTOR_DIMENSION; i++)
    {
        if (host_vector_sum_result[i] != host_vector_a[i] + host_vector_b[i])
        {
            success = false;
            break;
        }
        else
        {
            printf("%d %d = %d\n", host_vector_a[i], host_vector_b[i], host_vector_sum_result[i]);
        }
    }

    if (success)
    {
        printf("Program was successfull!");
    }

    // Free allocated memory.
    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_sum_result);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_sum_result);

    return 0;
}