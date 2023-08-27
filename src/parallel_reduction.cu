#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 8

__global__ void parallel_reduce_addition(int* array)
{
    // Idea : Consider the array 
    // 1 2 3 4 5 6 7 8. We will have log2(NUM_ELEMENTS) phases.
    // In phase one, we add element[i] to element[i + 1].
    // so, we have:
    // _ 3 _ 7 _ 11 _ 15
    // In phase two, we add element[i - 2] to element[i].
    // _ _ _ 10 _ _ _ 36
    // Finally, we add element[i - 4] to element[i].
    // _ _ _ _ _ _ _ 36.

    const int num_phases = int(__log2f(NUM_ELEMENTS));

    for (int i = 0; i < num_phases; i++)
    {
        for (int j = 0; j < NUM_ELEMENTS; j += pow(2, i + 1))
        {
            array[int(j + pow(2, i + 1) - 1)] += array[int(j + pow(2, i) - 1)];
        }
    }

    // The result is now in array[NUM_ELEMENTS - 1].
}

__global__ void parallel_reduce_max(int* array)
{
    const int num_phases = int(__log2f(NUM_ELEMENTS));

    for (int i = 0; i < num_phases; i++)
    {
        for (int j = 0; j < NUM_ELEMENTS; j += pow(2, i + 1))
        {
            array[int(j + pow(2, i + 1) - 1)] = max(array[int(j + pow(2, i) - 1)], array[int(j + pow(2, i + 1) - 1)]);
        }
    }

    // The result is now in array[NUM_ELEMENTS - 1].
}

int main()
{
    int *host_array = (int*)(malloc(NUM_ELEMENTS * sizeof(int)));
    int *device_array = NULL;

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = i + 1;
    }

    cudaMalloc((void**)&device_array, NUM_ELEMENTS * sizeof(int));
    cudaMemcpy(device_array, host_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

    const dim3 grid_dim(1, 1, 1);
    const dim3 thread_group_dim(NUM_ELEMENTS, 1, 1);

    parallel_reduce_addition<<<grid_dim, thread_group_dim>>>(device_array);
    cudaMemcpy(host_array, device_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("Sum of elements :  %d.\n", host_array[NUM_ELEMENTS - 1]);

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        host_array[i] = rand() % 100;
        printf("[%d] = %d.\n", i, host_array[i]);
    }

    cudaMemcpy(device_array, host_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    parallel_reduce_max<<<grid_dim, thread_group_dim>>>(device_array);
    
    cudaMemcpy(host_array, device_array, NUM_ELEMENTS * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("Max of elements :  %d.\n", host_array[NUM_ELEMENTS - 1]);

    free(host_array);
    cudaFree(device_array);
}