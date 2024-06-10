#include <stdio.h>
#include <stdlib.h>

__global__ void cube(const float* input_array, float* output_array)
{
    int tid = threadIdx.x;
    float value = input_array[tid];
    output_array[tid] = (value * value * value);
}

int main()
{
    // Allocate host memory and data.
    constexpr size_t NUM_ELEMENTS = 96;
    constexpr size_t ARRAY_BYTES = sizeof(float) * NUM_ELEMENTS;

    float* h_input_data = (float*)malloc(ARRAY_BYTES);
    float* h_output_data = (float*)malloc(ARRAY_BYTES);

    for (size_t i = 0; i < NUM_ELEMENTS; i++)
    {
        h_input_data[i] = i;
    }

    // Allocate device memory.
    float* d_input_data = nullptr;
    float* d_output_data = nullptr;
    cudaMalloc(&d_input_data, ARRAY_BYTES);
    cudaMalloc(&d_output_data, ARRAY_BYTES);

    // Now, copy data from CPU to GPU.
    cudaMemcpy(d_input_data, h_input_data, ARRAY_BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Launch kernel.
    cube<<<dim3(1, 1, 1), dim3(NUM_ELEMENTS, 1, 1)>>>(d_input_data, d_output_data);

    // Copy data from GPU to CPU to visualize the results.
    cudaMemcpy(h_output_data, d_output_data, ARRAY_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < NUM_ELEMENTS; i++)
    {
        if (i % 4 == 0) printf("\n");
        printf("%.2f ^ 3 = %.2f\t", h_input_data[i], h_output_data[i]);
    }

    free(h_input_data);
    free(h_output_data);
    cudaFree(d_input_data);
    cudaFree(d_output_data);

    return 0;
}