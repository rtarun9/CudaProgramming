#include <stdio.h>

#define ARRAY_LEN 64

__global__ void merge_sort(int* input_buffer, int* output_buffer, int num_elements_per_array)
{
    size_t left_array_start = threadIdx.x * num_elements_per_array* 2;
    size_t right_array_start = left_array_start + num_elements_per_array;

    size_t left_array_end = left_array_start + num_elements_per_array - 1;
    if (left_array_end >= ARRAY_LEN)
    {
        left_array_end = ARRAY_LEN - 1;
    }

    size_t right_array_end = right_array_start + num_elements_per_array - 1;
    if (right_array_end >= ARRAY_LEN)
    {
        right_array_end = ARRAY_LEN - 1;
    }


    size_t index = left_array_start;

    while (left_array_start <= left_array_end && right_array_start <= right_array_end)
    {
        if (input_buffer[left_array_start] < input_buffer[right_array_start])
        {
            output_buffer[index++] = input_buffer[left_array_start++];
        }
        else
        {
            output_buffer[index++] = input_buffer[right_array_start++];
        }
    }

    while (left_array_start <= left_array_end)
    {
        output_buffer[index++] = input_buffer[left_array_start++];
    }

    while (right_array_start<= right_array_end)
    {
        output_buffer[index++] = input_buffer[right_array_start++];
    }

}

int main()
{
    int* host_input_buffer = (int*)malloc(sizeof(int) * ARRAY_LEN);
    int* host_output_buffer = (int*)malloc(sizeof(int) * ARRAY_LEN);

    int* dev_input_buffer = nullptr;
    cudaMalloc(&dev_input_buffer, sizeof(int) * ARRAY_LEN);

    int* dev_output_buffer = nullptr;
    cudaMalloc(&dev_output_buffer, sizeof(int) * ARRAY_LEN);

    // Put some random data into input buffer.
    for (size_t i = 0; i < ARRAY_LEN; ++i)
    {
        host_input_buffer[i] = ARRAY_LEN - i;
    }

    cudaMemcpy(dev_input_buffer, host_input_buffer, sizeof(int) * ARRAY_LEN, cudaMemcpyKind::cudaMemcpyHostToDevice);

    size_t number_of_iterations = (size_t)log2(ARRAY_LEN);
    for (size_t i = 1; i <= number_of_iterations; i++)
    {
        const size_t number_of_threads_to_dispatch = (size_t)(ceil(ARRAY_LEN / powf(2.0f , i)));
        printf("iter index :: %d\tdispatching num threads :: %d\n", i, number_of_threads_to_dispatch);
        merge_sort<<<1u, number_of_threads_to_dispatch>>>(dev_input_buffer, dev_output_buffer, pow(2, i - 1));

        cudaMemcpy(host_output_buffer, dev_output_buffer, sizeof(int) * ARRAY_LEN, cudaMemcpyKind::cudaMemcpyDeviceToHost);


        if ((i != number_of_iterations))
        {
            int* temp = dev_input_buffer;
            dev_input_buffer = dev_output_buffer;
            dev_output_buffer = temp;
        }
    }

    printf("Input -> Output buffer");
    for (size_t i = 0; i < ARRAY_LEN; i++)
    {
        printf("%d -> %d\n", host_input_buffer[i], host_output_buffer[i]);
    }

    cudaFree(dev_output_buffer);
    cudaFree(dev_input_buffer);

    free(host_output_buffer);
    free(host_input_buffer);
}