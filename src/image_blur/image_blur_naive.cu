#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../external/stb_image_write.h"

// Naive blur : Load from global memory and directly write to the output image.
// Assume the loaded image is in grayscale.
__global__ void naive_blur(const unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    const int t_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int t_y = threadIdx.y + blockIdx.y * blockDim.y;

    const size_t pixel_index = t_x + t_y * width;

    float pixel_sum = 0.0f;
    for (int i = -2; i <= 2; i++)
    {
        for (int j = -2; j <= 2; j++)
        {
            pixel_sum += input_image[(t_x + j) + (t_y + i) * width];
        }
    }

    output_image[pixel_index] = (unsigned char)(pixel_sum / 25.0f);

    return;
}


int main()
{
    // First, read the source image and extract relavant data.
    int width = 0;
    int height = 0;
    unsigned char* h_input_image_data = stbi_load("../../assets/images/test_image_grayscale.png", &width, &height, nullptr, 1);

    printf("Image width and height : %d %d\n", width, height);

    // Allocate memory for the output (host) data and input and output (device) data.
    const size_t GRAY_SCALE_IMAGE_BYTES = sizeof(unsigned char) * width * height;

    unsigned char* h_output_image_data = (unsigned char*)malloc(GRAY_SCALE_IMAGE_BYTES);

    unsigned char* d_input_image_data = nullptr;
    unsigned char* d_output_image_data = nullptr;

    cudaMalloc(&d_input_image_data, GRAY_SCALE_IMAGE_BYTES);
    cudaMalloc(&d_output_image_data, GRAY_SCALE_IMAGE_BYTES);

    cudaMemcpy(d_input_image_data, h_input_image_data, GRAY_SCALE_IMAGE_BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Launch kernel.
    // Each thread block will be of 8 x 8 threads. Based on input image, find the number of blocks to launch.
    const dim3 block_dim = dim3(8, 8, 1);
    const dim3 grid_dim = dim3((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y, 1);

    printf("Block dim :: %d %d %d\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Grid dim :: %d %d %d\n", grid_dim.x, grid_dim.y, grid_dim.z);

    // Naive blur.
    {
        naive_blur<<<grid_dim, block_dim>>>(d_input_image_data, d_output_image_data, width, height);

        // Copy output to host memory.
        cudaMemcpy(h_output_image_data, d_output_image_data, GRAY_SCALE_IMAGE_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        // Write output in image format (with file name : output_image_grayscale.png).
        const size_t output_image_row_stride = sizeof(unsigned char) * 1 * width;
        if (stbi_write_png("../../assets/images/naive_blur.png", width, height, 1, h_output_image_data, output_image_row_stride))
        {
            printf("Successfully wrote output image to ../assets/images/naive_blur.png");
        }
        else
        {
            printf("Failed to write to output image");
        }
    }

    stbi_image_free(h_input_image_data);
    free(h_output_image_data);

    cudaFree(d_input_image_data);
    cudaFree(d_output_image_data);
}