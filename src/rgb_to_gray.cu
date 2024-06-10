#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

__global__ void rgb_to_grayscale(const unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    const int t_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int t_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Why 4 writes are being made? Because each thread maps to one pixel, but in input image there are 4 unsigned chars representing each pixel.
    // In the output image, a single pixel corresponds to only one unsigned char, the grayscale value.
    const size_t output_image_pixel_index = t_x + t_y * width;
    const size_t input_image_pixel_red_component_index = output_image_pixel_index * 4; // a 1d pixel index that represents the first (red component) of pixel this thread is computing value for.

    const unsigned char input_image_r = input_image[input_image_pixel_red_component_index + 0];
    const unsigned char input_image_g = input_image[input_image_pixel_red_component_index + 1];
    const unsigned char input_image_b = input_image[input_image_pixel_red_component_index + 2];

    output_image[output_image_pixel_index] = (unsigned char)((input_image_r * 0.3 + 0.59 * input_image_g + input_image_b * 0.11));

    return;
}

int main()
{
    // First, read the source image and extract relavant data.
    int width = 0;
    int height = 0;
    unsigned char* h_input_image_data = stbi_load("../assets/images/test_image_rgb.png", &width, &height, nullptr, 4);

    printf("Image width and height : %d %d\n", width, height);

    // Allocate memory for the output (host) data and input and output (device) data.
    const size_t RGB_IMAGE_BYTES = sizeof(unsigned char) * 4 * width * height;
    const size_t GRAY_SCALE_IMAGE_BYTES = sizeof(unsigned char) * width * height;

    unsigned char* h_output_image_data = (unsigned char*)malloc(GRAY_SCALE_IMAGE_BYTES);

    unsigned char* d_input_image_data = nullptr;
    unsigned char* d_output_image_data = nullptr;

    cudaMalloc(&d_input_image_data, RGB_IMAGE_BYTES);
    cudaMalloc(&d_output_image_data, GRAY_SCALE_IMAGE_BYTES);

    cudaMemcpy(d_input_image_data, h_input_image_data, RGB_IMAGE_BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Launch kernel.
    // Each thread block will be of 16 x 61 threads. Based on input image, find the number of blocks to launch.
    const dim3 block_dim = dim3(16, 16, 1);
    const dim3 grid_dim = dim3((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y, 1);

    printf("Block dim :: %d %d %d\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Grid dim :: %d %d %d\n", grid_dim.x, grid_dim.y, grid_dim.z);

    rgb_to_grayscale<<<grid_dim, block_dim>>>(d_input_image_data, d_output_image_data, width, height);

    // Copy output to host memory.
    cudaMemcpy(h_output_image_data, d_output_image_data, GRAY_SCALE_IMAGE_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // Write output in image format (with file name : output_image_grayscale.png).
    const size_t output_image_row_stride = sizeof(unsigned char) * 1 * width;
    if (stbi_write_png("../assets/images/test_output_image_grayscale.png", width, height, 1, h_output_image_data, output_image_row_stride))
    {
        printf("Successfully wrote output image to ../assets/images/test_output_image_grayscale.png");
    }
    else
    {
        printf("Failed to write to output image");
    }

    stbi_image_free(h_input_image_data);
    free(h_output_image_data);

    cudaFree(d_input_image_data);
    cudaFree(d_output_image_data);
}