#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../external/stb_image_write.h"

// The goal is simple : Keep same number of blocks, but increase the threads per block.
// This WILL create lot of overlap, which is what we want so as to reduce thread divergence & extra memory fetches to fill shared memory.
// This is done so that all data can be loaded into shared memory without any extra divergence for border / corner conditions.
#define FILTER_DIM 5
#define BLOCK_DIM_WITHOUT_PADDING 8
#define BLOCK_DIM (BLOCK_DIM_WITHOUT_PADDING + FILTER_DIM)
#define SMEM_ARRAY_DIM (BLOCK_DIM)

__global__ void shared_mem_blur(const unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    __shared__ int smem_pixel_values[SMEM_ARRAY_DIM][SMEM_ARRAY_DIM];

    // The pixel location in output image this thread will write to (if eligible).
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;

    // In the overlapping thread block, thes shifted thread index -> image pixel relationship is:
    const int shifted_t_x = t_x - FILTER_DIM / 2;
    const int shifted_t_y = t_y - FILTER_DIM / 2;

    const int output_pixel_x = shifted_t_x + blockIdx.x * (BLOCK_DIM_WITHOUT_PADDING);
    const int output_pixel_y= shifted_t_y + blockIdx.y * (BLOCK_DIM_WITHOUT_PADDING);

    // Write to shared memory.
    // Note that some values might go OOB, so that case must be handled.
    if (output_pixel_x< 0 || output_pixel_x > width || output_pixel_y < 0 || output_pixel_y > height)
    {
        smem_pixel_values[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
        smem_pixel_values[threadIdx.y][threadIdx.x] = input_image[output_pixel_x+ output_pixel_y* (width)];
    }
    __syncthreads();

    if (shifted_t_x >= 0 && shifted_t_x < BLOCK_DIM_WITHOUT_PADDING && shifted_t_y >= 0 && shifted_t_y < BLOCK_DIM_WITHOUT_PADDING)
    {
        float pixel_sum = 0.0f;
        for (int i = -FILTER_DIM/2; i <= FILTER_DIM/2; i++)
        {
            for (int j = -FILTER_DIM/2; j <= FILTER_DIM/2; j++)
            {
                pixel_sum += smem_pixel_values[threadIdx.y+ i][threadIdx.x+ j];
            }
        }
        output_image[output_pixel_x+ output_pixel_y* width] = pixel_sum / ((float)FILTER_DIM * FILTER_DIM);
    }

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
    // Number of blocks will be as expected, note that number of threads per block increases.
    // This is so that all data can be loaded into shared memory without much divergence (as for edge pixels you need data of pixels that might be computed in other blocks).
    const dim3 block_dim = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 grid_dim = dim3((width + BLOCK_DIM_WITHOUT_PADDING - 1) / BLOCK_DIM_WITHOUT_PADDING, (height + BLOCK_DIM_WITHOUT_PADDING - 1) / BLOCK_DIM_WITHOUT_PADDING, 1);

    printf("Block dim :: %d %d %d\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Grid dim :: %d %d %d\n", grid_dim.x, grid_dim.y, grid_dim.z);

    // shared mem blur.
    {
        shared_mem_blur<<<grid_dim, block_dim>>>(d_input_image_data, d_output_image_data, width, height);

        // Copy output to host memory.
        cudaMemcpy(h_output_image_data, d_output_image_data, GRAY_SCALE_IMAGE_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        // Write output in image format .
        const size_t output_image_row_stride = sizeof(unsigned char) * 1 * width;
        if (stbi_write_png("../../assets/images/shared_mem_blur.png", width, height, 1, h_output_image_data, output_image_row_stride))
        {
            printf("Successfully wrote output image to ../assets/images/shared_mem_blur.png");
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