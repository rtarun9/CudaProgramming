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
#define BLOCK_DIM (8 + FILTER_DIM)
#define SMEM_ARRAY_DIM (BLOCK_DIM)

__global__ void shared_mem_blur(const unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    __shared__ int smem_pixel_values[SMEM_ARRAY_DIM][SMEM_ARRAY_DIM];

    const int t_x =  threadIdx.x + blockIdx.x * blockDim.x;
    const int t_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Now, only some of these threads map to valid pixels in the image.
    // For this, first shift the value of t_x and t_y by FILTER_DIM/2.
    const int shifted_thread_x = threadIdx.x- FILTER_DIM /2;
    const int shifted_thread_y = threadIdx.y- FILTER_DIM /2;

    // Now, use these values and find output image pixel locations that map to it.
    const int output_thread_x = shifted_thread_x + blockIdx.x * BLOCK_DIM_WITHOUT_PADDING;
    const int output_thread_y = shifted_thread_y+ blockIdx.y * BLOCK_DIM_WITHOUT_PADDING;

    if (output_thread_x >= 0 && output_thread_x < width && output_thread_y >= 0 && output_thread_y < height)
    {
        output_image[output_thread_x+ output_thread_y* width] = input_image[output_thread_x+ output_thread_y* width];
    }
    return;

    // Global index of threads (including overlapped regions).
    const int overlapped_x =  threadIdx.x + blockIdx.x * blockDim.x;
    const int overlapped_y = threadIdx.y + blockIdx.y * blockDim.y;

    const int output_t_x =  threadIdx.x + blockIdx.x * blockDim.x;
    const int output_t_y =  threadIdx.y + blockIdx.y * blockDim.y;

    const int input_image_width = width + FILTER_DIM;

    const int smem_index_x = threadIdx.x;
    const int smem_index_y = threadIdx.y; 

    smem_pixel_values[smem_index_y][smem_index_x] = input_image[t_x + t_y * input_image_width];
    __syncthreads();

    // For border threads (in the block), load the additional data into shared memory.
    if (threadIdx.x == 0)
    {
        for (int i = 1; i <= FILTER_DIM / 2; i++)
        {
            smem_pixel_values[smem_index_y][smem_index_x - i] = input_image[t_x - i + t_y * input_image_width];
        }
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        for (int i = 1; i <= FILTER_DIM / 2; i++)
        {
            smem_pixel_values[smem_index_y][smem_index_x + i] = input_image[t_x + i + t_y * input_image_width];
        }

    }

    if (threadIdx.y == blockDim.x - 1)
    {
        for (int i = 1; i <= FILTER_DIM / 2; i++)
        {
            smem_pixel_values[smem_index_y + i][smem_index_x] = input_image[t_x +  (t_y + i) * input_image_width];
        }

    }

    if (threadIdx.y == 0)
    {
        for (int i = 1; i <= FILTER_DIM / 2; i++)
        {
            smem_pixel_values[smem_index_y - i][smem_index_x] = input_image[t_x +  (t_y - i) * input_image_width];
        }
    }

    __syncthreads();

    // NOTE : Remove this once a suitable solution is found! Special edge case for teh corner pixels (4 per block)
    bool corner_condition = (threadIdx.x == 0 && threadIdx.y == 0 || threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 || threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 || threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1);
    if (corner_condition)
    {
        for (int i = -FILTER_DIM / 2; i <= FILTER_DIM / 2; i++)
        {
            for (int j = -FILTER_DIM/2; j <= FILTER_DIM/2; j++)
            {
                smem_pixel_values[smem_index_y + i][smem_index_x + j] = input_image[t_x + j +  (t_y + i) * input_image_width];
            }
        }
    }
    const size_t pixel_index = output_t_x+ output_t_y * width;

    float pixel_sum = 0.0f;
    for (int i = -FILTER_DIM/2; i <= FILTER_DIM/2; i++)
    {
        for (int j = -FILTER_DIM/2; j <= FILTER_DIM/2; j++)
        {
            pixel_sum += smem_pixel_values[smem_index_y + i][smem_index_x+ j];
        }
    }

    output_image[pixel_index] = (unsigned char)(pixel_sum / (float)(FILTER_DIM * FILTER_DIM));

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
    const dim3 grid_dim = dim3((width + BLOCK_DIM_WITHOUT_PADDING - 1) / block_dim.x, (height + BLOCK_DIM_WITHOUT_PADDING - 1) / block_dim.y, 1);

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