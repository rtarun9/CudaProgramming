// A test file to see in what memory patters / techniques bank conflicts are as low as possible.

#include <stdio.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET_1(n) (n / NUM_BANKS) 
#define CONFLICT_FREE_OFFSET_2(n) ((n / NUM_BANKS) + (n / (LOG_NUM_BANKS * 2)))

__global__ void naive_bank_accesses()
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("[Naive] Thread idx : %d accesses bank : %d\n", tx, tx % NUM_BANKS);
}

__global__ void calculated_offset_bank_accesses_1()
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("[Calcluated_1] Thread idx : %d accesses bank : %d\n", tx, (tx + CONFLICT_FREE_OFFSET_1(tx)) % NUM_BANKS);
}

__global__ void calculated_offset_bank_accesses_2()
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("[Calcluated_2] Thread idx : %d accesses bank : %d\n", tx, (tx + CONFLICT_FREE_OFFSET_2(tx)) % NUM_BANKS);
}

int main()
{
    naive_bank_accesses<<<1, 32>>>();
    calculated_offset_bank_accesses_1<<<1, 32>>>();
    calculated_offset_bank_accesses_2<<<1, 32>>>();
    return 0;
}