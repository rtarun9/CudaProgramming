#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>

void print_array(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("array[%d] = %d.\n", i, array[i]);
    }
}

void print_matrix(int* array, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%d ", array[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fill_with_increment(int* array, int size, int increment)
{
    array[0] = 1;

    for (int i = 1; i < size; i++)
    {
        array[i] = array[i - 1] + increment;
    }
}

#endif