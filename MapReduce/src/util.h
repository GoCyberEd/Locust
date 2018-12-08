#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ int my_strlen(const char * str);
__host__ __device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len = 256);
__host__ __device__ char * my_strcpy(char *strDest, const char *strSrc);
__host__ __device__ char* my_strtok(char *str, const char* delim);
__host__ __device__ char * my_strtok_r(char * __restrict s, const char * __restrict delim, char ** __restrict last);
__host__ __device__ void my_reverse(char str[], int length);
__host__ __device__ char* my_itoa(int num, char* str, int base);
