#pragma once
#include <cuda_runtime.h>

__host__ __device__ int my_strlen(const char * str);
__host__ __device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len = 256);
__host__ __device__ char * my_strcpy(char *strDest, const char *strSrc);
__host__ __device__ char* my_strtok(char *str, const char* delim);
