#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

struct KeyValuePair {
	public:
		__host__ __device__ KeyValuePair();
		__host__ __device__ KeyValuePair(bool is_device);
		__host__ KeyValuePair(int k_num, char* v);
		__host__ KeyValuePair(char* k, char* v);
		__host__ __device__ KeyValuePair(char* k, char* v, bool is_device);
		__host__ __device__ void set(char* k, char* v);
		__host__ KeyValuePair* to_device();
		__host__ __device__ KeyValuePair* to_host();
		__host__ static void to_string(const KeyValuePair* kv, char* s);

		__host__ __device__ void test();
		char* key;
		char* value;
		bool is_device;
};

//struct KeyValuePair {
//	char* key;
//	char* value;
//};

class KVComparator {
public:
	__host__ __device__ bool operator() (const KeyValuePair *kv1, const KeyValuePair *kv2);
	/*
	__host__ __device__ bool operator() (const KeyValuePair &kv1, const KeyValuePair &kv2) {
		// The signatures are different, we need to sort an array of pointers, not an array of values
		unsigned char *temp1 = (unsigned char *) &(kv1.key);
		unsigned char *temp2 = (unsigned char *) &(kv2.key);
		while (*temp1 && *temp2) {
			if (*temp1 == *temp2) {
				temp1++;
				temp2++;
			}
			else {
				if (*temp1 < *temp2) {
					return false;
				}
				else {
					return true;
				}
			}
		}
		return false;
	}
	*/
};
