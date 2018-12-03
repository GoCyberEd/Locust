#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class KeyValuePair {
	public:
		__host__ KeyValuePair();
		__host__ KeyValuePair(int k_num, char* v);
		__host__ KeyValuePair(char* k, char* v);
		__host__ void set(char* k, char* v);
		__host__ static void to_string(const KeyValuePair* kv, char* s);
		char* key;
		char* value;
};

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
