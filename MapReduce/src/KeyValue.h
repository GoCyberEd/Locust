#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "util.h"

struct KeyValuePair {
	public:
		//__host__ __device__ KeyValuePair();
		//__host__ __device__ KeyValuePair(bool is_device);
		//__host__ KeyValuePair(int k_num, char* v);
		//__host__ KeyValuePair(char* k, char* v);
		//__host__ __device__ KeyValuePair(char* k, char* v, bool is_device);
		//__host__ __device__ void set(char* k, char* v);
		//__host__ KeyValuePair* to_device();
		//__host__ __device__ KeyValuePair* to_host();
		//__host__ static void to_string(const KeyValuePair* kv, char* s);

		//__host__ __device__ void test();
		char key[100];
		char value[100];
		int ind;
		//bool is_device;
};

struct KeyIntValuePair {
public:
	char key[30];
	int value;
	int count;
};

//struct KeyValuePair {
//	char* key;
//	char* value;
//};

class KVComparator {
public:
	__host__ __device__ bool operator() (const KeyValuePair& kv1, const KeyValuePair& kv2) {
		unsigned char *temp1 = (unsigned char *) &(kv1.key);
		unsigned char *temp2 = (unsigned char *) &(kv2.key);
		int i = 0;
		while (1) {
			if (temp1[i] != temp2[i]) {
				return temp1[i] < temp2[i];
			}
			i++;
		}
	}
};

class KVComparatorCPU {
public:
	__host__ __device__ bool operator() (const KeyValuePair *kv1, const KeyValuePair *kv2) {
		if (!kv1 || !kv1->key) {
			return false;
		}
		else if (!kv2 || !kv2->key) {
			return true;
		}

		int i = 0;
		while (1) {
			if (kv1->key[i] != kv2->key[i]) {
				return kv1->key[i] < kv2->key[i];
			}
			i++;
		}
	}
};



class KeyValueNotEmpty {
public:
	__host__ __device__ bool operator()(const KeyValuePair& kvp) {
		return my_strlen(kvp.key) > 0;
	}
};

class KeyIntValueNotEmpty {
public:
	__host__ __device__ bool operator()(const KeyIntValuePair& kvp) {
		return my_strlen(kvp.key) > 0;
	}
};
