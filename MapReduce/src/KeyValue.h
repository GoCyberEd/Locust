#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "util.h"

struct KeyValuePair {
public:
	char key[100];
	char value[100];
	int ind;
};

struct KeyIntValuePair {
public:
	char key[30];
	int value = 0;
	int count = 0;
};

class KIVComparator {
public:
	__host__ __device__ bool operator() (const KeyIntValuePair& kv1, const KeyIntValuePair& kv2) {
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

class KeyIntValueZero {
public:
	__host__ __device__ bool operator()(const KeyIntValuePair& kvp) {
		return kvp.value == 0;
	}
};
