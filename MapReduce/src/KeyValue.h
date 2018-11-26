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
	__host__ bool operator() (const KeyValuePair *kv1, const KeyValuePair *kv2);
};
