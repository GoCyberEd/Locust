#pragma once

class KeyValuePair {
	public:
		KeyValuePair();
		KeyValuePair(int k_num, char* v);
		KeyValuePair(char* k, char* v);
		void set(char* k, char* v);
		char* key;
		char* value;
};

class KVComparator {
public:
	__host__ __device__ bool operator() (const KeyValuePair &kv1, const KeyValuePair &kv2) {
		return kv1.key < kv2.key;
	}
};
