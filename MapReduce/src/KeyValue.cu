#include <string.h>
#include <stdio.h>
#include <stdlib.h>  
#include <cuda_runtime.h>

#include "KeyValue.h"

KeyValuePair::KeyValuePair() {
	key = NULL;
	value = NULL;
	is_empty = 1;
}

KeyValuePair::KeyValuePair(int k_num, char* v) {
	char str[100];
	sprintf(str, "%i", k_num);
	set(str, v);
}

KeyValuePair::KeyValuePair(char* k, char* v) {
	set(k, v);
}

void KeyValuePair::set(char* k, char* v) {
	char* key_ptr = (char*) malloc(sizeof(char) * strlen(k) + 1);
	strcpy(key_ptr, k);
	char* val_ptr = (char*) malloc(sizeof(char) * strlen(v) + 1);
	strcpy(val_ptr, v);

	key = key_ptr;
	value = val_ptr;
	is_empty = 0;
}

KeyValuePair* KeyValuePair::to_device() {
	char* dev_k = NULL;
	cudaMalloc(&dev_k, sizeof(char) * strlen(key));
	cudaMemcpy(dev_k, key, sizeof(char) * strlen(key), cudaMemcpyHostToDevice);
	char *dev_v = NULL;
	cudaMalloc(&dev_v, sizeof(char) * strlen(value));
	cudaMemcpy(dev_v, value, sizeof(char) * strlen(value), cudaMemcpyHostToDevice);

	KeyValuePair* dev_kv = NULL;
	cudaMalloc((void**)&dev_kv, sizeof(KeyValuePair));
	KeyValuePair tmp_kv = KeyValuePair();
	tmp_kv.key = dev_k;
	tmp_kv.value = dev_v;
	tmp_kv.is_empty = is_empty;
	cudaMemcpy(dev_kv, &tmp_kv, sizeof(KeyValuePair), cudaMemcpyHostToDevice);

	return dev_kv;
}

void KeyValuePair::to_string(const KeyValuePair* kv, char* s) {
	sprintf(s, "Key: %s | Value: %s", kv->key, kv->value);
}

bool KVComparator::operator() (const KeyValuePair *kv1, const KeyValuePair *kv2) {
	if (!kv1 || !kv1->key) {
		return false;
	} else if (!kv2 || !kv2->key) {
		return true;
	}

	int i = 0;
	while(1) {
		if (kv1->key[i] != kv2->key[i]) {
			return kv1->key[i] < kv2->key[i];
		}
		i++;
	}
}
