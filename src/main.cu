#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "KeyValue.h"

#define MAX_LINES_FILE_READ 1024
#define EMITS_PER_LINE 10
#define MAX_EMITS (MAX_LINES_FILE_READ * EMITS_PER_LINE)

__host__ void loadFile(char fname[], KeyValuePair** kvs, int* length) {
	FILE* fp = fopen(fname, "r");
	if (fp == NULL)
	    exit(EXIT_FAILURE);

	char* line = NULL;
	size_t len = 0;
	int line_num = 0;
	while ((getline(&line, &len, fp)) != -1) {
	    //printf("%s", line);
	    kvs[line_num] = new KeyValuePair(line_num, line);
	    line_num ++;
	}
	fclose(fp);
	if (line)
	    free(line);
	*length = line_num;
}

__host__ __device__ void printKeyValues(KeyValuePair** kvs, int length) {
	for(int i = 0; i < length; i++) {
		if (kvs[i] == NULL) {
			printf("[%i = null]\n", i);
		} else {
			printf("%s \t %s\n", kvs[i]->key, kvs[i]->value);
		}
	}
}

__host__ __device__ void emit(KeyValuePair kv, KeyValuePair** out, int n) {
	out[n] = new KeyValuePair(kv);
}

__host__ __device__ void map(KeyValuePair kv, KeyValuePair** out, int n) {
	char* tokens = strtok(kv.value, " ,.-\t");
	int i = 0;
	while (tokens != NULL) {
		if (i >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			return;
		}
		emit(KeyValuePair(tokens, "1"), out, n + i);
		tokens = strtok(NULL, " ,.-\t");
		i++;
	}
}

__host__ void cpuMap(KeyValuePair** in, KeyValuePair** out, int length) {
	for (int i = 0; i < length; i++) {
		map(*in[i], out, i * EMITS_PER_LINE);
	}
}

__global__ void kernMap(KeyValuePair** in, KeyValuePair** out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;

	map(*in[i], out, i * EMITS_PER_LINE);
}

__host__ void reduce(int start, int end, KeyValuePair** in, KeyValuePair** out, int n) {
	char* key = in[start]->key;
	char value[50];
	sprintf(value, "%i", end-start);
	out[n] = new KeyValuePair(key, value);
}

__host__ void cpuReduce(KeyValuePair** in, KeyValuePair** out, int length) {
	if (in[0] == NULL) return;

	char* key = in[0]->key;
	int start = 0;
	int n = 0;
	for (int i = 0; i < length; i++) {
		if (in[i] == NULL || strcmp(key, in[i]->key) != 0) {
			reduce(start, i, in, out, n);
			if(in[i] == NULL) {
				return; //Sorted, so we must be at the end
			}

			key = in[i]->key;
			start = i;
			n++; //TODO this math doesn't work out, ensure we can't overflow keys
		}
	}
}

__host__ int main(int argc, char* argv[]) {
	std::cout << "Running\n";
	// Load file
	int length = 0;
	KeyValuePair* file_kvs[MAX_LINES_FILE_READ] = {NULL};
	loadFile("LICENSE", file_kvs, &length);
	//printf("Length: %i\n", length);
	//printKeyValues(kvs, length);

	// Map stage
	KeyValuePair* map_kvs[MAX_EMITS] = {NULL};
	cpuMap(file_kvs, map_kvs, length);
	//printKeyValues(map_kvs, MAX_EMITS);

	//Remove any null references (stream compaction)
	//TODO

	// Sort filtered map output
	/*
	KeyValuePair** dev_map_kvs;
	int sz = MAX_EMITS * sizeof(KeyValuePair*);
	cudaMalloc(&dev_map_kvs, sz);
	cudaMemcpy(dev_map_kvs, map_kvs, sz, cudaMemcpyHostToDevice);
	thrust::device_ptr<KeyValuePair> dev_ptr(*dev_map_kvs);
	thrust::sort(dev_ptr, dev_ptr + MAX_EMITS, KVComparator());
	*/
	std::sort(map_kvs, map_kvs + MAX_EMITS, KVComparator());
	//printKeyValues(map_kvs, MAX_EMITS);

	// Reduce stage
	KeyValuePair* reduce_kvs[MAX_EMITS] = {NULL};
	cpuReduce(map_kvs, reduce_kvs, MAX_EMITS);
	std::sort(reduce_kvs, reduce_kvs + MAX_EMITS, KVComparator());
	printKeyValues(reduce_kvs, MAX_EMITS);

	std::cout << "Done\n";
	return 0;
}
