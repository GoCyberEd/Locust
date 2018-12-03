#define _GNU_SOURCE
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "util.h"
#include "KeyValue.h"

#define MAX_LINES_FILE_READ 1024
#define EMITS_PER_LINE 10
#define MAX_EMITS (MAX_LINES_FILE_READ * EMITS_PER_LINE)
#define GPU_IMPLEMENTATION 1

#define WINDOWS 0
#define LINUX 1
#define COMPILE_OS LINUX

__host__ void loadFile(char fname[], KeyValuePair** kvs, int* length) {
#if COMPILE_OS == WINDOWS
	std::ifstream input(fname);
	int line_num = 0;
	for (std::string line; getline(input, line); )
	{
		char *cstr = new char[line.length() + 1];
		strcpy(cstr, line.c_str());
		kvs[line_num] = new KeyValuePair(line_num, cstr);
		line_num++;
		delete[] cstr;
	}
	*length = line_num;
#elif COMPILE_OS == LINUX
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
#endif
}

__host__ __device__ void printKeyValues(KeyValuePair** kvs, int length) {
	for(int i = 0; i < length; i++) {
		if (kvs[i] == NULL) {
			//printf("[%i = null]\n", i);
		} else {
			printf("%s \t %s\n", kvs[i]->key, kvs[i]->value);
		}
	}
}

__host__ __device__ void emit(KeyValuePair kv, KeyValuePair** out, int n) {
	out[n] = new KeyValuePair(kv);
}

__host__ __device__ void map(KeyValuePair kv, KeyValuePair** out, int n) {
	char* tokens = my_strtok(kv.value, " ,.-\t");
	int i = 0;
	while (tokens != NULL) {
		if (i >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			return;
		}
		emit(KeyValuePair(tokens, "1"), out, n + i);
		tokens = my_strtok(NULL, " ,.-\t");
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
		if (in[i] == NULL || my_strcmp(key, in[i]->key) != 0) {
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

void GPUMapReduce(KeyValuePair* map_kvs, int length) {
	KeyValuePair** dev_map_kvs;
	int sz = MAX_EMITS * sizeof(KeyValuePair*);
	cudaMalloc(&dev_map_kvs, sz);
	cudaMemcpy(dev_map_kvs, map_kvs, sz, cudaMemcpyHostToDevice);
	//kernMap << <1024, 1024 >> > (map_kvs, dev_map_kvs, length);
	//thrust::device_ptr<KeyValuePair> dev_ptr(*dev_map_kvs);
	//thrust::sort(dev_ptr, dev_ptr + MAX_EMITS, KVComparator());
	printKeyValues(dev_map_kvs, length);
	cudaFree(dev_map_kvs);
}

// host_array the array of KV pointers, should be sorted and NULL terminated
// dev_array will be newly allocated and the pointer returned
__host__ KeyValuePair** copyKVPairToCuda(KeyValuePair** host_array, int length) {
	/*
	// Allocate memory on device
	KeyValuePair** dev_map_kvs;
	int sz = sizeof(KeyValuePair**) * length;
	cudaMalloc(&dev_map_kvs, sz);

	// For each element, copy the actual KeyValue object into cuda memory and update
	// reference in list
	int i = 0;
	while (host_array[i] != NULL) {
		cudaMalloc(&dev_map_kvs[i], sizeof(KeyValuePair));
		// Copy actual object
		// host_array[i] may need to be dereferenced?
		cudaMemcpy(dev_map_kvs[i], host_array[i], sizeof(KeyValuePair), cudaMemcpyHostToDevice);
		i++;
	} */
	// Step 1: Create host array of device pointers
	KeyValuePair* host_of_device[length] = { NULL };
	int i = 0;
	while (host_array[i] != NULL && i < length) {
		host_of_device[i] = host_array[i]->to_device();
		i++;
	}

	// Step 2: Move host array to device
	KeyValuePair** dev_kvs;
	int sz = sizeof(KeyValuePair**) * length;
	cudaMalloc(&dev_kvs, sz);
	cudaMemcpy(dev_kvs, host_of_device, sz, cudaMemcpyHostToDevice);

	return dev_kvs;
}

__host__ int main(int argc, char* argv[]) {
	std::cout << "Running\n";
	// Load file
	int length = 0;
	KeyValuePair* file_kvs[MAX_LINES_FILE_READ] = {NULL};
	loadFile("../LICENSE", file_kvs, &length);
	//printf("Length: %i\n", length);
	//printKeyValues(kvs, length);

	// Map stage
	KeyValuePair* map_kvs[MAX_EMITS] = {NULL};
	
	//printKeyValues(map_kvs, MAX_EMITS);

	//Remove any null references (stream compaction)
	//TODO
#if GPU_IMPLEMENTATION
	// Sort filtered map output
	KeyValuePair** dev_file_kvs = copyKVPairToCuda(file_kvs, MAX_LINES_FILE_READ);

	KeyValuePair* null_array[MAX_EMITS] = { NULL };

	KeyValuePair** dev_map_kvs;
	int sz = MAX_EMITS * sizeof(KeyValuePair*);
	cudaMalloc(&dev_map_kvs, sz);
	cudaMemcpy(dev_map_kvs, null_array, sz, cudaMemcpyHostToDevice);
	kernMap << <1024, 1024 >> > (dev_file_kvs, dev_map_kvs, length);
	thrust::device_ptr<KeyValuePair*> dev_ptr(dev_map_kvs);
	thrust::sort(dev_ptr, dev_ptr + MAX_EMITS, KVComparator());
	// Can't print these, it's mapped to device ptrs
	//printKeyValues(dev_map_kvs, length);
	// TODO: This doesn't free the actual KV objects, just the arrays.
	cudaFree(dev_file_kvs);
	cudaFree(dev_map_kvs);
	//*/
	
#else
	cpuMap(file_kvs, map_kvs, length);
	std::sort(map_kvs, map_kvs + MAX_EMITS, KVComparator());

	// Reduce stage
	KeyValuePair* reduce_kvs[MAX_EMITS] = {NULL};
	cpuReduce(map_kvs, reduce_kvs, MAX_EMITS);
	std::sort(reduce_kvs, reduce_kvs + MAX_EMITS, KVComparator());
	printKeyValues(reduce_kvs, MAX_EMITS);
#endif
	
	std::cout << "\nDone\n";
	return 0;
}
