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
#include <chrono>
#include "util.h"
#include "KeyValue.h"

#define MAX_LINES_FILE_READ 1024
#define EMITS_PER_LINE 10
#define MAX_EMITS (MAX_LINES_FILE_READ * EMITS_PER_LINE)
#define GPU_IMPLEMENTATION 1

#define WINDOWS 0
#define LINUX 1
#define COMPILE_OS WINDOWS

__host__ void loadFile(char fname[], KeyValuePair* kvs, int* length) {
#if COMPILE_OS == WINDOWS
	std::ifstream input(fname);
	int line_num = 0;
	for (std::string line; getline(input, line); )
	{
		char *cstr = new char[line.length() + 1];
		strcpy(cstr, line.c_str());
		itoa(line_num, kvs[line_num].key, 10);
		my_strcpy(kvs[line_num].value, cstr);
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

__host__ __device__ void printKeyValues(KeyValuePair* kvs, int length) {
	for(int i = 0; i < length; i++) {
		if (my_strlen(kvs[i].key) == 0) {
			//printf("[%i = null]\n", i);
		} else {
			printf("print key: %s \t value: %s\n", kvs[i].key, kvs[i].value);
		}
	}
}

__host__ __device__ void emit(KeyValuePair kv, KeyValuePair** out, int n) {
	//out[n] = new KeyValuePair(kv);
	
}

__host__ __device__ void map(KeyValuePair kv, KeyValuePair* out, int i, bool is_device) {
	//char* tokens = my_strtok(kv.value, " ,.-\t");
	//int i = 0;

	//while (tokens != NULL) {
	//	if (i >= EMITS_PER_LINE) {
	//		printf("WARN: Exceeded emit limit\n");
	//		return;
	//	}
	//	KeyValuePair curOut = out[n + i];
	//	my_strcpy(curOut.key, tokens);
	//	my_strcpy(curOut.value, "1");
	//	printf("out key: %c, value: %c", curOut.key, curOut.value);		
	//	tokens = my_strtok(NULL, " ,.-\t");
	//	i++;
	//}
	char* pSave = NULL;
	char* tokens = my_strtok_r(kv.value, " ,.-\t", &pSave);
	int count = 0;

	while (tokens != NULL) {
		if (count >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			break;
		}
		KeyValuePair* curOut = &out[i * EMITS_PER_LINE + count];
		my_strcpy(curOut->key, tokens);
		my_strcpy(curOut->value, "1");
		printf("out [%d][%d] key: %s, value: %s \n", i, count, curOut->key, curOut->value);
		tokens = my_strtok_r(NULL, " ,.-\t", &pSave);
		count++;
	}
	
}

__host__ void cpuMap(KeyValuePair* in, KeyValuePair* out, int length) {
	for (int i = 0; i < length; i++) {
		map(in[i], out, i * EMITS_PER_LINE, 0);
	}
}

__global__ void kernMap(KeyValuePair* in, KeyValuePair* out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	//printf("%d", i);
	printf("kernMap input key: %s, value: %s\n", in[i].key, in[i].value);
	map(in[i], out, i, 1);
	printf("kernMap output key: %s, value: %s\n", out[i].key, out[i].value);
}

__host__ void reduce(int start, int end, KeyValuePair** in, KeyValuePair** out, int n) {
	char* key = in[start]->key;
	char value[50];
	sprintf(value, "%i", end-start);
	//out[n] = new KeyValuePair(key, value);
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

//// host_array the array of KV pointers, should be sorted and NULL terminated
//// dev_array will be newly allocated and the pointer returned
//__host__ KeyValuePair** copyKVPairToCuda(KeyValuePair** host_array, int length) {
//	// Step 1: Create host array of device pointers
//	KeyValuePair* host_of_device[MAX_LINES_FILE_READ] = { NULL };
//	int i = 0;
//	while (host_array[i] != NULL && i < length) {
//		host_of_device[i] = host_array[i]->to_device();
//		i++;
//	}
//
//	// Step 2: Move host array to device
//	KeyValuePair** dev_kvs;
//	int sz = sizeof(KeyValuePair**) * length;
//	cudaMalloc(&dev_kvs, sz);
//	cudaMemcpy(dev_kvs, host_of_device, sz, cudaMemcpyHostToDevice);
//	return dev_kvs;
//}
//
//__host__ __device__ KeyValuePair** copyKVPairFromCuda(KeyValuePair** dev_array, int length) {
//	// Step 1: Create device array of host pointers
//	
//	KeyValuePair* device_of_host[MAX_EMITS] = { NULL };
//	int i = 0;
//	while (i < length) {
//		printf("%d \n", i);
//		dev_array[i]->test();
//		printf("%d \n", i);
//		//device_of_host[i] = dev_array[i]->to_host();
//		i++;
//	}
//	
//	// Step 2: Move device array to host
//	KeyValuePair** host_kvs;
//	int sz = sizeof(KeyValuePair**) * length;
//	host_kvs = (KeyValuePair**)malloc(sz);
//	cudaMemcpy(host_kvs, device_of_host, sz, cudaMemcpyDeviceToHost);
//	return host_kvs;
//}

__host__ int main(int argc, char* argv[]) {
	typedef std::chrono::high_resolution_clock Clock;

	std::cout << "Running\n";
	// Load file
	int length = 0;
	KeyValuePair file_kvs[MAX_LINES_FILE_READ] = {NULL};
	loadFile("LICENSE", file_kvs, &length);
	printf("Length: %i\n", length);
	
	//char str[] = "- This, a sample string.";
	//char * pch;
	//printf("Splitting string \"%s\" into tokens:\n", str);
	//pch = my_strtok(str, " ,.-");
	//while (pch != NULL)
	//{
	//	printf("%s\n", pch);
	//	pch = my_strtok(NULL, " ,.-");
	//}

	//char value[100] = "- This, a sample string.";
	//char* tokens;
	//tokens = my_strtok(value, " ,.-");

	//while (tokens != NULL) {
	//	printf("tokens is %s \n", tokens);
	//	tokens = my_strtok(NULL, " ,.-");
	//}

	//printKeyValues(file_kvs, length);

	// Map stage
	//KeyValuePair* map_kvs[MAX_EMITS] = {NULL};
	
	//printKeyValues(map_kvs, MAX_EMITS);

	//Remove any null references (stream compaction)
	//TODO
#if GPU_IMPLEMENTATION
	// Sort filtered map output
	KeyValuePair* dev_file_kvs = NULL;
	cudaMalloc((void **)&dev_file_kvs, MAX_LINES_FILE_READ * sizeof(KeyValuePair));
	cudaMemcpy(dev_file_kvs, file_kvs, MAX_LINES_FILE_READ * sizeof(KeyValuePair), cudaMemcpyHostToDevice);

	KeyValuePair* dev_map_kvs = NULL;
	cudaMalloc((void **)&dev_map_kvs, MAX_EMITS * sizeof(KeyValuePair));
	kernMap << <1, 128 >> > (dev_file_kvs, dev_map_kvs, length);

	KeyValuePair* map_kvs = NULL;
	map_kvs = (KeyValuePair*)malloc(MAX_EMITS * sizeof(KeyValuePair));
	cudaMemcpy(map_kvs, dev_map_kvs, MAX_EMITS * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
	//printKeyValues(map_kvs, MAX_EMITS);
	KeyValuePair* test_file_kvs = NULL;
	test_file_kvs = (KeyValuePair*)malloc(MAX_LINES_FILE_READ * sizeof(KeyValuePair));
	cudaMemcpy(test_file_kvs, dev_file_kvs, MAX_LINES_FILE_READ * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
	//printKeyValues(test_file_kvs, MAX_LINES_FILE_READ);
	free(test_file_kvs);
	cudaFree(dev_file_kvs);
	cudaFree(dev_map_kvs);
	//KeyValuePair* null_array[MAX_EMITS];
	//for (int i = 0; i < MAX_EMITS; i++) {
	//	null_array[i] = new KeyValuePair(1);
	//}
 
	//KeyValuePair** dev_map_kvs;
	//int sz = MAX_EMITS * sizeof(KeyValuePair*);
	//cudaMalloc(&dev_map_kvs, sz);
	//cudaMemcpy(dev_map_kvs, null_array, sz, cudaMemcpyHostToDevice);
	//auto t0 = Clock::now();
	//kernMap << <1024, 1024 >> > (dev_file_kvs, dev_map_kvs, length);
	//auto t1 = Clock::now();
	//printf("%d nanoseconds \n", t1 - t0);

	//KeyValuePair** host_map_kvs = copyKVPairFromCuda(dev_map_kvs, MAX_EMITS);
	//printKeyValues(host_map_kvs, length);
	//thrust::device_ptr<KeyValuePair*> dev_ptr(dev_map_kvs);
	//thrust::sort(dev_ptr, dev_ptr + MAX_EMITS, KVComparator());
	// Can't print these, it's mapped to device ptrs
	//printKeyValues(dev_map_kvs, length);
	// TODO: This doesn't free the actual KV objects, just the arrays.
	//cudaFree(dev_file_kvs);
	//cudaFree(dev_map_kvs);
	//*/
	
#else
	auto t0 = Clock::now();
	cpuMap(file_kvs, map_kvs, length);
	auto t1 = Clock::now();
	printf("%d nanoseconds \n", t1 - t0);
	std::sort(map_kvs, map_kvs + MAX_EMITS, KVComparator());

	// Reduce stage
	KeyValuePair* reduce_kvs[MAX_EMITS] = {NULL};
	cpuReduce(map_kvs, reduce_kvs, MAX_EMITS);
	std::sort(reduce_kvs, reduce_kvs + MAX_EMITS, KVComparator());
	//printKeyValues(reduce_kvs, MAX_EMITS);
#endif
	
	std::cout << "\nDone\n";
	return 0;
}
