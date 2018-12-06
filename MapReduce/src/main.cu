#define _GNU_SOURCE
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
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
		my_strcpy(cstr, line.c_str());
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

__host__ __device__ void printKeyValues(KeyValuePair* kvs, int length, bool ind) {
	for(int i = 0; i < length; i++) {
		if (my_strlen(kvs[i].key) == 0) {
			printf("[%i = null]\n", i);
		} else {
			if (ind) {
				printf("print key: %s \t value: %d\n", kvs[i].key, kvs[i].ind);
			}
			else {
				printf("print key: %s \t value: %s\n", kvs[i].key, kvs[i].value);
			}
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
	char* tokens = my_strtok_r(kv.value, " ,.-;:'()\"\t", &pSave);
	int count = 0;

	while (tokens != NULL) {
		if (count >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			break;
		}
		KeyValuePair* curOut = &out[i * EMITS_PER_LINE + count];
		my_strcpy(curOut->key, tokens);
		my_strcpy(curOut->value, "1");
		//printf("out [%d][%d] key: %s, value: %s \n", i, count, curOut->key, curOut->value);
		tokens = my_strtok_r(NULL, " ,.-;:'()\"\t", &pSave);
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
	//printf("kernMap input key: %s, value: %s\n", in[i].key, in[i].value);
	map(in[i], out, i, 1);
	//printf("kernMap output key: %s, value: %s\n", out[i].key, out[i].value);
}

__global__ void kernReduce(KeyValuePair* in, KeyValuePair* out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;

}

__host__ void reduce(int start, int end, KeyValuePair* in, KeyValuePair** out, int n) {
	char* key = in[start].key;
	char value[50];
	sprintf(value, "%i", end-start);
	KeyValuePair* curOut = out[n];
	strcpy(curOut->key, key);
	strcpy(curOut->value, value);
	//out[n] = new KeyValuePair(key, value);
}


__host__ void cpuReduce(KeyValuePair* in, KeyValuePair** out, int length) {
	//if (in[0] == NULL) return;

	char* key = in[0].key;
	int start = 0;
	int n = 0;
	for (int i = 0; i < length; i++) {
		if (my_strlen(in[i].key) != NULL || my_strcmp(key, in[i].key) != 0) {
			reduce(start, i, in, out, n);
			//if(in[i] == NULL) {
			//	return; //Sorted, so we must be at the end
			//}

			key = in[i].key;
			start = i;
			n++; //TODO this math doesn't work out, ensure we can't overflow keys
		}
	}
}

__global__ void kernFindUniqBool(KeyValuePair* in, KeyValuePair* out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	if (i == 0 || my_strcmp(in[i].key, in[i - 1].key)) {
		KeyValuePair* curOut = &out[i];
		//char* value;
		//my_itoa(i, value, 10);
		my_strcpy(curOut->key, in[i].key);
		curOut->ind = i;
		//my_strcpy(curOut->value, value);
		return;
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
	kernMap << <128, 128 >> > (dev_file_kvs, dev_map_kvs, length);
	


	KeyValuePair* iter_end = thrust::partition(thrust::device, dev_map_kvs, dev_map_kvs + MAX_EMITS, KeyValueNotEmpty());
	int kv_num_map = iter_end - dev_map_kvs;
	printf("Remain kv number is %d \n", kv_num_map);
	thrust::device_ptr<KeyValuePair> dev_ptr(dev_map_kvs);
	thrust::sort(thrust::device, dev_ptr, dev_ptr + kv_num_map, KVComparator());

	KeyValuePair* map_kvs = NULL;
	map_kvs = (KeyValuePair*)malloc(kv_num_map * sizeof(KeyValuePair));
	cudaMemcpy(map_kvs, dev_map_kvs, kv_num_map * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
	//printKeyValues(map_kvs, kv_num_map);

	//bool* bool_array = NULL;
	//cudaMalloc((void **)&bool_array, kv_num * sizeof(bool));
	//cudaMemset(bool_array, 0, kv_num * sizeof(bool));

	KeyValuePair* dev_reduce_kvs = NULL;
	cudaMalloc((void **)&dev_reduce_kvs, kv_num_map * sizeof(KeyValuePair));

	kernFindUniqBool << <128, 128 >> >(dev_map_kvs, dev_reduce_kvs, kv_num_map);
	KeyValuePair* iter_end_reduce = thrust::partition(thrust::device, dev_reduce_kvs, dev_reduce_kvs + kv_num_map, KeyValueNotEmpty());
	int kv_num_reduce = iter_end_reduce - dev_reduce_kvs;

	KeyValuePair* reduce_kvs = NULL;
	reduce_kvs = (KeyValuePair*)malloc(kv_num_reduce * sizeof(KeyValuePair));
	cudaMemcpy(reduce_kvs, dev_reduce_kvs, kv_num_reduce * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
	printKeyValues(reduce_kvs, kv_num_reduce, true);

	free(map_kvs);

	cudaFree(dev_file_kvs);
	cudaFree(dev_map_kvs);

	free(reduce_kvs);
	cudaFree(dev_reduce_kvs);
 
	
#else
	KeyValuePair* map_kvs = NULL;
	map_kvs = (KeyValuePair*)malloc(MAX_EMITS * sizeof(KeyValuePair));
	auto t0 = Clock::now();
	cpuMap(file_kvs, map_kvs, length);
	auto t1 = Clock::now();
	printf("%d nanoseconds \n", t1 - t0);
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
