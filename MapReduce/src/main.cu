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

#define MAX_LINES_FILE_READ 4500
#define EMITS_PER_LINE 20
#define MAX_EMITS (MAX_LINES_FILE_READ * EMITS_PER_LINE)
#define GPU_IMPLEMENTATION 1

#define WINDOWS 0
#define LINUX 1
#define COMPILE_OS WINDOWS

#if GPU_IMPLEMENTATION
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


__host__ __device__ void printKeyValues(KeyValuePair* kvs, int length) {
	for(int i = 0; i < length; i++) {
		if (my_strlen(kvs[i].key) == 0) {
			//printf("[%i = null]\n", i);
		} else {
			printf("print key: %s \t value: %s\n", kvs[i].key, kvs[i].value);
		}
	}
}

__host__ __device__ void printKeyIntValues(KeyIntValuePair* kvs, int length) {
	for (int i = 0; i < length; i++) {
		if (my_strlen(kvs[i].key) == 0) {
			//printf("[%i = null]\n", i);
		}
		else {
			printf("print key: %s \t count: %d\n", kvs[i].key, kvs[i].count);
		}
	}
}

__host__ __device__ void emit(KeyValuePair kv, KeyValuePair** out, int n) {
	//out[n] = new KeyValuePair(kv);
	
}

__host__ __device__ void map(KeyValuePair kv, KeyIntValuePair* out, int i, bool is_device) {
	char* pSave = NULL;
	char* tokens = my_strtok_r(kv.value, " ,.-;:'()\"\t", &pSave);
	int count = 0;
	while (tokens != NULL) {
		if (count >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			break;
		}
		KeyIntValuePair* curOut = &out[i * EMITS_PER_LINE + count];
		my_strcpy(curOut->key, tokens);
		curOut->value = 1;
		//my_strcpy(curOut->value, "1");
		//printf("out [%d][%d] key: %s, value: %s \n", i, count, curOut->key, curOut->value);
		tokens = my_strtok_r(NULL, " ,.-;:'()\"\t", &pSave);
		count++;
	}
}

__global__ void kernMap(KeyValuePair* in, KeyIntValuePair* out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	map(in[i], out, i, 1);
}

__global__ void kernFindUniqBool(KeyIntValuePair* in, KeyIntValuePair* out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	if (i == 0 || my_strcmp(in[i].key, in[i - 1].key)) {
		KeyIntValuePair* curOut = &out[i];
		//char* value;
		//my_itoa(i, value, 10);
		my_strcpy(curOut->key, in[i].key);
		//printf("curOut->key is %s \n", curOut->key);
		curOut->value = i;
		curOut->count = 0;
		//printf("curOut->value is %d \n", curOut->value);
		//my_strcpy(curOut->value, value);
		return;
	}
	else {
		KeyIntValuePair* curOut = &out[i];
		my_strcpy(curOut->key, "");
		curOut->value = 0;
	}
}

__global__ void kernGetCount(KeyIntValuePair* in, int length, int end) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	KeyIntValuePair* curOut = &in[i];
	if (i == length - 1) {
		curOut->count = end - curOut->value;
	}
	else {
		curOut->count = in[i + 1].value - curOut->value;
	}
}
#else

__host__ void loadFile(char fname[], KeyValuePair** kvs, int* length) {
#if COMPILE_OS == WINDOWS
	std::ifstream input(fname);
	int line_num = 0;

	for (std::string line; getline(input, line); )
	{
		
		char *cstr = new char[line.length() + 1];
		strcpy(cstr, line.c_str());
		kvs[line_num] = new KeyValuePair();
		KeyValuePair* curkvs = kvs[line_num];		
		my_itoa(line_num, curkvs->key, 10);
		my_strcpy(curkvs->value, cstr);
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
		line_num++;
	}
	fclose(fp);
	if (line)
		free(line);
	*length = line_num;
#endif
}

__host__ __device__ void printKeyValues(KeyValuePair** kvs, int length) {
	for (int i = 0; i < length; i++) {
		if (kvs[i] == NULL) {
			//printf("[%i = null]\n", i);
		}
		else {
			printf("print key: %s \t value: %s\n", kvs[i]->key, kvs[i]->value);
		}
	}
}

__host__ __device__ void emit(KeyValuePair kv, KeyValuePair** out, int n) {
	out[n] = new KeyValuePair(kv);
}

__host__ __device__ void map(KeyValuePair kv, KeyValuePair** out, int n, bool is_device) {
	char* pSave = NULL;
	char* tokens = my_strtok_r(kv.value, " ,.-;:'()\"\t", &pSave);
	int i = 0;
	while (tokens != NULL) {
		if (i >= EMITS_PER_LINE) {
			printf("WARN: Exceeded emit limit\n");
			return;
		}
		out[n * EMITS_PER_LINE + i] = new KeyValuePair();
		KeyValuePair* curOut = out[n * EMITS_PER_LINE + i];
		my_strcpy(curOut->key, tokens);
		my_strcpy(curOut->value, "1");
		tokens = my_strtok_r(NULL, " ,.-;:'()\"\t", &pSave);
		i++;
	}
}

__host__ void cpuMap(KeyValuePair** in, KeyValuePair** out, int length) {
	for (int i = 0; i < length; i++) {
		map(*in[i], out, i, 0);
	}
}

__global__ void kernMap(KeyValuePair** in, KeyValuePair** out, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	//printf("Reading input key: %s, %s", in[i]->key, in[i]->value);
	map(*in[i], out, i * EMITS_PER_LINE, 1);
}

__host__ void reduce(int start, int end, KeyValuePair** in, KeyValuePair** out, int n) {
	char* key = in[start]->key;
	char value[50];
	sprintf(value, "%i", end - start);
	out[n] = new KeyValuePair();
	KeyValuePair* curOut = out[n];
	my_strcpy(curOut->key, key);
	my_strcpy(curOut->value, value);
}


__host__ void cpuReduce(KeyValuePair** in, KeyValuePair** out, int length) {
	if (in[0] == NULL) return;

	char* key = in[0]->key;
	int start = 0;
	int n = 0;
	for (int i = 0; i < length; i++) {
		if (in[i] == NULL || my_strcmp(key, in[i]->key) != 0) {
			reduce(start, i, in, out, n);
			if (in[i] == NULL) {
				return; //Sorted, so we must be at the end
			}

			key = in[i]->key;
			start = i;
			n++; //TODO this math doesn't work out, ensure we can't overflow keys
		}
	}
}
#endif



__host__ int main(int argc, char* argv[]) {
	typedef std::chrono::high_resolution_clock Clock;

	std::cout << "Running\n";
	char* filename = "hamlet.txt";
#if GPU_IMPLEMENTATION
	// Sort filtered map output
	int length = 0;
	KeyValuePair file_kvs[MAX_LINES_FILE_READ] = { NULL };
	loadFile(filename, file_kvs, &length);
	printf("Length: %i\n", length);

	KeyValuePair* dev_file_kvs = NULL;
	cudaMalloc((void **)&dev_file_kvs, MAX_LINES_FILE_READ * sizeof(KeyValuePair));
	cudaMemcpy(dev_file_kvs, file_kvs, MAX_LINES_FILE_READ * sizeof(KeyValuePair), cudaMemcpyHostToDevice);

	KeyIntValuePair* dev_map_kvs = NULL;
	cudaMalloc((void **)&dev_map_kvs, MAX_EMITS * sizeof(KeyIntValuePair));

	auto t0 = Clock::now();
	kernMap << <128, 256 >> > (dev_file_kvs, dev_map_kvs, length);
	auto t1 = Clock::now();
	printf("GPU mapping %d nanoseconds \n", t1 - t0);

	// stream compaction
	KeyIntValuePair* iter_end = thrust::partition(thrust::device, dev_map_kvs, dev_map_kvs + MAX_EMITS, KeyIntValueNotEmpty());
	int kv_num_map = iter_end - dev_map_kvs;
	//printf("Remain kv number is %d \n", kv_num_map);
	thrust::device_ptr<KeyIntValuePair> dev_ptr(dev_map_kvs);
	thrust::sort(thrust::device, dev_ptr, dev_ptr + kv_num_map, KIVComparator());

	auto t2 = Clock::now();
	printf("GPU stream compaction and sorting %d nanoseconds \n", t2 - t1);


	//KeyIntValuePair* map_kvs = NULL;
	//map_kvs = (KeyIntValuePair*)malloc(kv_num_map * sizeof(KeyIntValuePair));
	//cudaMemcpy(map_kvs, dev_map_kvs, kv_num_map * sizeof(KeyIntValuePair), cudaMemcpyDeviceToHost);
	//printKeyIntValues(map_kvs, kv_num_map);

	//bool* bool_array = NULL;
	//cudaMalloc((void **)&bool_array, kv_num * sizeof(bool));
	//cudaMemset(bool_array, 0, kv_num * sizeof(bool));

	KeyIntValuePair* dev_reduce_kvs = NULL;
	cudaMalloc((void **)&dev_reduce_kvs, kv_num_map * sizeof(KeyIntValuePair));


	
	auto t3 = Clock::now();
	kernFindUniqBool << <128, 256 >> >(dev_map_kvs, dev_reduce_kvs, kv_num_map);


	//KeyIntValuePair* reduce_kvs = NULL;
	//reduce_kvs = (KeyIntValuePair*)malloc(kv_num_map * sizeof(KeyIntValuePair));
	//cudaMemcpy(reduce_kvs, dev_reduce_kvs, kv_num_map * sizeof(KeyIntValuePair), cudaMemcpyDeviceToHost);
	//printKeyIntValues(reduce_kvs, kv_num_map);

	KeyIntValuePair* iter_end_reduce = thrust::partition(thrust::device, dev_reduce_kvs, dev_reduce_kvs + kv_num_map, KeyIntValueNotEmpty());
	int kv_num_reduce = iter_end_reduce - dev_reduce_kvs;

	kernGetCount << <128, 256 >> >(dev_reduce_kvs, kv_num_reduce, kv_num_map);

	auto t4 = Clock::now();
	printf("GPU reduce %d nanoseconds \n", t4 - t3);

	KeyIntValuePair* reduce_kvs = NULL;
	reduce_kvs = (KeyIntValuePair*)malloc(kv_num_reduce * sizeof(KeyIntValuePair));
	cudaMemcpy(reduce_kvs, dev_reduce_kvs, kv_num_reduce * sizeof(KeyIntValuePair), cudaMemcpyDeviceToHost);
	printKeyIntValues(reduce_kvs, kv_num_reduce);

	//free(map_kvs);

	cudaFree(dev_file_kvs);
	cudaFree(dev_map_kvs);

	free(reduce_kvs);
	cudaFree(dev_reduce_kvs);
 
	
#else
	int length = 0;
	KeyValuePair* file_kvs[MAX_LINES_FILE_READ] = { NULL };
	loadFile(filename, file_kvs, &length);
	KeyValuePair* map_kvs[MAX_EMITS] = { NULL };
	auto t0 = Clock::now();
	cpuMap(file_kvs, map_kvs, length);
	auto t1 = Clock::now();
	printf("CPU mapping %d nanoseconds \n", t1 - t0);
	std::sort(map_kvs, map_kvs + MAX_EMITS, KVComparatorCPU());
	auto t2 = Clock::now();
	printf("CPU sorting %d nanoseconds \n", t2 - t1);
	// Reduce stage
	KeyValuePair* reduce_kvs[MAX_EMITS] = { NULL };
	auto t3 = Clock::now();
	cpuReduce(map_kvs, reduce_kvs, MAX_EMITS);
	auto t4 = Clock::now();
	printf("CPU reducing %d nanoseconds \n", t4 - t3);
	std::sort(reduce_kvs, reduce_kvs + MAX_EMITS, KVComparatorCPU());
	printKeyValues(reduce_kvs, MAX_EMITS);
	//KeyValuePair* map_kvs = NULL;
	//map_kvs = (KeyValuePair*)malloc(MAX_EMITS * sizeof(KeyValuePair));
	//auto t0 = Clock::now();
	//cpuMap(file_kvs, map_kvs, length);
	//auto t1 = Clock::now();
	//printf("%d nanoseconds \n", t1 - t0);

	//std::sort(map_kvs, map_kvs + MAX_EMITS, KVComparator());

	//// Reduce stage
	//KeyValuePair* reduce_kvs = NULL;
	//reduce_kvs = (KeyValuePair*)malloc(MAX_EMITS * sizeof(KeyValuePair));
	//cpuReduce(map_kvs, reduce_kvs, MAX_EMITS);
	//std::sort(reduce_kvs, reduce_kvs + MAX_EMITS, KVComparator());
	//printKeyValues(reduce_kvs, MAX_EMITS);
	//
	//free(map_kvs);
	//free(reduce_kvs);
#endif
	
	std::cout << "\nDone\n";
	std::cin.ignore();
	return 0;
}
