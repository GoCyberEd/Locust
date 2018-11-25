#include <string>
#include <iostream>
#include <fstream>

#include "KeyValue.h"

#define MAX_LINES 1024

void loadFile(char fname[], KeyValuePair** kvs, int* length) {
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

int main(int argc, char* argv[]) {
	std::cout << "Running\n";
	int length = 0;
	KeyValuePair* kvs[MAX_LINES];
	loadFile("LICENSE", kvs, &length);
	printf("Length: %i\n", length);

	for(int i = 0; i < length; i++) {
		printf("%s \t %s\n", kvs[i]->key, kvs[i]->value);
	}

	std::cout << "Done\n";
	return 0;
}
