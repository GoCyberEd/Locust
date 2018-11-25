#include <string>
#include <iostream>
#include <fstream>

#define MAX_LINE_LENGTH 1024

void loadFile(char fname[]) {
	FILE* fp = fopen(fname, "r");
	if (fp == NULL)
	    exit(EXIT_FAILURE);

	char* line = NULL;
	size_t len = 0;
	while ((getline(&line, &len, fp)) != -1) {
	    printf("%s", line);
	}
	fclose(fp);
	if (line)
	    free(line);
}

int main(int argc, char* argv[]) {
	std::cout << "Running";
	loadFile("LICENSE");

	return 0;
}
