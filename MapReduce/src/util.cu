#include <cuda_runtime.h>
__host__ __device__ int my_strlen(const char * str) {
	const char *s;
	for (s = str; *s; ++s) {

	}
	return s - str;
}

__host__ __device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len = 256) {
	for (int i = 0; ; i++) {
		if (str_a[i] != str_b[i]) {
			return str_a[i] < str_b[i] ? -1 : 1;
		}
		if (str_a[i] == '\0') {
			return 0;
		}
	}
}

__host__ __device__ char * my_strcpy(char *strDest, const char *strSrc)
{
	char *temp = strDest;
	while (*strDest++ = *strSrc++);
	return temp;
}

__host__ __device__ char* my_strtok(char *str, const char* delim) {
	static char* buffer;
	if (str != NULL) buffer = str;
	if (buffer[0] == '\0') return NULL;

	char *ret = buffer, *b;
	const char *d;

	for (b = buffer; *b != '\0'; b++) {
		for (d = delim; *d != '\0'; d++) {
			if (*b == *d) {
				*b = '\0';
				buffer = b + 1;
				if (b == ret) {
					ret++;
					continue;
				}
				return ret;
			}
		}
	}

	return ret;
}
