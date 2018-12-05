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

//__host__ __device__ char* my_strtok(char *str, const char* delim) {
//	char* buffer;
//	if (str != NULL) buffer = str;
//	if (buffer[0] == '\0') return NULL;
//
//	char *ret = buffer, *b;
//	const char *d;
//
//	for (b = buffer; *b != '\0'; b++) {
//		for (d = delim; *d != '\0'; d++) {
//			if (*b == *d) {
//				*b = '\0';
//				buffer = b + 1;
//				if (b == ret) {
//					ret++;
//					continue;
//				}
//				return ret;
//			}
//		}
//	}
//
//	return ret;
//}

__host__ __device__ char * my_strtok_r(char * __restrict s, const char * __restrict delim, char ** __restrict last)
{
	char *spanp, *tok;
	int c, sc;

	if (s == NULL && (s = *last) == NULL)
		return (NULL);

cont:
	c = *s++;
	for (spanp = (char *)delim; (sc = *spanp++) != 0;) {
		if (c == sc)
			goto cont;
	}

	if (c == 0) {	
		*last = NULL;
		return (NULL);
	}
	tok = s - 1;

	for (;;) {
		c = *s++;
		spanp = (char *)delim;
		do {
			if ((sc = *spanp++) == c) {
				if (c == 0)
					s = NULL;
				else
					s[-1] = '\0';
				*last = s;
				return (tok);
			}
		} while (sc != 0);
	}
}
