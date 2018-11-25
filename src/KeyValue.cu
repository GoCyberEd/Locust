#include <string.h>
#include <stdio.h>

#include "KeyValue.h"

KeyValuePair::KeyValuePair() {
	key = NULL;
	value = NULL;
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
}
