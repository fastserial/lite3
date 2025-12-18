/*
    TRON (formerly known as Lite³): Tree Root Object Notation

    Copyright © 2025 Elias de Jong <elias@fastserial.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

      __ __________________        ____
    _  ___ ___/ /___(_)_/ /_______|_  /
     _  _____/ / __/ /_  __/  _ \_/_ < 
      ___ __/ /___/ / / /_ /  __/____/ 
           /_____/_/  \__/ \___/       
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <errno.h>

#include "lite3.h"


static unsigned char buf[1024*64];

static const char ALPHANUMS[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static const int ALPHANUMS_LEN = sizeof(ALPHANUMS) - 1;

#define LITE3_TEST_ARRAY_COUNT (1024*1024)


int main()
{
	srand(52073821); // seed number generator


	size_t buflen = 0;
	size_t bufsz = sizeof(buf);

	if (lite3_init_obj(buf, &buflen, bufsz) < 0) {
		perror("Failed to initialize object");
		return 1;
	}


	int key_len = 2;
	size_t key_size = (size_t)key_len + 1;

	// array to store random character, try to find colliding keys
	size_t key_arr_size = (size_t)(LITE3_TEST_ARRAY_COUNT * key_size);
	char *key_arr = malloc(key_arr_size);
	if (!key_arr) {
		perror("Could not allocate key_arr");
		return 1;
	}

	// array for storing keys that have been found to collide
	size_t colliding_keys_buflen = 0;
	char *colliding_keys_arr = malloc(key_arr_size * 2);
	if (!colliding_keys_arr) {
		perror("Could not allocate colliding_keys_arr");
		free(key_arr);
		return 1;
	}

	// fill array with pseudorandom alphanumeric characters
	for (int i = 0; i < (int)key_arr_size; i++) {
		key_arr[i] = ALPHANUMS[rand() % ALPHANUMS_LEN];
	}

	// loop over key_arr, try to find colliding keys and store them in colliding_keys_arr
	char *prev_key = key_arr;
	uint32_t prev_hash = 0;
	for (int i = 0; i < (int)key_arr_size; i += (int)key_size) {
		uint32_t hash = LITE3_DJB2_HASH_SEED;
		for (int k = 0; k < key_len; k++) {
			hash = ((hash << 5) + hash) + (uint8_t)(*(key_arr + (uintptr_t)(i + k)));
		}
		char *curr_key = key_arr + (uintptr_t)i;
		if (prev_hash == hash && memcmp(prev_key, curr_key, (size_t)key_len) != 0) {
			prev_key[key_len] = '\0';
			curr_key[key_len] = '\0';
			if (	lite3_set_null(buf, &buflen, 0, bufsz, prev_key)	< 0
				|| lite3_set_null(buf, &buflen, 0, bufsz, curr_key)	< 0) {
				perror("Failed to insert key");
				goto cleanup;
			}
			memcpy(colliding_keys_arr + (uintptr_t)colliding_keys_buflen, prev_key, key_size * 2);
			colliding_keys_buflen += key_size * 2;
		}
		prev_hash = hash;
		prev_key = key_arr + (uintptr_t)i;
	}

	// for every key we inserted, can we actually find it back in the message?
	for (int i = 0; i < (int)colliding_keys_buflen; i += (int)key_size) {
		const char *test_key = (const char *)(colliding_keys_arr + (uintptr_t)i);
		
		bool key_exists = lite3_exists(buf, buflen, 0, test_key);
		if (!key_exists) {
			printf("key does not exist: %s\n", test_key);
			goto cleanup;
		}
	}

	// if (lite3_json_print(buf, buflen, 0) < 0) {
	// 	perror("Failed to print JSON");
	// 	goto cleanup;
	// }

	free(key_arr);
	free(colliding_keys_arr);
	return 0;
cleanup:
	free(key_arr);
	free(colliding_keys_arr);
	return 1;
}