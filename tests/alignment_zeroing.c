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
#include <string.h>
#include <assert.h>

#include "lite3.h"



int main() {
	#ifdef LITE3_ZERO_MEM_EXTRA

	unsigned char buf[1024] __attribute__((aligned(LITE3_NODE_ALIGNMENT)));
	size_t buflen = 0;
	size_t bufsz = sizeof(buf);

	// Fill buffer with non-zero garbage
	memset(buf, 0xEE, bufsz);

	if (lite3_init_obj(buf, &buflen, bufsz) < 0) { // LITE3_NODE_SIZE (96)
		perror("Failed to initialize object");
		return 1;
	}
	#ifdef LITE3_DEBUG
		printf("Test 1\n");
		printf("buflen after init: %zu\n", buflen);
	#endif

	// Object insert adds 99 bytes: LITE3_NODE_SIZE (96) + "a" (size 2 including \0) + key_tag (size 1)
	// 1 padding byte is inserted to reach 100 bytes, for 4 byte alignment.
	size_t obj_ofs;
	if (lite3_set_obj(buf, &buflen, 0, bufsz, "a", &obj_ofs) < 0) { 
		perror("Failed to set object");
		return 1;
	}
	#ifdef LITE3_DEBUG
		printf("buflen after 'a': %zu\n", buflen);
		printf("Padding byte at index %d: 0x%02X (expected 0x%02X)\n", LITE3_NODE_SIZE, buf[LITE3_NODE_SIZE], LITE3_ZERO_MEM_8);
	#endif

	// Validate padding byte was zeroed
	assert(buf[LITE3_NODE_SIZE] == LITE3_ZERO_MEM_8);

	// Reset buffer to garbage for second test
	memset(buf, 0xEE, bufsz);

	if (lite3_init_obj(buf, &buflen, bufsz) < 0) { // LITE3_NODE_SIZE (96)
		perror("Failed to initialize object");
		return 1;
	}
	#ifdef LITE3_DEBUG
		printf("\nTest 2\n");
		printf("buflen after init: %zu\n", buflen);
	#endif

	// Object insert adds 112 bytes (LITE3_NODE_SIZE (96) + keyval (16))
	// key_tag(1) + "key1\0"(5) + val_tag(1) + str_len(4) + "val1\0"(5) = 16 bytes.
	if (lite3_set_str(buf, &buflen, 0, bufsz, "key1", "val1") < 0) {
		perror("Failed to set string");
		return 1;
	}
	#ifdef LITE3_DEBUG
		printf("buflen after 'key1': %zu\n", buflen);
	#endif
	
	size_t test_buflen = buflen;

	// Overwrite "key1":"val1" with an Object
	if (lite3_set_obj(buf, &buflen, 0, bufsz, "key1", NULL) < 0) {
		perror("Failed to set object");
		return 1;
	}
	#ifdef LITE3_DEBUG
		printf("buflen after update 'key1': %zu\n", buflen);
		printf("Padding bytes at %zu, %zu: 0x%02X 0x%02X\n", test_buflen, test_buflen + 1, buf[test_buflen], buf[test_buflen + 1]);
	#endif

	assert(buf[test_buflen] == LITE3_ZERO_MEM_8);
	assert(buf[test_buflen + 1] == LITE3_ZERO_MEM_8);

	#endif // LITE3_ZERO_MEM_EXTRA
	return 0;
}

