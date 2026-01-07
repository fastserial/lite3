/*
    Lite³: A JSON-Compatible Zero-Copy Serialization Format

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

/*
 * Tests for:
 * - lite3_arr_get_type() (buffer API)
 * - lite3_ctx_arr_get_type() (context API)
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#include "lite3.h"
#include "lite3_context_api.h"


/* Test lite3_arr_get_type with buffer API */
static int test_arr_get_type_buffer_api(void)
{
	unsigned char buf[2048];
	size_t buflen = 0;
	size_t bufsz = sizeof(buf);

	// Initialize as array
	if (lite3_init_arr(buf, &buflen, bufsz) < 0) {
		perror("Failed to initialize array");
		return 1;
	}

	// Append various types
	if (lite3_arr_append_str(buf, &buflen, 0, bufsz, "hello") < 0
		|| lite3_arr_append_i64(buf, &buflen, 0, bufsz, 42) < 0
		|| lite3_arr_append_f64(buf, &buflen, 0, bufsz, 3.14) < 0
		|| lite3_arr_append_bool(buf, &buflen, 0, bufsz, true) < 0
		|| lite3_arr_append_null(buf, &buflen, 0, bufsz) < 0) {
		perror("Failed to append values");
		return 1;
	}

	// Test type queries
	assert(lite3_arr_get_type(buf, buflen, 0, 0) == LITE3_TYPE_STRING);
	assert(lite3_arr_get_type(buf, buflen, 0, 1) == LITE3_TYPE_I64);
	assert(lite3_arr_get_type(buf, buflen, 0, 2) == LITE3_TYPE_F64);
	assert(lite3_arr_get_type(buf, buflen, 0, 3) == LITE3_TYPE_BOOL);
	assert(lite3_arr_get_type(buf, buflen, 0, 4) == LITE3_TYPE_NULL);

	// Test out of bounds returns LITE3_TYPE_INVALID
	assert(lite3_arr_get_type(buf, buflen, 0, 5) == LITE3_TYPE_INVALID);
	assert(lite3_arr_get_type(buf, buflen, 0, 100) == LITE3_TYPE_INVALID);

	return 0;
}


/* Test lite3_ctx_arr_get_type with context API */
static int test_arr_get_type_context_api(void)
{
	lite3_ctx *ctx = lite3_ctx_create();
	if (!ctx) {
		perror("Failed to create context");
		return 1;
	}

	// Initialize as array
	if (lite3_ctx_init_arr(ctx) < 0) {
		perror("Failed to initialize array");
		lite3_ctx_destroy(ctx);
		return 1;
	}

	// Append various types
	if (lite3_ctx_arr_append_str(ctx, 0, "world") < 0
		|| lite3_ctx_arr_append_i64(ctx, 0, 123) < 0
		|| lite3_ctx_arr_append_bool(ctx, 0, false) < 0) {
		perror("Failed to append values");
		lite3_ctx_destroy(ctx);
		return 1;
	}

	// Test type queries
	assert(lite3_ctx_arr_get_type(ctx, 0, 0) == LITE3_TYPE_STRING);
	assert(lite3_ctx_arr_get_type(ctx, 0, 1) == LITE3_TYPE_I64);
	assert(lite3_ctx_arr_get_type(ctx, 0, 2) == LITE3_TYPE_BOOL);

	// Test out of bounds
	assert(lite3_ctx_arr_get_type(ctx, 0, 3) == LITE3_TYPE_INVALID);

	lite3_ctx_destroy(ctx);
	return 0;
}


/* Test nested array type queries */
static int test_arr_get_type_nested(void)
{
	lite3_ctx *ctx = lite3_ctx_create();
	if (!ctx) {
		perror("Failed to create context");
		return 1;
	}

	// Initialize as object
	if (lite3_ctx_init_obj(ctx) < 0) {
		perror("Failed to initialize object");
		lite3_ctx_destroy(ctx);
		return 1;
	}

	// Add a nested array
	size_t arr_ofs;
	if (lite3_ctx_set_arr(ctx, 0, "items", &arr_ofs) < 0) {
		perror("Failed to set array");
		lite3_ctx_destroy(ctx);
		return 1;
	}

	// Append to nested array
	size_t nested_obj_ofs;
	if (lite3_ctx_arr_append_i64(ctx, arr_ofs, 1) < 0
		|| lite3_ctx_arr_append_obj(ctx, arr_ofs, &nested_obj_ofs) < 0
		|| lite3_ctx_arr_append_str(ctx, arr_ofs, "test") < 0) {
		perror("Failed to append to nested array");
		lite3_ctx_destroy(ctx);
		return 1;
	}

	// Test type queries on nested array
	assert(lite3_ctx_arr_get_type(ctx, arr_ofs, 0) == LITE3_TYPE_I64);
	assert(lite3_ctx_arr_get_type(ctx, arr_ofs, 1) == LITE3_TYPE_OBJECT);
	assert(lite3_ctx_arr_get_type(ctx, arr_ofs, 2) == LITE3_TYPE_STRING);

	lite3_ctx_destroy(ctx);
	return 0;
}


/* Test root type query via lite3_ctx_get_root_type(ctx) */
static int test_root_type_query_context_api(void)
{
	lite3_ctx *ctx;

	// Test object root
	ctx = lite3_ctx_create();
	if (!ctx) {
		perror("Failed to create context");
		return 1;
	}
	if (lite3_ctx_init_obj(ctx) < 0) {
		perror("Failed to initialize object");
		lite3_ctx_destroy(ctx);
		return 1;
	}
	assert(lite3_ctx_get_root_type(ctx) == LITE3_TYPE_OBJECT);
	lite3_ctx_destroy(ctx);

	// Test array root
	ctx = lite3_ctx_create();
	if (!ctx) {
		perror("Failed to create context");
		return 1;
	}
	if (lite3_ctx_init_arr(ctx) < 0) {
		perror("Failed to initialize array");
		lite3_ctx_destroy(ctx);
		return 1;
	}
	assert(lite3_ctx_get_root_type(ctx) == LITE3_TYPE_ARRAY);
	lite3_ctx_destroy(ctx);

	return 0;
}


/* Test root type query via lite3_get_root_type(buf, buflen) - buffer API */
static int test_root_type_query_buffer_api(void)
{
	unsigned char buf[2048];
	size_t buflen = 0;
	size_t bufsz = sizeof(buf);

	// Test object root
	if (lite3_init_obj(buf, &buflen, bufsz) < 0) {
		perror("Failed to initialize object");
		return 1;
	}
	assert(lite3_get_root_type(buf, buflen) == LITE3_TYPE_OBJECT);

	// Test array root
	buflen = 0;
	if (lite3_init_arr(buf, &buflen, bufsz) < 0) {
		perror("Failed to initialize array");
		return 1;
	}
	assert(lite3_get_root_type(buf, buflen) == LITE3_TYPE_ARRAY);

	return 0;
}


/* Test root type query on empty/uninitialized buffer */
static int test_root_type_empty_buffer(void)
{
	lite3_ctx *ctx = lite3_ctx_create();
	if (!ctx) {
		perror("Failed to create context");
		return 1;
	}

	// Don't initialize - buflen should be 0
	// Root type query should return INVALID due to validation
	assert(lite3_ctx_get_root_type(ctx) == LITE3_TYPE_INVALID);

	lite3_ctx_destroy(ctx);
	return 0;
}


int main(void)
{
	int ret;

	if ((ret = test_arr_get_type_buffer_api()) != 0) {
		fprintf(stderr, "test_arr_get_type_buffer_api failed\n");
		return ret;
	}

	if ((ret = test_arr_get_type_context_api()) != 0) {
		fprintf(stderr, "test_arr_get_type_context_api failed\n");
		return ret;
	}

	if ((ret = test_arr_get_type_nested()) != 0) {
		fprintf(stderr, "test_arr_get_type_nested failed\n");
		return ret;
	}

	if ((ret = test_root_type_query_context_api()) != 0) {
		fprintf(stderr, "test_root_type_query_context_api failed\n");
		return ret;
	}

	if ((ret = test_root_type_query_buffer_api()) != 0) {
		fprintf(stderr, "test_root_type_query_buffer_api failed\n");
		return ret;
	}

	if ((ret = test_root_type_empty_buffer()) != 0) {
		fprintf(stderr, "test_root_type_empty_buffer failed\n");
		return ret;
	}

	return 0;
}