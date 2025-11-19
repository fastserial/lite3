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
#include "lite3.h"



#ifdef LITE3_JSON
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#include "yyjson/yyjson.h"
#include "nibble_base64/base64.h"



// Typedef for primitive types
typedef float       f32;
typedef double      f64;
typedef int8_t      i8;
typedef uint8_t     u8;
typedef int16_t     i16;
typedef uint16_t    u16;
typedef int32_t     i32;
typedef uint32_t    u32;
typedef int64_t     i64;
typedef uint64_t    u64;



// Forward declarations
int _lite3_json_enc_obj(const unsigned char *buf, size_t buflen, size_t ofs, size_t nesting_depth, yyjson_mut_doc *doc, yyjson_mut_val *coll);
int _lite3_json_enc_arr(const unsigned char *buf, size_t buflen, size_t ofs, size_t nesting_depth, yyjson_mut_doc *doc, yyjson_mut_val *coll);

int _lite3_json_enc_switch(const unsigned char *buf, size_t buflen, size_t nesting_depth, yyjson_mut_doc *doc, yyjson_mut_val **yy_val, lite3_val *val)
{
        enum lite3_type type = lite3_val_type(val);
        switch (type) {
        case LITE3_TYPE_NULL:
        	*yy_val = yyjson_mut_null(doc);
        	break;
        case LITE3_TYPE_BOOL:
                *yy_val = yyjson_mut_bool(doc, lite3_val_bool(val));
                break;
        case LITE3_TYPE_I64:
                *yy_val = yyjson_mut_sint(doc, lite3_val_i64(val));
                break;
        case LITE3_TYPE_F64:
                *yy_val = yyjson_mut_double(doc, lite3_val_f64(val));
                break;
        case LITE3_TYPE_BYTES:
                size_t bytes_len;
                const u8 *bytes = lite3_val_bytes(val, &bytes_len);
                int b64_len;
                char *b64 = nibble_base64(bytes, (int)bytes_len, &b64_len);
                if (!b64) {
                	LITE3_PRINT_ERROR("FAILED TO CONVERT BYTES TO BASE64\n");
                	// No need to free the `b64` pointer, since the allocation would have failed anyways.
                	return -1;
                }
                /*
		According to `yyjson` docs:
			yyjson_mut_strn(...) 	--> "The input string is not copied, you should keep this string unmodified for the lifetime of this JSON document."
			yyjson_mut_strncpy(...) --> "The input string is copied and held by the document."

		So we must use `yyjson_mut_strncpy(...)` here so we can free the temporary base64 buffer.
		NOTE: `b64_len` does NOT contain the NULL terminator. `yyjson` docs say: "The `str` should be a UTF-8 string, null-terminator is not required."
                */
                *yy_val = yyjson_mut_strncpy(doc, b64, (size_t)b64_len);
                free(b64);
                break;
        case LITE3_TYPE_STRING:
                size_t str_len;
                const char *str = lite3_val_str_n(val, &str_len);
                /*
                Here we can use `yyjson_mut_strn(...)` since the value will remain backed up by the Lite³ buffer for the duration of the JSON-conversion process.
		`str_len` excludes the NULL terminator. `yyjson` docs say: "The `str` should be a UTF-8 string, null-terminator is not required."
                */
                *yy_val = yyjson_mut_strn(doc, str, str_len);
                break;
        case LITE3_TYPE_OBJECT:
        	*yy_val = yyjson_mut_obj(doc);
        	size_t obj_ofs = (size_t)((u8 *)val - buf);
	        if (_lite3_json_enc_obj(buf, buflen, obj_ofs, nesting_depth, doc, *yy_val) < 0)
			return -1;
        	break;
        case LITE3_TYPE_ARRAY:
        	*yy_val = yyjson_mut_arr(doc);
        	size_t arr_ofs = (size_t)((u8 *)val - buf);
	        if (_lite3_json_enc_arr(buf, buflen, arr_ofs, nesting_depth, doc, *yy_val) < 0)
			return -1;
        	break;
        default:
		LITE3_PRINT_ERROR("FAILED TO BUILD JSON DOCUMENT: VALUE TYPE INVALID\n");
		errno = EINVAL;
		return -1;
        }
        return 0;
}

/*
	Function that recursively builds a JSON document.
		- Returns 0 on success
		- Returns < 0 on error
*/
int _lite3_json_enc_obj(const unsigned char *buf, size_t buflen, size_t ofs, size_t nesting_depth, yyjson_mut_doc *doc, yyjson_mut_val *coll)
{
        if (++nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX) {
		LITE3_PRINT_ERROR("FAILED TO BUILD JSON DOCUMENT: nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX\n");
		errno = EINVAL;
		return -1;
        }
        lite3_iter iter;
        int ret;
        if ((ret = lite3_iter_create(buf, buflen, ofs, &iter)) < 0)
        	return ret;
        lite3_str key;
        lite3_val *val;
        size_t val_ofs;
        yyjson_mut_val *yy_val;
        while ((ret = lite3_iter_next(buf, buflen, &iter, &key, &val_ofs)) == LITE3_ITER_ITEM) {
        	val = (lite3_val *)(buf + val_ofs);
        	if ((ret = _lite3_json_enc_switch(buf, buflen, nesting_depth, doc, &yy_val, val)) < 0)
        		return ret;
                if (!yyjson_mut_obj_add(coll, yyjson_mut_str(doc, LITE3_STR(buf, key)), yy_val)) {
			LITE3_PRINT_ERROR("FAILED TO BUILD JSON DOCUMENT: ADDING KEY-VALUE PAIR FAILED\n");
			errno = EINVAL;
			return -1;
                }
        }
	return ret;
}

/*
	Function that recursively builds a JSON document.
		- Returns 0 on success
		- Returns < 0 on error
*/
int _lite3_json_enc_arr(const unsigned char *buf, size_t buflen, size_t ofs, size_t nesting_depth, yyjson_mut_doc *doc, yyjson_mut_val *coll)
{
        if (++nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX) {
		LITE3_PRINT_ERROR("FAILED TO BUILD JSON DOCUMENT: nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX\n");
		errno = EINVAL;
		return -1;
        }
        lite3_iter iter;
        int ret;
        if ((ret = lite3_iter_create(buf, buflen, ofs, &iter)) < 0)
        	return ret;
        lite3_val *val;
        size_t val_ofs;
        yyjson_mut_val *yy_val;
        while ((ret = lite3_iter_next(buf, buflen, &iter, NULL, &val_ofs)) == LITE3_ITER_ITEM) {
        	val = (lite3_val *)(buf + val_ofs);
        	if ((ret = _lite3_json_enc_switch(buf, buflen, nesting_depth, doc, &yy_val, val)) < 0)
        		return ret;
                if (!yyjson_mut_arr_append(coll, yy_val)) {
			LITE3_PRINT_ERROR("FAILED TO BUILD JSON DOCUMENT: APPENDING ARRAY ELEMENT FAILED\n");
			errno = EINVAL;
			return -1;
                }
        }
	return ret;
}

yyjson_mut_doc *_lite3_json_enc_doc(const unsigned char *buf, size_t buflen, size_t ofs)
{
        if (_lite3_verify_get(buf, buflen, ofs) < 0)
                return NULL;
	yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
	if (!doc)
		return NULL;
	yyjson_mut_val *root;
        switch (*(buf + ofs)) {
        case LITE3_TYPE_OBJECT:
        	root = yyjson_mut_obj(doc);
	        if (_lite3_json_enc_obj(buf, buflen, ofs, 0, doc, root) < 0)
			goto error;
        	break;
        case LITE3_TYPE_ARRAY:
        	root = yyjson_mut_arr(doc);
	        if (_lite3_json_enc_arr(buf, buflen, ofs, 0, doc, root) < 0)
			goto error;
        	break;
        default:
		LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
		errno = EINVAL;
		goto error;
        }
        yyjson_mut_doc_set_root(doc, root);
	return doc;
error:
	yyjson_mut_doc_free(doc);
	return NULL;
}

int lite3_json_print(const unsigned char *buf, size_t buflen, size_t ofs)
{
	yyjson_mut_doc *doc = _lite3_json_enc_doc(buf, buflen, ofs);
	if (!doc)
		return -1;
	size_t len;
	yyjson_write_err err;
	char *json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY, NULL, &len, &err);
	yyjson_mut_doc_free(doc);

	if (!json) {
		LITE3_PRINT_ERROR("FAILED TO WRITE JSON\tyyjson error code: %u msg:%s\n", err.code, err.msg);
		errno = EIO;
		return -1;
	}
	fwrite(json, 1, len, stdout);
	fputc('\n', stdout);
	free(json);
	return 0;
}

char *lite3_json_enc(const unsigned char *buf, size_t buflen, size_t ofs, size_t *restrict out_len)
{
	yyjson_mut_doc *doc = _lite3_json_enc_doc(buf, buflen, ofs);
	if (!doc)
		return NULL;
	yyjson_write_err err;
	char *json = yyjson_mut_write_opts(doc, YYJSON_WRITE_NOFLAG, NULL, out_len, &err);
	yyjson_mut_doc_free(doc);
	if (!json) {
		LITE3_PRINT_ERROR("FAILED TO WRITE JSON\tyyjson error code: %u msg:%s\n", err.code, err.msg);
		errno = EIO;
		return NULL;
	}
	return json;
}

char *lite3_json_enc_pretty(const unsigned char *buf, size_t buflen, size_t ofs, size_t *restrict out_len)
{
	yyjson_mut_doc *doc = _lite3_json_enc_doc(buf, buflen, ofs);
	if (!doc)
		return NULL;
	yyjson_write_err err;
	char *json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY, NULL, out_len, &err);
	yyjson_mut_doc_free(doc);
	if (!json) {
		LITE3_PRINT_ERROR("FAILED TO WRITE JSON\tyyjson error code: %u msg:%s\n", err.code, err.msg);
		errno = EIO;
		return NULL;
	}
	return json;
}

int64_t lite3_json_enc_buf(const unsigned char *buf, size_t buflen, size_t ofs, char *restrict json_buf, size_t json_bufsz)
{
	yyjson_mut_doc *doc = _lite3_json_enc_doc(buf, buflen, ofs);
	if (!doc)
		return -1;
	yyjson_write_err err;
	size_t ret = yyjson_mut_write_buf(json_buf, json_bufsz, doc, YYJSON_WRITE_NOFLAG, &err);
	assert(ret <= INT64_MAX);
	yyjson_mut_doc_free(doc);
	if (ret == 0) {
		LITE3_PRINT_ERROR("FAILED TO WRITE JSON\tyyjson error code: %u msg:%s\n", err.code, err.msg);
		errno = EIO;
		return -1;
	}
	return (i64)ret;
}

int64_t lite3_json_enc_buf_pretty(const unsigned char *buf, size_t buflen, size_t ofs, char *restrict json_buf, size_t json_bufsz)
{
	yyjson_mut_doc *doc = _lite3_json_enc_doc(buf, buflen, ofs);
	if (!doc) 
		return -1;
	yyjson_write_err err;
	size_t ret = yyjson_mut_write_buf(json_buf, json_bufsz, doc, YYJSON_WRITE_PRETTY, &err);
	assert(ret <= INT64_MAX);
	yyjson_mut_doc_free(doc);
	if (ret == 0) {
		LITE3_PRINT_ERROR("FAILED TO WRITE JSON\tyyjson error code: %u msg:%s\n", err.code, err.msg);
		errno = EIO;
		return -1;
	}
	return (i64)ret;
}
#endif // LITE3_JSON