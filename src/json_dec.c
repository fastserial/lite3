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
#include <stdint.h>
#include <errno.h>

#include "yyjson/yyjson.h"



// Forward declarations
int _lite3_json_dec_obj(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *obj);
int _lite3_json_dec_arr(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *arr);

int _lite3_json_dec_obj_switch(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *yy_key, yyjson_val *yy_val)
{
        const char *key = yyjson_get_str(yy_key);
        yyjson_type type = yyjson_get_type(yy_val);
        int ret;
        switch (type) {
        case YYJSON_TYPE_NULL:
                if ((ret = lite3_set_null(buf, inout_buflen, ofs, bufsz, key)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_BOOL:
                switch (yyjson_get_subtype(yy_val)) {
                case YYJSON_SUBTYPE_FALSE:
                        if ((ret = lite3_set_bool(buf, inout_buflen, ofs, bufsz, key, 0)) < 0)
                                return ret;
                        break;
                case YYJSON_SUBTYPE_TRUE:
                        if ((ret = lite3_set_bool(buf, inout_buflen, ofs, bufsz, key, 1)) < 0)
                                return ret;
                        break;
                default:
                        LITE3_PRINT_ERROR("FAILED TO READ JSON: EXPECTING BOOL SUBTYPE\n");
                        errno = EINVAL;
                        return -1;
                } 
                break;
        case YYJSON_TYPE_NUM:
                switch (yyjson_get_subtype(yy_val)) {
                case YYJSON_SUBTYPE_SINT:
                        int64_t num_i64 = yyjson_get_sint(yy_val);
                        if ((ret = lite3_set_i64(buf, inout_buflen, ofs, bufsz, key, num_i64)) < 0)
                                return ret;
                        break;
                case YYJSON_SUBTYPE_UINT:
                        uint64_t num_u64 = yyjson_get_uint(yy_val);
                        if (num_u64 <= INT64_MAX) {
                                if ((ret = lite3_set_i64(buf, inout_buflen, ofs, bufsz, key, (int64_t)num_u64)) < 0)
                                        return ret;
                                break;
                        } else {
                                if ((ret = lite3_set_f64(buf, inout_buflen, ofs, bufsz, key, (double)num_u64)) < 0) // Number to big for (signed) int64_t, fall back to double
                                        return ret;
                                break;
                        }
                case YYJSON_SUBTYPE_REAL:
                        double num_f64 = yyjson_get_real(yy_val);
                        if ((ret = lite3_set_f64(buf, inout_buflen, ofs, bufsz, key, num_f64)) < 0)
                                return ret;
                        break;
                default:
                        LITE3_PRINT_ERROR("FAILED TO READ JSON: EXPECTING NUM SUBTYPE\n");
                        errno = EINVAL;
                        return -1;
                }
                break;
        case YYJSON_TYPE_STR:
                const char *str = yyjson_get_str(yy_val);
                size_t len = yyjson_get_len(yy_val);
                if ((ret = lite3_set_str_n(buf, inout_buflen, ofs, bufsz, key, str, len)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_OBJ:
                size_t obj_ofs;
                if ((ret = lite3_set_obj(buf, inout_buflen, ofs, bufsz, key, &obj_ofs)) < 0)
                        return ret;
                if ((ret = _lite3_json_dec_obj(buf, inout_buflen, obj_ofs, bufsz, nesting_depth, doc, yy_val)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_ARR:
                size_t arr_ofs;
                if ((ret = lite3_set_arr(buf, inout_buflen, ofs, bufsz, key, &arr_ofs)) < 0)
                        return ret;
                if ((ret = _lite3_json_dec_arr(buf, inout_buflen, arr_ofs, bufsz, nesting_depth, doc, yy_val)) < 0)
                        return ret;
                break;
        default:
                LITE3_PRINT_ERROR("FAILED TO READ JSON: INVALID TYPE\n");
                errno = EINVAL;
                return -1;
        }
        return ret;
}

int _lite3_json_dec_arr_switch(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *yy_val)
{
        yyjson_type type = yyjson_get_type(yy_val);
        int ret;
        switch (type) {
        case YYJSON_TYPE_NULL:
                if ((ret = lite3_arr_append_null(buf, inout_buflen, ofs, bufsz)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_BOOL:
                switch (yyjson_get_subtype(yy_val)) {
                case YYJSON_SUBTYPE_FALSE:
                        if ((ret = lite3_arr_append_bool(buf, inout_buflen, ofs, bufsz, 0)) < 0)
                                return ret;
                        break;
                case YYJSON_SUBTYPE_TRUE:
                        if ((ret = lite3_arr_append_bool(buf, inout_buflen, ofs, bufsz, 1)) < 0)
                                return ret;
                        break;
                default:
                        LITE3_PRINT_ERROR("FAILED TO READ JSON: EXPECTING BOOL SUBTYPE\n");
                        errno = EINVAL;
                        return -1;
                }
                break;
        case YYJSON_TYPE_NUM:
                switch (yyjson_get_subtype(yy_val)) {
                case YYJSON_SUBTYPE_SINT:
                        int64_t num_i64 = yyjson_get_sint(yy_val);
                        if ((ret = lite3_arr_append_i64(buf, inout_buflen, ofs, bufsz, num_i64)) < 0)
                                return ret;
                        break;
                case YYJSON_SUBTYPE_UINT:
                        uint64_t num_u64 = yyjson_get_uint(yy_val);
                        if (num_u64 <= INT64_MAX) {
                                if ((ret = lite3_arr_append_i64(buf, inout_buflen, ofs, bufsz, (int64_t)num_u64)) < 0)
                                        return ret;
                                break;
                        } else {
                                if ((ret = lite3_arr_append_f64(buf, inout_buflen, ofs, bufsz, (double)num_u64)) < 0) // Number to big for (signed) int64_t, fall back to double
                                        return ret;
                                break;
                        }
                case YYJSON_SUBTYPE_REAL:
                        double num_f64 = yyjson_get_real(yy_val);
                        if ((ret = lite3_arr_append_f64(buf, inout_buflen, ofs, bufsz, num_f64)) < 0)
                                return ret;
                        break;
                default:
                        LITE3_PRINT_ERROR("FAILED TO READ JSON: EXPECTING NUM SUBTYPE\n");
                        errno = EINVAL;
                        return -1;
                }
                break;
        case YYJSON_TYPE_STR:
                const char *str = yyjson_get_str(yy_val);
                size_t len = yyjson_get_len(yy_val);
                if ((ret = lite3_arr_append_str_n(buf, inout_buflen, ofs, bufsz, str, len)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_OBJ:
                size_t obj_ofs;
                if ((ret = lite3_arr_append_obj(buf, inout_buflen, ofs, bufsz, &obj_ofs)) < 0)
                        return ret;
                if ((ret = _lite3_json_dec_obj(buf, inout_buflen, obj_ofs, bufsz, nesting_depth, doc, yy_val)) < 0)
                        return ret;
                break;
        case YYJSON_TYPE_ARR:
                size_t arr_ofs;
                if ((ret = lite3_arr_append_arr(buf, inout_buflen, ofs, bufsz, &arr_ofs)) < 0)
                        return ret;
                if ((ret = _lite3_json_dec_arr(buf, inout_buflen, arr_ofs, bufsz, nesting_depth, doc, yy_val)) < 0)
                        return ret;
                break;
        default:
                LITE3_PRINT_ERROR("FAILED TO READ JSON: INVALID TYPE\n");
                errno = EINVAL;
                return -1;
        }
        return ret;
}

int _lite3_json_dec_obj(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *obj)
{
        if (++nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX) {
                LITE3_PRINT_ERROR("FAILED TO READ JSON: nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX\n");
                errno = EINVAL;
                return -1;
        }
        yyjson_val *key, *val;
        yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
        int ret = 0;
        while ((key = yyjson_obj_iter_next(&iter))) {
                val = yyjson_obj_iter_get_val(key);
                if ((ret = _lite3_json_dec_obj_switch(buf, inout_buflen, ofs, bufsz, nesting_depth, doc, key, val)) < 0)
                        return ret;
        }
        return ret;
}

int _lite3_json_dec_arr(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t nesting_depth, yyjson_doc *doc, yyjson_val *arr)
{
        if (++nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX) {
                LITE3_PRINT_ERROR("FAILED TO READ JSON: nesting_depth > LITE3_JSON_NESTING_DEPTH_MAX\n");
                errno = EINVAL;
                return -1;
        }
        yyjson_val *val;
        yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
        int ret = 0;
        while ((val = yyjson_arr_iter_next(&iter))) {
                if ((ret = _lite3_json_dec_arr_switch(buf, inout_buflen, ofs, bufsz, nesting_depth, doc, val)) < 0)
                        return ret;
        }
        return ret;
}

int _lite3_json_dec_doc(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, yyjson_doc *doc)
{
        yyjson_val *root_val = yyjson_doc_get_root(doc);
        int ret = 0;
        switch (yyjson_get_type(root_val)) {
        case YYJSON_TYPE_OBJ:
                if ((ret = lite3_init_obj(buf, out_buflen, bufsz)) < 0)
                        goto error;
                if ((ret = _lite3_json_dec_obj(buf, out_buflen, 0, bufsz, 0, doc, root_val)) < 0)
                        goto error;
                break;
        case YYJSON_TYPE_ARR:
                if ((ret = lite3_init_arr(buf, out_buflen, bufsz)) < 0)
                        goto error;
                if ((ret = _lite3_json_dec_arr(buf, out_buflen, 0, bufsz, 0, doc, root_val)) < 0)
                        goto error;
                break;
        default:
                LITE3_PRINT_ERROR("FAILED TO READ JSON: EXPECTING ARRAY OR OBJECT TYPE\n");
                errno = EINVAL;
                goto error;
        }
        yyjson_doc_free(doc);
        return ret;
error:
        yyjson_doc_free(doc);
        return ret;
}

int lite3_json_dec(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, const char *restrict json_str, size_t json_len)
{
        yyjson_read_err err;
        yyjson_doc *doc = yyjson_read_opts((char *)json_str, json_len, YYJSON_READ_NOFLAG , NULL, &err);
        if (!doc) {
                LITE3_PRINT_ERROR("FAILED TO READ JSON STRING\tyyjson error code: %u\tmsg:%s\tat byte position: %lu\n", err.code, err.msg, err.pos);
                errno = EINVAL;
                return -1;
        }
        return _lite3_json_dec_doc(buf, out_buflen, bufsz, doc);
}

int lite3_json_dec_file(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, const char *restrict path)
{
        yyjson_read_err err;
        yyjson_doc *doc = yyjson_read_file(path, YYJSON_READ_NOFLAG , NULL, &err);
        if (!doc) {
                LITE3_PRINT_ERROR("FAILED TO READ JSON FILE\tyyjson error code: %u\tmsg:%s\tat byte position: %lu\n", err.code, err.msg, err.pos);
                errno = EINVAL;
                return -1;
        }
        return _lite3_json_dec_doc(buf, out_buflen, bufsz, doc);
}

int lite3_json_dec_fp(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, FILE *fp)
{
        yyjson_read_err err;
        yyjson_doc *doc = yyjson_read_fp(fp, YYJSON_READ_NOFLAG , NULL, &err);
        if (!doc) {
                LITE3_PRINT_ERROR("FAILED TO READ JSON FILE POINTER\tyyjson error code: %u\tmsg:%s\tat byte position: %lu\n", err.code, err.msg, err.pos);
                errno = EINVAL;
                return -1;
        }
        return _lite3_json_dec_doc(buf, out_buflen, bufsz, doc);
}
#endif // LITE3_JSON