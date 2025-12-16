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

/**
Lite³ Context API Header

@file lite3_context_api.h
@author Elias de Jong
@copyright Copyright © 2025 Elias de Jong <elias@fastserial.com>
@date 2025-09-20
*/
#ifndef LITE3_CONTEXT_API_H
#define LITE3_CONTEXT_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include "lite3.h"



#ifdef __cplusplus
extern "C" {
#endif

/**
Context API

Lite³ provides an alternative API called the 'Context API'. Compared to the @ref lite3_buffer_api,
this API represents a more accessible alternative where memory allocations are hidden from the user,
providing all the same functionality without requiring manual buffer management.

Contexts are containers for Lite³ buffers, containing a single buffer for a single message.
Instead of buffers being passed directly, functions take a `*ctx` variable as argument.
The context will automatically resize when needed, similar to `std::vector` in C++.

If you want to access the buffer inside of a context, you can like so:
```C
ctx->buf        // (uint8_t *) buffer pointer
ctx->buflen     // (size_t) message length in bytes
```
There is also a `ctx->bufsz` member that stores the total capacity inside the allocation. Contexts start out with
a size of `LITE3_CONTEXT_BUF_SIZE_MIN` (default: 1024) and will reallocate 4X the current capacity if they run out of space.
This happens automatically, though you might occasionally see error messages in your terminal when this happens:
```text
NO BUFFER SPACE FOR ENTRY INSERTION
NO BUFFER SPACE FOR NODE SPLIT
```
This is nothing to worry about, as the context will use these errors to grow its size.

@note
Contexts will preserve `LITE3_NODE_ALIGNMENT` alignment during (re)allocation (see @ref lite3_node_alignment).

@warning
Automatic memory management requires some overhead and internal allocations might happen unexpectedly.
Therefore, it is not recommended to use the Context API for realtime or latency-sensitive applications.
If you must, then create an oversized context up front and keep reusing it indefinitely to avoid reallocations.

@par Error handling
Unless otherwise specified, Lite³ uses the POSIX error handling convention.
This means functions will return a value of `-1` and set `errno` to one of several values:

| Err    | Hex    | Var            | Description                            |
| ------ | ------ | -------------- | -------------------------------------- |
| 2      | 0x02   | ENOENT         | No such file or directory              |
| 5      | 0x05   | EIO            | Input/output error                     |
| 14     | 0x0e   | EFAULT         | Bad address                            |
| 22     | 0x16   | EINVAL         | Invalid argument                       |
| 74     | 0x4a   | EBADMSG        | Bad message                            |
| 75     | 0x4b   | EOVERFLOW      | Value too large for defined data type  |
| 90     | 0x5a   | EMSGSIZE       | Message too long                       |
| 105    | 0x69   | ENOBUFS        | No buffer space available              |

@par Used feature breaking ANSI-C / C89 compatibility:
C99:
- inline functions
- compound literals
- flexible array member (no specified size)
- single-line comments //
- variadic macros (__VA_ARGS__)
- restrict pointers     
- initializing a structure by field names: `struct Point p = { .x = 1, .y = 2 };`

C11:
- static_assert

@defgroup lite3_context_api Context API
*/



/**
The minimum buffer size for a Lite³ context

Must be greater than `LITE3_NODE_ALIGNMENT`

@ingroup lite3_config
*/
#define LITE3_CONTEXT_BUF_SIZE_MIN 1024
static_assert(LITE3_CONTEXT_BUF_SIZE_MIN > (size_t)LITE3_NODE_ALIGNMENT_MASK, "LITE3_CONTEXT_BUF_SIZE_MIN must be greater than LITE3_NODE_ALIGNMENT_MASK");

/**
Lite³ context struct.

See @ref lite3_context_api.

@ingroup lite3_types
*/
typedef struct lite3_ctx {
        uint8_t __attribute__((aligned(LITE3_NODE_ALIGNMENT))) *buf;
        size_t buflen;
        size_t bufsz;
        void *underlying_buf;
} lite3_ctx;



/**
Managing Contexts

See @ref lite3_context_api.
This section contains various functions for creating and destroying contexts, as well as importing data.

@defgroup lite3_ctx_manage Managing Contexts
@ingroup lite3_context_api
@{
*/
/**
Create context with custom size

If you know that you will be storing a large message, it is more efficient to allocate a large context up front.
Otherwise, a small default context will copy and relocate data several times similar to `std::vector` in C++.

@param[in]      bufsz (`size_t`) context buffer size

@warning
1. Any context that is not reused for the lifetime of the program must be manually destroyed by `lite3_ctx_destroy()` to prevent memory leaks.
2. Any value for `bufsz` below `LITE3_CONTEXT_BUF_SIZE_MIN` will be automatically clamped to `LITE3_CONTEXT_BUF_SIZE_MIN`.
Any size above `LITE3_BUF_SIZE_MAX` will trigger an error.
*/
lite3_ctx *lite3_ctx_create_with_size(size_t bufsz);

/**
Create context by copying from a buffer

This function will copy data into a newly allocated context. The original data is not affected.

@warning
Any context that is not reused for the lifetime of the program must be manually destroyed by `lite3_ctx_destroy()` to prevent memory leaks.
*/
lite3_ctx *lite3_ctx_create_from_buf(
        const unsigned char *buf,       ///< [in] source buffer pointer
        size_t buflen                   ///< [in] source buffer used length (length of message)
);

/**
Create context by taking ownership of a buffer

When you have an existing allocation containing a Lite³ message,
you might want a context to to take ownership over it rather than copying all the data.
The passed buffer will also be freed when you call `lite3_ctx_destroy()` on the context later.

@warning
1. Any context that is not reused for the lifetime of the program must be manually destroyed by `lite3_ctx_destroy()` to prevent memory leaks.
2. `*buf` must be N-byte aligned according to `LITE3_NODE_ALIGNMENT`.
On 64-bit machines, buffers allocated by libc `malloc()` always work with the default Lite³ configuration.
Alignment should only become an issue if you use custom alignment (see @ref lite3_node_alignment).
*/
lite3_ctx *lite3_ctx_create_take_ownership(
        unsigned char *buf,     ///< [in] buffer pointer (created by `malloc()`)
        size_t buflen,          ///< [in] buffer used length (length of message)
        size_t bufsz            ///< [in] buffer max size (total size of allocation)
);

/**
Create context with minimum size

Creates a context with default size of `LITE3_CONTEXT_BUF_SIZE_MIN`.

@warning
Any context that is not reused for the lifetime of the program must be manually destroyed by `lite3_ctx_destroy()` to prevent memory leaks.
*/
static inline lite3_ctx *lite3_ctx_create(void) {
        return lite3_ctx_create_with_size(LITE3_CONTEXT_BUF_SIZE_MIN);
}

/**
Copy data into existing context

This function allows for efficient reuse of contexts.
For example, a listening server may want to copy packet data into an existing context.
If the new data fits into the existing context, then no new calls to `malloc()` are needed.
This avoids repeated allocations from creating and destroying of contexts.

@return 0 on success
@return < 0 on error
*/
int lite3_ctx_import_from_buf(
        lite3_ctx *ctx,                 ///< [in] context pointer
        const unsigned char *buf,       ///< [in] source buffer pointer
        size_t buflen                   ///< [in] source buffer used length (length of message)
);

/**
Destroy context

@param[in]      ctx (`lite3_ctx *`) context pointer

@warning
Any context that is not reused for the lifetime of the program must be manually destroyed by `lite3_ctx_destroy()` to prevent memory leaks.
*/
void lite3_ctx_destroy(lite3_ctx *ctx);

#ifndef DOXYGEN_IGNORE
// Private function
int lite3_ctx_grow_impl(lite3_ctx *ctx);
#endif // DOXYGEN_IGNORE
/// @} lite3_ctx_manage



/**
Object / array initialization

The JSON standard requires that the root-level type always be an 'object' or 'array'. This also applies to Lite³.

Before data can be inserted into an empty buffer, it must first be initialized as object or array.

@defgroup lite3_ctx_init Object / Array Initialization
@ingroup lite3_context_api
@{
*/
/**
Initialize a Lite³ context as an object

@return 0 on success
@return < 0 on error

@note
This function can also be used to reset an existing Lite³ message; the root node is simply replaced with an empty object.
*/
static inline int lite3_ctx_init_obj(lite3_ctx *ctx)
{
        return lite3_init_obj(ctx->buf, &ctx->buflen, ctx->bufsz);
}

/**
Initialize a Lite³ context as an array

@return 0 on success
@return < 0 on error

@note
This function can also be used to reset an existing Lite³ message; the root node is simply replaced with an empty array.
*/
static inline int lite3_ctx_init_arr(lite3_ctx *ctx)
{
        return lite3_init_arr(ctx->buf, &ctx->buflen, ctx->bufsz);
}
/// @} lite3_ctx_init



/**
Set key-value pair in object

An empty buffer must first be initialized using `lite3_ctx_init_obj()` or `lite3_ctx_init_arr()` before insertion. See @ref lite3_ctx_init.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@note
Inserting a value with an existing key will override the current value.

@warning
1. Insertions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase message size.
3. Overriding any value with an existing key can still grow the buffer and increase message size.
4. Overriding a **variable-length value (string/bytes)** will require extra buffer space if the new value is larger than the old.
The overridden space is never recovered, causing buffer size to grow indefinitely.

@defgroup lite3_ctx_obj_set Object Set
@ingroup lite3_context_api
@{
*/
/**
Set null in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_null(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        lite3_ctx_set_null_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline int lite3_ctx_set_null_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set boolean in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      value (`bool`) boolean value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_bool(ctx, ofs, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_bool_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_bool_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, bool value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_BOOL;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_BOOL]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set integer in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      value (`uint64_t`) integer value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_i64(ctx, ofs, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_i64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_i64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, int64_t value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_I64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set floating point in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      value (`double`) floating point value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_f64(ctx, ofs, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_f64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_f64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, double value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_F64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set bytes in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      bytes (`const unsigned char *`) bytes pointer
@param[in]      bytes_len (`size_t`) bytes amount

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_bytes(ctx, ofs, key, bytes, bytes_len) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_bytes_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), bytes, bytes_len); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_bytes_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, const unsigned char *__restrict bytes, size_t bytes_len)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_BYTES;
        memcpy(val->val, &bytes_len, lite3_type_sizes[LITE3_TYPE_BYTES]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_BYTES], bytes, bytes_len);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set string in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      str (`const char *`) string pointer

@return 0 on success
@return < 0 on error

@note
This function must call `strlen()` to learn the size of the string.
If you know the length beforehand, it is more efficient to call `lite3_ctx_set_str_n()`.
*/
#define lite3_ctx_set_str(ctx, ofs, key, str) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_str_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), str); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_str_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, const char *__restrict str)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_size);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set string in object by length

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[in]      str (`const char *`) string pointer
@param[in]      str_len (`size_t`) string length, exclusive of NULL-terminator.

@return 0 on success
@return < 0 on error

@warning
`str_len` is exclusive of the NULL-terminator.
*/
#define lite3_ctx_set_str_n(ctx, ofs, key, str, str_len) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_set_str_n_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), str, str_len); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_str_n_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, const char *__restrict str, size_t str_len)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        lite3_val *val;
        size_t str_size = str_len + 1;
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_len);
        *(val->val + lite3_type_sizes[LITE3_TYPE_STRING] + str_len) = 0x00; // Insert NULL-terminator
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set object in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out_ofs (`size_t *`) offset of the newly inserted object (if not needed, pass `NULL`)

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_obj(ctx, ofs, key, out_ofs) ({ \
        lite3_ctx *__lite3_ctx__ = (ctx); \
        size_t __lite3_ofs__ = (ofs); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_obj_set( \
                __lite3_ctx__->buf, \
                &__lite3_ctx__->buflen, \
                __lite3_ofs__, \
                __lite3_ctx__->bufsz)) < 0) \
                return __lite3_ret__; \
        \
        const char *__lite3_key__ = (key); \
        errno = 0; \
        while ((__lite3_ret__ = lite3_set_obj_impl( \
                __lite3_ctx__->buf, \
                &__lite3_ctx__->buflen, \
                __lite3_ofs__, \
                __lite3_ctx__->bufsz, \
                __lite3_key__, \
                LITE3_KEY_DATA(key), \
                out_ofs)) < 0) { \
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(__lite3_ctx__) == 0)) { \
                        continue; \
                } else { \
                        return __lite3_ret__; \
                } \
        } \
        __lite3_ret__; \
})

/**
Set array in object

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out_ofs (`size_t *`) offset of the newly inserted array (if not needed, pass `NULL`)

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_set_arr(ctx, ofs, key, out_ofs) ({ \
        lite3_ctx *__lite3_ctx__ = (ctx); \
        size_t __lite3_ofs__ = (ofs); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_obj_set( \
                __lite3_ctx__->buf, \
                &__lite3_ctx__->buflen, \
                __lite3_ofs__, \
                __lite3_ctx__->bufsz)) < 0) \
                return __lite3_ret__; \
        \
        const char *__lite3_key__ = (key); \
        errno = 0; \
        while ((__lite3_ret__ = lite3_set_arr_impl( \
                __lite3_ctx__->buf, \
                &__lite3_ctx__->buflen, \
                __lite3_ofs__, \
                __lite3_ctx__->bufsz, \
                __lite3_key__, \
                LITE3_KEY_DATA(key), \
                out_ofs)) < 0) { \
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(__lite3_ctx__) == 0)) { \
                        continue; \
                } else { \
                        return __lite3_ret__; \
                } \
        } \
        __lite3_ret__; \
})
/// @} lite3_ctx_obj_set



/**
Append value to array

An empty buffer must first be initialized using `lite3_ctx_init_obj()` or `lite3_ctx_init_arr()` before insertion. See @ref lite3_ctx_init.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
1. Append actions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase message size.

@defgroup lite3_ctx_arr_append Array Append
@ingroup lite3_context_api
@{
*/
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_set_by_index(lite3_ctx *ctx, size_t ofs, uint32_t index, size_t val_len, lite3_val **out)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        uint32_t size = (*(uint32_t *)(ctx->buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        if (LITE3_UNLIKELY(index > size)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
                errno = EINVAL;
                return -1;
        }
        lite3_key_data key_data = {
                .hash = index,
                .size = 0,
        };
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, NULL, key_data, val_len, out)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

static inline int _lite3_ctx_set_by_append(lite3_ctx *ctx, size_t ofs, size_t val_len, lite3_val **out)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        uint32_t size = (*(uint32_t *)(ctx->buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        lite3_key_data key_data = {
                .hash = size,
                .size = 0,
        };
        errno = 0;
        while ((ret = lite3_set_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, NULL, key_data, val_len, out)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Append null to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_null(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs)             ///< [in] start offset (0 == root)
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}

/**
Append boolean to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_bool(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        bool value)             ///< [in] boolean value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BOOL;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_BOOL]);
        return ret;
}

/**
Append integer to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_i64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        int64_t value)          ///< [in] integer value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_I64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}

/**
Append floating point to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_f64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        double value)           ///< [in] floating point value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_F64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}

/**
Append bytes to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_bytes(
        lite3_ctx *ctx,                         ///< [in] context pointer
        size_t ofs,                             ///< [in] start offset (0 == root)
        const unsigned char *__restrict bytes,  ///< [in] bytes pointer
        size_t bytes_len)                         ///< [in] bytes amount
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BYTES;
        memcpy(val->val, &bytes_len, lite3_type_sizes[LITE3_TYPE_BYTES]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_BYTES], bytes, bytes_len);
        return ret;
}

/**
Append string to array

@return 0 on success
@return < 0 on error

@note
This function must call `strlen()` to learn the size of the string.
If you know the length beforehand, it is more efficient to call `lite3_set_str_n()`.
*/
static inline int lite3_ctx_arr_append_str(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        const char *__restrict str)     ///< [in] string pointer
{
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_size);
        return ret;
}

/**
Append string to array by length

@return 0 on success
@return < 0 on error

@warning
`str_len` is exclusive of the NULL-terminator.
*/
static inline int lite3_ctx_arr_append_str_n(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        const char *__restrict str,     ///< [in] string pointer
        size_t str_len)                 ///< [in] string length, exclusive of NULL-terminator.
{
        lite3_val *val;
        size_t str_size = str_len + 1;
        int ret;
        if ((ret = _lite3_ctx_set_by_append(ctx, ofs, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_len);
        *(val->val + lite3_type_sizes[LITE3_TYPE_STRING] + str_len) = 0x00; // Insert NULL-terminator
        return ret;
}

/**
Append object to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_obj(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_ofs)     ///< [out] offset of the newly appended object (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        errno = 0;
        while ((ret = lite3_arr_append_obj_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, out_ofs)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

/**
Append array to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_append_arr(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_ofs)     ///< [out] offset of the newly appended array (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        errno = 0;
        while ((ret = lite3_arr_append_arr_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, out_ofs)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}
/// @} lite3_ctx_arr_append


/**
Set (or overwrite) value in array

An empty buffer must first be initialized using `lite3_ctx_init_obj()` or `lite3_ctx_init_arr()` before insertion. See @ref lite3_ctx_init.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@note
Setting a value at an existing index will override the current value.

@warning
1. Insertions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase message size.
3. Overriding any value with an existing key can still grow the buffer and increase message size.
4. Overriding a **variable-length value (string/bytes)** will require extra buffer space if the new value is larger than the old.
The overridden space is never recovered, causing buffer size to grow indefinitely.

@defgroup lite3_ctx_arr_set Array Set
@ingroup lite3_context_api
@{
*/
/**
Set null in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_null(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index)         ///< [in] array index
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}

/**
Set boolean in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_bool(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        bool value)             ///< [in] boolean value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BOOL;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_BOOL]);
        return ret;
}

/**
Set integer in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_i64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        int64_t value)          ///< [in] integer value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_I64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}

/**
Set float in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_f64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        double value)           ///< [in] floating point value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_F64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}

/**
Set bytes in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_bytes(
        lite3_ctx *ctx,                         ///< [in] context pointer
        size_t ofs,                             ///< [in] start offset (0 == root)
        uint32_t index,                         ///< [in] array index
        const unsigned char *__restrict bytes,  ///< [in] bytes pointer
        size_t bytes_len)                         ///< [in] bytes amount
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BYTES;
        memcpy(val->val, &bytes_len, lite3_type_sizes[LITE3_TYPE_BYTES]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_BYTES], bytes, bytes_len);
        return ret;
}

/**
Set string in array

@return 0 on success
@return < 0 on error

@note
This function must call `strlen()` to learn the size of the string.
If you know the length beforehand, it is more efficient to call `lite3_set_str_n()`.
*/
static inline int lite3_ctx_arr_set_str(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        const char *__restrict str)     ///< [in] string pointer
{
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_size);
        return ret;
}

/**
Set string in array by length

@return 0 on success
@return < 0 on error

@warning
`str_len` is exclusive of the NULL-terminator.
*/
static inline int lite3_ctx_arr_set_str_n(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        const char *__restrict str,     ///< [in] string pointer
        size_t str_len)                 ///< [in] string length, exclusive of NULL-terminator.
{
        lite3_val *val;
        size_t str_size = str_len + 1;
        int ret;
        if ((ret = _lite3_ctx_set_by_index(ctx, ofs, index, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_len);
        *(val->val + lite3_type_sizes[LITE3_TYPE_STRING] + str_len) = 0x00; // Insert NULL-terminator
        return ret;
}

/**
Set object in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_obj(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] offset of the newly inserted object (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        errno = 0;
        while ((ret = lite3_arr_set_obj_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, index, out_ofs)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

/**
Set array in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_set_arr(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] offset of the newly inserted array (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(ctx->buf, &ctx->buflen, ofs, ctx->bufsz)) < 0)
                return ret;
        errno = 0;
        while ((ret = lite3_arr_set_arr_impl(ctx->buf, &ctx->buflen, ofs, ctx->bufsz, index, out_ofs)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}
/// @} lite3_ctx_arr_set


/**
Utility functions

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@warning
Read-only operations are thread-safe. This includes all utility funtions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_ctx_utility Utility Functions
@ingroup lite3_context_api
@{
*/
/**
View the internal structure of a Lite³ buffer

Requires `LITE3_DEBUG` enabled. See @ref lite3_config.
*/
#ifdef LITE3_DEBUG
static inline void lite3_ctx_print(lite3_ctx *ctx) { lite3_print(ctx->buf, ctx->buflen); }
#else
static inline void lite3_ctx_print(lite3_ctx *ctx) { (void)ctx; }
#endif

/**
Find value by key and return value type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return lite3_type on success
@return `LITE3_TYPE_INVALID` on error (key cannot be found)
*/
#define lite3_ctx_get_type(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_type_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline enum lite3_type _lite3_ctx_get_type_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return LITE3_TYPE_INVALID;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return LITE3_TYPE_INVALID;
        enum lite3_type type = (enum lite3_type)val->type;
        return type;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and write back type size

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size *`) type size

@return 0 on success
@return < 0 on error

@note
For variable sized types like `LITE3_TYPE_BYTES` or `LITE3_TYPE_STRING`, the number of bytes (including NULL-terminator for string) are written back.
*/
#define lite3_ctx_get_type_size(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_type_size_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_type_size_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (val->type == LITE3_TYPE_STRING || val->type == LITE3_TYPE_BYTES) {
                *out = 0;
                memcpy(out, &val->val, lite3_type_sizes[LITE3_TYPE_BYTES]);
                return ret;
        }
        *out = lite3_type_sizes[val->type];
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Attempt to find a key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` on success
@return `false` on failure
*/
#define lite3_ctx_exists(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_exists_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_exists_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return true;
}
#endif // DOXYGEN_IGNORE

/**
Write back the number of object entries or array elements

This function can be called on objects and arrays.

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_count(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t *out)          ///< [out] number of object entries or array elements
{
        int ret;
        if ((ret = _lite3_verify_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        enum lite3_type type = (enum lite3_type)(*(ctx->buf + ofs));
        if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
                errno = EINVAL;
                return -1;
        }
        *out = (*(uint32_t *)(ctx->buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        return ret;
}

/**
Find value by key and test for null type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_null(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_null_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_null_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_NULL;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for bool type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_bool(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_bool_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_bool_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_BOOL;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for integer type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_i64(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_i64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_i64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_I64;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for floating point type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_f64(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_f64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_f64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_F64;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for bytes type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_bytes(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_bytes_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_bytes_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_BYTES;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for string type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_str(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_str_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_str_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_STRING;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for object type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_obj(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_obj_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_obj_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_OBJECT;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for array type

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_ctx_is_arr(ctx, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_is_arr_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_ctx_is_arr_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_ARRAY;
}
#endif // DOXYGEN_IGNORE
/// @} lite3_ctx_utility



/**
Get value from object by key

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
Read-only operations are thread-safe. This includes all `lite3_get_xxx()` funtions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_ctx_get Object Get
@ingroup lite3_context_api
@{
*/
/**
Get value from object

Unlike other `lite3_ctx_get_xxx()` functions, this function does not get a specific type.
Instead, it produces a generic `lite3_val` pointer, which points to a value inside the Lite³ buffer.
This can be useful in cases where you don't know the exact type of a value beforehand. See @ref lite3_val_fns.

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_val *`) opaque value pointer

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get(ctx, ofs, key, out) ({ \
        lite3_ctx *__lite3_ctx__ = (ctx); \
        size_t __lite3_ofs__ = (ofs); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_get(__lite3_ctx__->buf, __lite3_ctx__->buflen, __lite3_ofs__)) < 0) \
                return __lite3_ret__; \
        const char *__lite3_key__ = (key); \
        lite3_get_impl(__lite3_ctx__->buf, __lite3_ctx__->buflen, __lite3_ofs__, __lite3_key__, LITE3_KEY_DATA(key), out); \
})

/**
Get boolean value by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`bool *`) boolean value

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_bool(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_bool_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_bool_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, bool *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BOOL)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BOOL\n");
                errno = EINVAL;
                return -1;
        }
        *out = (bool)(*(val->val));
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get integer value by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`int64_t *`) integer value

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_i64(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_i64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_i64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, int64_t *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_I64)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_I64\n");
                errno = EINVAL;
                return -1;
        }
        memcpy(out, val->val, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get floating point value by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`double *`) floating point value

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_f64(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_f64_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_f64_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, double *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_F64)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_F64\n");
                errno = EINVAL;
                return -1;
        }
        memcpy(out, val->val, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get bytes value by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_bytes *`) bytes value

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_bytes(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_bytes_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_bytes_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, lite3_bytes *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BYTES)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BYTES\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_bytes){
                .gen = *(uint32_t *)(ctx->buf),
                .len = 0,
                .ptr = val->val + lite3_type_sizes[LITE3_TYPE_BYTES]
        };
        memcpy(&out->len, val->val, lite3_type_sizes[LITE3_TYPE_BYTES]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get string value by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_str *`) string value

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_str(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_str_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_str_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, lite3_str *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_STRING)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_STRING\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_str){
                .gen = *(uint32_t *)(ctx->buf),
                .len = 0,
                .ptr = (char *)(val->val + lite3_type_sizes[LITE3_TYPE_STRING])
        };
        memcpy(&out->len, val->val, lite3_type_sizes[LITE3_TYPE_STRING]);
        --out->len; // Lite³ stores string size including NULL-terminator. Correction required for public API.
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get object by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size_t *`) object offset

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_obj(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_obj_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_obj_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_OBJECT\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - ctx->buf);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get array by key

@param[in]      ctx (`lite3_ctx *`) context pointer
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size_t *`) array offset

@return 0 on success
@return < 0 on error
*/
#define lite3_ctx_get_arr(ctx, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_ctx_get_arr_impl(ctx, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_ctx_get_arr_impl(lite3_ctx *ctx, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(ctx->buf, ctx->buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_ARRAY\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - ctx->buf);
        return ret;
}
#endif // DOXYGEN_IGNORE
/// @} lite3_ctx_get



/**
Get value from array by index

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
Read-only operations are thread-safe. This includes all `lite3_arr_get_xxx()` funtions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_ctx_arr_get Array Get
@ingroup lite3_context_api
@{
*/
/**
Get boolean value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_bool(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        bool *out)              ///< [out] boolean value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BOOL)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BOOL\n");
                errno = EINVAL;
                return -1;
        }
        *out = (bool)(*(val->val));
        return ret;
}

/**
Get integer value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_i64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        int64_t *out)           ///< [out] integer value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_I64)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_I64\n");
                errno = EINVAL;
                return -1;
        }
        memcpy(out, val->val, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}

/**
Get floating point value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_f64(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        double *out)            ///< [out] floating point value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_F64)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_F64\n");
                errno = EINVAL;
                return -1;
        }
        memcpy(out, val->val, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}

/**
Get bytes value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_bytes(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        lite3_bytes *out)       ///< [out] bytes value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BYTES)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BYTES\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_bytes){
                .gen = *(uint32_t *)(ctx->buf),
                .len = 0,
                .ptr = val->val + lite3_type_sizes[LITE3_TYPE_BYTES]
        };
        memcpy(&out->len, val->val, lite3_type_sizes[LITE3_TYPE_BYTES]);
        return ret;
}

/**
Get string value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_str(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t index,         ///< [in] array index
        lite3_str *out)         ///< [out] string value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_STRING)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_STRING\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_str){
                .gen = *(uint32_t *)(ctx->buf),
                .len = 0,
                .ptr = (char *)(val->val + lite3_type_sizes[LITE3_TYPE_STRING])
        };
        memcpy(&out->len, val->val, lite3_type_sizes[LITE3_TYPE_STRING]);
        --out->len; // Lite³ stores string size including NULL-terminator. Correction required for public API.
        return ret;
}

/**
Get object by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_obj(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] object offset
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_OBJECT\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - (ctx->buf));
        return ret;
}

/**
Get array by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_arr_get_arr(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] array offset
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(ctx->buf, ctx->buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_ARRAY\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - (ctx->buf));
        return ret;
}
/// @} lite3_ctx_arr_get



/**
Create and use iterators for objects/arrays

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@warning
Read-only operations are thread-safe. This includes all iterator funtions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_ctx_iter Iterators
@ingroup lite3_context_api
@{
*/
#ifdef DOXYGEN_ONLY
/// Return value of `lite3_ctx_iter_next()`; iterator produced an item, continue;
#define LITE3_ITER_ITEM 1
/// Return value of `lite3_ctx_iter_next()`; iterator finished; stop.
#define LITE3_ITER_DONE 0
#endif

/**
Create a lite3 iterator for the given object or array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_ctx_iter_create(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs,             ///< [in] start offset (0 == root)
        lite3_iter *out)        ///< [out] iterator struct pointer
{
        int ret;
        if ((ret = _lite3_verify_get(ctx->buf, ctx->buflen, ofs)) < 0)
                return ret;
        return lite3_iter_create_impl(ctx->buf, ctx->buflen, ofs, out);
}

/**
Get the next item from a lite3 iterator

To use in conjunctions with @ref lite3_val_fns, the `*out_val_ofs` can be casted to `(lite3_val *)`.

@return `LITE3_ITER_ITEM` (== 1) on item produced
@return `LITE3_ITER_DONE` (== 0) on success (no more items)
@return < 0 on error

@note
If the user does not want a key or value, simply pass NULL. For array iterators, `out_key` should always be passed as `NULL`.

@warning
Iterators are read-only. Any attempt to write to the buffer using `lite3_ctx_set_xxxx()` will immediately invalidate the iterator.
If you need to make changes to the buffer, first prepare your changes, then apply them afterwards in one batch.
*/
static inline int lite3_ctx_iter_next(
        lite3_ctx *ctx,         ///< [in] context pointer
        lite3_iter *iter,       ///< [in] iterator struct pointer
        lite3_str *out_key,     ///< [out] current key (if not needed, pass `NULL`)
        size_t *out_val_ofs)    ///< [out] current value offset (if not needed, pass `NULL`)
{
        return lite3_iter_next(ctx->buf, ctx->buflen, iter, out_key, out_val_ofs);
}
/// @} lite3_ctx_iter


/**
Conversion between Lite³ and JSON

All JSON functionality is enabled internally by the `yyjson` library.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Number reading
Lite³ implements an integer and floating point type, while JSON only implements a single 'number' type. Some rules apply to JSON number reading:
- Numbers without a decimal point are read as `int64_t`.
- Numbers with a decimal point are read as `double` with correct rounding.
- If a number is too large for `int64_t`, it is converted to `double`.
- If a `double` number overflows (reaches infinity), an error is returned.
- If a number does not conform to the JSON standard, an error is returned.

See: `enum lite3_type`

@note
This feature requires subdependencies enabled via build flags. See the `Makefile` for more details.
Also `LITE3_JSON` must be defined inside the `lite3.h` header or using compiler `-D` flags. See @ref lite3_config.

@warning
Read-only operations are thread-safe. This includes all JSON encode funtions.
Decoding JSON however is not thread-safe for the given Lite³ buffer and requires manual locking.
Mixing reads and writes on the same Lite³ buffer is also not thread-safe.

@defgroup lite3_ctx_json JSON Conversion
@ingroup lite3_context_api
@{
*/
#if defined(DOXYGEN_ONLY) && !defined(LITE3_JSON)
#define LITE3_JSON
#endif // DOXYGEN_ONLY

#ifdef LITE3_JSON
/**
Convert JSON string to Lite³

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
static inline int lite3_ctx_json_dec(
        lite3_ctx *ctx,                 ///< [in] context pointer
        const char *__restrict json_str,///< [in] JSON input string (string)
        size_t json_len)                ///< [in] JSON input string length (bytes, including or excluding NULL-terminator)
{
        int ret;
        errno = 0;
        while ((ret = lite3_json_dec(ctx->buf, &ctx->buflen, ctx->bufsz, json_str, json_len)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

/**
Convert JSON from file path to Lite³

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
static inline int lite3_ctx_json_dec_file(
        lite3_ctx *ctx,                 ///< [in] context pointer
        const char *__restrict path)    ///< [in] JSON file path (string)
{
        int ret;
        errno = 0;
        while ((ret = lite3_json_dec_file(ctx->buf, &ctx->buflen, ctx->bufsz, path)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

/**
Convert JSON from file pointer to Lite³

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
static inline int lite3_ctx_json_dec_fp(
        lite3_ctx *ctx,         ///< [in] context pointer
        FILE *fp)               ///< [in] JSON file pointer
{
        int ret;
        errno = 0;
        while ((ret = lite3_json_dec_fp(ctx->buf, &ctx->buflen, ctx->bufsz, fp)) < 0) {
                if (errno == ENOBUFS && (lite3_ctx_grow_impl(ctx) == 0)) {
                        continue;
                } else {
                        return ret;
                }
        }
        return ret;
}

/**
Print Lite³ buffer as JSON to `stdout`

@return 0 on success
@return < 0 on error

Very useful to make Lite³ structures human-readable inside a terminal.
The `ofs` parameter can be used to selectively print an internal object or array. To print out the entire structure, call from the root using `ofs == 0`.

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.
*/
static inline int lite3_ctx_json_print(
        lite3_ctx *ctx,         ///< [in] context pointer
        size_t ofs)             ///< [in] start offset (0 == root)
{
        return lite3_json_print(ctx->buf, ctx->buflen, ofs);
}

/**
Convert Lite³ to JSON string

@return `char *` pointer to the JSON string on success
@return `NULL` on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.

@warning
You must manually call `free()` on the pointer returned by this function.
*/
static inline char *lite3_ctx_json_enc(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_len)     ///< [out] string length excluding NULL-terminator (if not needed, pass `NULL`)
{
        return lite3_json_enc(ctx->buf, ctx->buflen, ofs, out_len);
}

/**
Convert Lite³ to JSON prettified string

The prettified string uses a tab space indent of 4.

@return `char *` pointer to the JSON string on success
@return `NULL` on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.

@warning
You must manually call `free()` on the pointer returned by this function.
*/
static inline char *lite3_ctx_json_enc_pretty(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_len)     ///< [out] string length excluding NULL-terminator (if not needed, pass `NULL`)
{
        return lite3_json_enc_pretty(ctx->buf, ctx->buflen, ofs, out_len);
}

/**
Convert Lite³ to JSON and write to output buffer

@return >= 0 on success (number of bytes written)
@return < 0 on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.
*/
static inline int64_t lite3_ctx_json_enc_buf(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        char *__restrict json_buf,      ///< [in] JSON output buffer
        size_t json_bufsz               ///< [in] JSON output buffer max size (bytes)
)
{
        return lite3_json_enc_buf(ctx->buf, ctx->buflen, ofs, json_buf, json_bufsz);
}

/**
Convert Lite³ to prettified JSON and write to output buffer

The prettified string uses a tab space indent of 4.

@return >= 0 on success (number of bytes written)
@return < 0 on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.
*/
static inline int64_t lite3_ctx_json_enc_pretty_buf(
        lite3_ctx *ctx,                 ///< [in] context pointer
        size_t ofs,                     ///< [in] start offset (0 == root)
        char *__restrict json_buf,      ///< [in] JSON output buffer
        size_t json_bufsz)              ///< [in] JSON output buffer max size (bytes)
{
        return lite3_json_enc_pretty_buf(ctx->buf, ctx->buflen, ofs, json_buf, json_bufsz);
}
#else
static inline int lite3_ctx_json_dec(lite3_ctx *ctx, const char *__restrict json_str, size_t json_len)
{
        (void)ctx; (void)json_str; (void)json_len;
        return -1;
}

static inline int lite3_ctx_json_dec_file(lite3_ctx *ctx, const char *__restrict path)
{
        (void)ctx; (void)path;
        return -1;
}

static inline int lite3_ctx_json_dec_fp(lite3_ctx *ctx, FILE *fp)
{
        (void)ctx; (void)fp;
        return -1;
}

static inline int lite3_ctx_json_print(lite3_ctx *ctx, size_t ofs)
{
        (void)ctx; (void)ofs;
        return 0;
}

static inline char *lite3_ctx_json_enc(lite3_ctx *ctx, size_t ofs, size_t *out_len)
{
        (void)ctx; (void)ofs; (void)out_len;
        return NULL;
}

static inline char *lite3_ctx_json_enc_pretty(lite3_ctx *ctx, size_t ofs, size_t *out_len)
{
        (void)ctx; (void)ofs; (void)out_len;
        return NULL;
}

static inline int64_t lite3_ctx_json_enc_buf(lite3_ctx *ctx, size_t ofs, char *__restrict json_buf, size_t json_bufsz)
{
        (void)ctx; (void)ofs; (void)json_buf; (void)json_bufsz;
        return -1;
}

static inline int64_t lite3_ctx_json_enc_pretty_buf(lite3_ctx *ctx, size_t ofs, char *__restrict json_buf, size_t json_bufsz)
{
        (void)ctx; (void)ofs; (void)json_buf; (void)json_bufsz;
        return -1;
}
#endif
/// @} lite3_ctx_json

#ifdef __cplusplus
}
#endif

#endif // LITE3_CONTEXT_API_H