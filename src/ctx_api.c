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
#include "lite3_context_api.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>



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



// If x is already a power of 2, the function returns x
static inline size_t next_power_of_2(size_t x)
{
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
#if SIZE_MAX == UINT64_MAX // Check if size_t is 64-bit
        x |= x >> 32;
#endif
        x++;
        return x;
}

static inline size_t clamp(size_t num, size_t min_val, size_t max_val) {
        return num < min_val ? min_val : (num > max_val ? max_val : num);
}

lite3_ctx *lite3_ctx_create_with_size(size_t bufsz)
{
        if (LITE3_UNLIKELY(bufsz > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return NULL;
        }
        lite3_ctx *ctx = malloc(sizeof(lite3_ctx));
        if (!ctx)
                return NULL;
        bufsz = (bufsz < LITE3_CONTEXT_BUF_SIZE_MIN) ? LITE3_CONTEXT_BUF_SIZE_MIN : bufsz;
        ctx->underlying_buf = malloc(bufsz);
        if (!ctx->underlying_buf) {
                free(ctx);
                return NULL;
        }
        ctx->buf = (u8 *)(((uintptr_t)ctx->underlying_buf + LITE3_NODE_ALIGNMENT_MASK) & ~LITE3_NODE_ALIGNMENT_MASK);
        ctx->buflen = 0;
        ctx->bufsz = (size_t)((uintptr_t)ctx->underlying_buf + (uintptr_t)bufsz - (uintptr_t)ctx->buf);
        return ctx;
}

lite3_ctx *lite3_ctx_create_from_buf(const unsigned char *buf, size_t buflen)
{
        if (LITE3_UNLIKELY(!(buf && buflen))) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: buffer cannot be empty or NULL\n");
                errno = EINVAL;
                return NULL;
        }
        if (LITE3_UNLIKELY(buflen > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return NULL;
        }
        size_t new_size = next_power_of_2(buflen + LITE3_NODE_ALIGNMENT_MASK);
        new_size = new_size == 0 ? LITE3_BUF_SIZE_MAX : new_size;
        new_size = clamp(new_size, LITE3_CONTEXT_BUF_SIZE_MIN, LITE3_BUF_SIZE_MAX);

        if (LITE3_UNLIKELY(buflen > new_size - LITE3_NODE_ALIGNMENT_MASK)) {
                LITE3_PRINT_ERROR("NEW SIZE OVERFLOW\n");
                errno = EOVERFLOW;
                return NULL;
        }
        lite3_ctx *ret = lite3_ctx_create_with_size(new_size);
        if (ret) {
                memcpy(ret->buf, buf, buflen);
                ret->buflen = buflen;
        }
        return ret;
}

lite3_ctx *lite3_ctx_create_take_ownership(unsigned char *buf, size_t buflen, size_t bufsz)
{
        if (LITE3_UNLIKELY(!(buf && bufsz))) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: buffer cannot be NULL or zero size\n");
                errno = EINVAL;
                return NULL;
        }
        if (LITE3_UNLIKELY(bufsz > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return NULL;
        }
        if (LITE3_UNLIKELY(buflen > bufsz)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: buflen > bufsz\n");
                errno = EINVAL;
                return NULL;
        }
        if (LITE3_UNLIKELY(bufsz < LITE3_CONTEXT_BUF_SIZE_MIN)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < LITE3_CONTEXT_BUF_SIZE_MIN\n");
                errno = EINVAL;
                return NULL;
        }
        if (LITE3_UNLIKELY(((uintptr_t)buf & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: *buf not aligned to LITE3_NODE_ALIGNMENT\n");
                errno = EINVAL;
                return NULL;
        }
        lite3_ctx *ret = malloc(sizeof(lite3_ctx));
        if (!ret)
                return NULL;
        ret->buf = buf;
        ret->buflen = buflen;
        ret->bufsz = bufsz;
        ret->underlying_buf = buf;
        return ret;
}


int lite3_ctx_grow_impl(lite3_ctx *ctx)
{
        if (ctx->bufsz >= LITE3_BUF_SIZE_MAX) {
                LITE3_PRINT_ERROR("MESSAGE SIZE: bufsz >= LITE3_BUF_SIZE_MAX\n");
                errno = EMSGSIZE;
                return -1;
        }
        size_t current_size = (size_t)((uintptr_t)ctx->buf - (uintptr_t)ctx->underlying_buf) + ctx->bufsz;
        size_t new_size = ctx->bufsz < (LITE3_BUF_SIZE_MAX / 4) ? (current_size << 2) : LITE3_BUF_SIZE_MAX; // Increase size by 4X up to LITE3_BUF_SIZE_MAX
        new_size = clamp(new_size, LITE3_CONTEXT_BUF_SIZE_MIN, LITE3_BUF_SIZE_MAX);

        if (LITE3_UNLIKELY(current_size > new_size - LITE3_NODE_ALIGNMENT_MASK)) {
                LITE3_PRINT_ERROR("NEW SIZE OVERFLOW\n");
                errno = EOVERFLOW;
                return -1;
        }
        void *new = malloc(new_size);
        if (!new)
                return -1;
        u8 *new_buf = (u8 *)(((uintptr_t)new + LITE3_NODE_ALIGNMENT_MASK) & ~LITE3_NODE_ALIGNMENT_MASK);
        memcpy(new_buf, ctx->buf, ctx->buflen);
        ctx->buf = new_buf;
        ctx->bufsz = (size_t)((uintptr_t)new + (uintptr_t)new_size - (uintptr_t)new_buf);
        free(ctx->underlying_buf);
        ctx->underlying_buf = new;
        errno = 0;
        return 0;
}

int lite3_ctx_import_from_buf(lite3_ctx *ctx, const unsigned char *buf, size_t buflen)
{
        if (LITE3_UNLIKELY(!(buf && buflen))) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: buffer cannot be empty or NULL\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(buflen > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return -1;
        }
        if (buflen > ctx->bufsz) {
                size_t new_size = next_power_of_2(buflen + LITE3_NODE_ALIGNMENT_MASK);
                new_size = new_size == 0 ? LITE3_BUF_SIZE_MAX : new_size;
                new_size = clamp(new_size, LITE3_CONTEXT_BUF_SIZE_MIN, LITE3_BUF_SIZE_MAX);

                if (LITE3_UNLIKELY(buflen > new_size - LITE3_NODE_ALIGNMENT_MASK)) {
                        LITE3_PRINT_ERROR("NEW SIZE OVERFLOW\n");
                        errno = EOVERFLOW;
                        return -1;
                }
                free(ctx->underlying_buf);
                void *new = malloc(new_size);
                if (!new)
                        return -1;
                ctx->underlying_buf = new;
                u8 *new_buf = (u8 *)(((uintptr_t)new + LITE3_NODE_ALIGNMENT_MASK) & ~LITE3_NODE_ALIGNMENT_MASK);
                ctx->buf = new_buf;
                ctx->bufsz = (size_t)((uintptr_t)new + (uintptr_t)new_size - (uintptr_t)new_buf);
        }
        ctx->buflen = buflen;
        memcpy(ctx->buf, buf, buflen);
        return 0;
}

void lite3_ctx_destroy(lite3_ctx *ctx)
{
        free(ctx->underlying_buf);
        ctx->buf = NULL;
        ctx->underlying_buf = NULL;
        free(ctx);
}