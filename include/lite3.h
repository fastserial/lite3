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

/**
Lite³ Buffer API Header

@file lite3.h
@author Elias de Jong
@copyright MIT License - Copyright © 2025 Elias de Jong <elias@fastserial.com>
@date 2025-09-20
*/
#ifndef LITE3_H
#define LITE3_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <errno.h>



#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
    defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(__or1k__) || defined(__OR1K__)
    #error "Byte order must be little-endian"
#endif

static_assert(sizeof(double) == 8, "Double must be 8 bytes");

#ifndef DOXYGEN_IGNORE
#define LITE3_LIKELY(expr)   __builtin_expect(!!(expr), 1)
#define LITE3_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif // DOXYGEN_IGNORE



#ifdef __cplusplus
extern "C" {
#endif

/**
Buffer API

Functions in the Buffer API use caller-provided buffers.
Some scenarios where this is useful:
1. the user points to a buffer and Lite³ serializes directly into it;
2. the user points to an existing Lite³ message to perform lookups, iterators or mutations on it directly;
3. the user does not tolerate unexpected latency variations from automatic memory management / reallocation.

Overall, maximum control is given to the user. This also means it is the user's responsibility to allocate enough memory, and retry if necessary.

All functions include handles for proper retry logic. When a mutation fails because of insufficient buffer space,
the function will return `-1` and set `errno` to the `ENOBUFS` signal. After this, the caller can allocate more space and try again.

@note
If you are using Lite³ for the first time, it is recommended to start with the @ref lite3_context_api.
This API represents a more accessible alternative where memory allocations are hidden from the user,
providing all the same functionality without requiring manual buffer management.

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

@defgroup lite3_buffer_api Buffer API
*/



/**
Configuration options for the Lite³ library.

Configuration options can be toggled either by manually (un)commenting `#define` inside the header `include/lite3.h`, or by passing `-D` flags to your compiler.

For example, library error messages are disabled by default. However it is recommended to enable them to receive feedback during development. To do this, either:
1. uncomment the line `// #define LITE3_ERROR_MESSAGES` inside the header file: `include/lite3.h`
2. build the library using compilation flag `-DLITE3_ERROR_MESSAGES`

@defgroup lite3_config Library Configuration Options
@{
*/
#ifndef DOXYGEN_IGNORE
#define LITE3_ZERO_MEM_8  0x00
#define LITE3_ZERO_MEM_32 0x00000000
#endif // DOXYGEN_IGNORE

/**
Overwrite deleted values with NULL bytes (0x00).

Enabled by default.

This is a safety feature since not doing so would leave 'deleted' entries intact inside the datastructure until they are overwritten by other values.
Disable if you do not care about leaking deleted data.

@note When following canonical encoding rules, features `LITE3_ZERO_MEM_DELETED` and `LITE3_ZERO_MEM_EXTRA` are required.
*/
#define LITE3_ZERO_MEM_DELETED

#if defined(DOXYGEN_ONLY) && !defined(LITE3_ZERO_MEM_DELETED)
#define LITE3_ZERO_MEM_DELETED
#endif // DOXYGEN_ONLY

/**
Overwrite any unused bytes inside the Lite³ buffer with NULL bytes (0x00).

Enabled by default.

This is a safety feature to prevent leaking uninitialized memory bytes into messages.
It is also useful for debugging, as it makes the structure a lot more readable.
If you plan to use compression, this option makes Lite³ structures achieve better compression ratios.

@note When following canonical encoding rules, features `LITE3_ZERO_MEM_DELETED` and `LITE3_ZERO_MEM_EXTRA` are required.
*/
#define LITE3_ZERO_MEM_EXTRA

#if defined(DOXYGEN_ONLY) && !defined(LITE3_ZERO_MEM_EXTRA)
#define LITE3_ZERO_MEM_EXTRA
#endif // DOXYGEN_ONLY

/**
Speeds up iterators, but may cause crashes on non-x86 platforms like ARM.

Enabled by default.

This *may* happen when:
1. reading near unallocated page boundaries
2. untrusted messages contain invalid offsets
*/
#define LITE3_PREFETCHING

#if defined(DOXYGEN_ONLY) && !defined(LITE3_PREFETCHING)
#define LITE3_PREFETCHING
#endif // DOXYGEN_ONLY

/**
Print library-specific error messages to `stdout`.

Disabled by default.

It is recommended to enable this during development.
*/
// #define LITE3_ERROR_MESSAGES

#if defined(DOXYGEN_ONLY) && !defined(LITE3_ERROR_MESSAGES)
#define LITE3_ERROR_MESSAGES
#endif // DOXYGEN_ONLY

#ifndef DOXYGEN_IGNORE
#ifdef LITE3_ERROR_MESSAGES
        #include <stdio.h>
        #define LITE3_PRINT_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#else
        #define LITE3_PRINT_ERROR(format, ...) /* nothing */
#endif
#endif // DOXYGEN_IGNORE

/**
Print library-specific debug information to `stdout`.

Disabled by default.

The output mainly consists of print statements for every value that gets inserted.
Enabling this also turns on `LITE3_ZERO_MEM_DELETED` and `LITE3_ZERO_MEM_EXTRA` features to make the binary structure more readable in memory.

Also enables the function `lite3_print(const unsigned char *buf, size_t buflen)` to view the internal structure of Lite³ buffers.
*/
// #define LITE3_DEBUG

#if defined(DOXYGEN_ONLY) && !defined(LITE3_DEBUG)
#define LITE3_DEBUG
#endif // DOXYGEN_ONLY

#ifndef DOXYGEN_IGNORE
#ifdef LITE3_DEBUG
        #include <stdio.h>

        #ifndef LITE3_ZERO_MEM_DELETED
        #define LITE3_ZERO_MEM_DELETED
        #endif

        #ifndef LITE3_ZERO_MEM_EXTRA
        #define LITE3_ZERO_MEM_EXTRA
        #endif
        // In `LITE3_DEBUG` mode, bytes are replaced with underscores (`0x5F`) for better readability.
        #undef LITE3_ZERO_MEM_8
        #undef LITE3_ZERO_MEM_32
        #define LITE3_ZERO_MEM_8  0x5F
        #define LITE3_ZERO_MEM_32 0x5F5F5F5F

        #define LITE3_PRINT_DEBUG(format, ...) printf(format, ##__VA_ARGS__)
        void lite3_print(const unsigned char *buf, size_t buflen); // View the internal structure of a Lite³ buffer
#else
        #define LITE3_PRINT_DEBUG(format, ...) /* nothing */
        static inline void lite3_print(const unsigned char *buf, size_t buflen) { (void)buf; (void)buflen; }
#endif
#endif // DOXYGEN_IGNORE

/**
Maximum Lite³ buffer size

Because of 32-bit indexes used inside the structure, Lite³ can only physically support a size up to `UINT32_MAX`.
For safety reasons or to preserve resources, it may be desirable to set a lower maximum size.
*/
#define LITE3_BUF_SIZE_MAX      UINT32_MAX

/**
B-tree node alignment

This determines the address alignment at which nodes are placed inside a Lite³ buffer.

Always set to 4-byte alignment according to the node struct's largest member variable (uint32_t).
This setting cannot be changed.

@important
The start of a message (the `*buf` pointer) MUST ALWAYS start on an address that is (at least) 4-byte aligned.
On 64-bit machines, libc malloc() guarantees 16-byte alignment for allocations >= 16 bytes.

@anchor lite3_node_alignment
*/
#define LITE3_NODE_ALIGNMENT            4

#ifndef DOXYGEN_IGNORE
#define LITE3_NODE_ALIGNMENT_MASK       ((uintptr_t)(LITE3_NODE_ALIGNMENT - 1))
#endif // DOXYGEN_IGNORE

/**
B-tree node size setting

Set to 96 bytes (1.5 cache lines) by default. For the vast majority of applications, this setting should never need changing.

@note
Changing this setting also requires changing other settings. See `struct node` inside `lite3.c` for more info.

@important
Do not change this setting unless performance profiling shows real improvements and you know what you are doing.
*/
// #define LITE3_NODE_SIZE              48      // key_count: 0-3       LITE3_NODE_SIZE: 48 (0.75 cache lines)
#define LITE3_NODE_SIZE              96      // key_count: 0-7       LITE3_NODE_SIZE: 96 (1.5 cache lines)
// #define LITE3_NODE_SIZE              192     // key_count: 0-15      LITE3_NODE_SIZE: 192 (3 cache lines)
// #define LITE3_NODE_SIZE              384     // key_count: 0-31      LITE3_NODE_SIZE: 384 (6 cache lines)
// #define LITE3_NODE_SIZE              768     // key_count: 0-63      LITE3_NODE_SIZE: 768 (12 cache lines)

/**
Maximum B-tree height.

Limits the number of node traversals during a lookup.
Can only be changed together with `LITE3_NODE_SIZE`.

@note
Changing this setting also requires changing other settings. See `struct node` inside `lite3.c` for more info.
*/
// #define LITE3_TREE_HEIGHT_MAX           14      // key_count: 0-3       LITE3_NODE_SIZE: 48 (0.75 cache lines)
#define LITE3_TREE_HEIGHT_MAX           9      // key_count: 0-7       LITE3_NODE_SIZE: 96 (1.5 cache lines)
// #define LITE3_TREE_HEIGHT_MAX           7       // key_count: 0-15      LITE3_NODE_SIZE: 192 (3 cache lines)
// #define LITE3_TREE_HEIGHT_MAX           5       // key_count: 0-31      LITE3_NODE_SIZE: 384 (6 cache lines)
// #define LITE3_TREE_HEIGHT_MAX           4       // key_count: 0-63      LITE3_NODE_SIZE: 768 (12 cache lines)

/**
Offset of the `size_kc` field inside `struct node`.

Can only be changed together with `LITE3_NODE_SIZE`.

@note
Changing this setting also requires changing other settings. See `struct node` inside `lite3.c` for more info.
*/
// #define LITE3_NODE_SIZE_KC_OFFSET       16   // key_count: 0-3       LITE3_NODE_SIZE: 48 (0.75 cache lines)
#define LITE3_NODE_SIZE_KC_OFFSET       32   // key_count: 0-7       LITE3_NODE_SIZE: 96 (1.5 cache lines)
// #define LITE3_NODE_SIZE_KC_OFFSET       64   // key_count: 0-15      LITE3_NODE_SIZE: 192 (3 cache lines)
// #define LITE3_NODE_SIZE_KC_OFFSET       128  // key_count: 0-31      LITE3_NODE_SIZE: 384 (6 cache lines)
// #define LITE3_NODE_SIZE_KC_OFFSET       256  // key_count: 0-63      LITE3_NODE_SIZE: 768 (12 cache lines)

#ifndef DOXYGEN_IGNORE
#define LITE3_NODE_SIZE_SHIFT 6
#define LITE3_NODE_SIZE_MASK ((u32)~((1 << 6) - 1)) // 26 MSB

#define LITE3_DJB2_HASH_SEED ((uint32_t)5381)
#endif // DOXYGEN_IGNORE

/**
Enable hash probing to tolerate 32-bit hash collisions.

Hash probing configuration (quadratic open addressing for 32-bit hashes: h_i = h_0 + i^2)

Limit attempts with `LITE3_HASH_PROBE_MAX` (defaults to 128). Probing cannot be disabled.
*/
#ifndef LITE3_HASH_PROBE_MAX
#define LITE3_HASH_PROBE_MAX 128U
#endif

#if LITE3_HASH_PROBE_MAX < 2
    #error "LITE3_HASH_PROBE_MAX must be >= 2"
#endif

#define LITE3_VERIFY_KEY_OK 0
#define LITE3_VERIFY_KEY_HASH_COLLISION 1

/**
Macro to calculate DJB2 key hashes at compile-time

Lite³ compares hash digest of keys instead of direct string comparisons.
The hashes are stored inside the nodes of the B-tree, and the algorithm will traverse them to find a given key.

Calculating such key hashes adds runtime cost and is inside the critical path for tree traversal.
Fortunately, for string literals we can calculate them at compile time and eliminate the runtime cost completely.
This technique makes the API somewhat macro-heavy, but the savings are significant enough to justify it.

One downside is that this can noticably increase compile times due to pressure on the preprocessor.
Therefore this feature can be disabled to speed up build times during development.
*/
#define LITE3_KEY_HASH_COMPILE_TIME

#ifndef DOXYGEN_IGNORE
#ifdef LITE3_KEY_HASH_COMPILE_TIME
#define LITE3_MOD 4294967296ULL

#define LITE3_P0 33ULL
#define LITE3_P1 ((LITE3_P0 * LITE3_P0) % LITE3_MOD)
#define LITE3_P2 ((LITE3_P1 * LITE3_P1) % LITE3_MOD)
#define LITE3_P3 ((LITE3_P2 * LITE3_P2) % LITE3_MOD)
#define LITE3_P4 ((LITE3_P3 * LITE3_P3) % LITE3_MOD)
#define LITE3_P5 ((LITE3_P4 * LITE3_P4) % LITE3_MOD)

#define LITE3_BIT0(k) (((k) >> 0) & 1ULL)
#define LITE3_BIT1(k) (((k) >> 1) & 1ULL)
#define LITE3_BIT2(k) (((k) >> 2) & 1ULL)
#define LITE3_BIT3(k) (((k) >> 3) & 1ULL)
#define LITE3_BIT4(k) (((k) >> 4) & 1ULL)
#define LITE3_BIT5(k) (((k) >> 5) & 1ULL)

#define LITE3_POW33(k) \
        ((((((1ULL \
        * (LITE3_BIT0(k) ? LITE3_P0 : 1ULL) \
        ) % LITE3_MOD \
        * (LITE3_BIT1(k) ? LITE3_P1 : 1ULL) \
        ) % LITE3_MOD \
        * (LITE3_BIT2(k) ? LITE3_P2 : 1ULL) \
        ) % LITE3_MOD \
        * (LITE3_BIT3(k) ? LITE3_P3 : 1ULL) \
        ) % LITE3_MOD \
        * (LITE3_BIT4(k) ? LITE3_P4 : 1ULL) \
        ) % LITE3_MOD \
        * (LITE3_BIT5(k) ? LITE3_P5 : 1ULL) \
        ) % LITE3_MOD

#define LITE3_SUM_1(s, i, p) \
        ((unsigned long long)(unsigned char)(s)[(i)] * LITE3_POW33((p)))

#define LITE3_SUM_2(s, i, p) \
        (LITE3_SUM_1(s, (i), (p)) + LITE3_SUM_1(s, (i) + 1, (p) - 1))

#define LITE3_SUM_4(s, i, p) \
        (LITE3_SUM_2(s, (i), (p)) + LITE3_SUM_2(s, (i) + 2, (p) - 2))

#define LITE3_SUM_8(s, i, p) \
        (LITE3_SUM_4(s, (i), (p)) + LITE3_SUM_4(s, (i) + 4, (p) - 4))

#define LITE3_SUM_16(s, i, p) \
        (LITE3_SUM_8(s, (i), (p)) + LITE3_SUM_8(s, (i) + 8, (p) - 8))

#define LITE3_SUM_32(s, i, p) \
        (LITE3_SUM_16(s, (i), (p)) + LITE3_SUM_16(s, (i) + 16, (p) - 16))

#define LITE3_OFFSET_BIT0(len) \
        (((len & 2ULL) ? 2ULL : 0ULL) + \
        ((len & 4ULL) ? 4ULL : 0ULL) + \
        ((len & 8ULL) ? 8ULL : 0ULL) + \
        ((len & 16ULL) ? 16ULL : 0ULL) + \
        ((len & 32ULL) ? 32ULL : 0ULL))

#define LITE3_OFFSET_BIT1(len) \
        (((len & 4ULL) ? 4ULL : 0ULL) + \
        ((len & 8ULL) ? 8ULL : 0ULL) + \
        ((len & 16ULL) ? 16ULL : 0ULL) + \
        ((len & 32ULL) ? 32ULL : 0ULL))

#define LITE3_OFFSET_BIT2(len) \
        (((len & 8ULL) ? 8ULL : 0ULL) + \
        ((len & 16ULL) ? 16ULL : 0ULL) + \
        ((len & 32ULL) ? 32ULL : 0ULL))

#define LITE3_OFFSET_BIT3(len) \
        (((len & 16ULL) ? 16ULL : 0ULL) + \
        ((len & 32ULL) ? 32ULL : 0ULL))

#define LITE3_OFFSET_BIT4(len) \
        (((len & 32ULL) ? 32ULL : 0ULL))

#define LITE3_OFFSET_BIT5(len) \
        (0ULL)

#define LITE3_POLY_HASH(s, len) \
        ((len) & 1ULL ? LITE3_SUM_1(s, LITE3_OFFSET_BIT0((len)), (len) - 1 - LITE3_OFFSET_BIT0((len))) : 0ULL) + \
        ((len) & 2ULL ? LITE3_SUM_2(s, LITE3_OFFSET_BIT1((len)), (len) - 1 - LITE3_OFFSET_BIT1((len))) : 0ULL) + \
        ((len) & 4ULL ? LITE3_SUM_4(s, LITE3_OFFSET_BIT2((len)), (len) - 1 - LITE3_OFFSET_BIT2((len))) : 0ULL) + \
        ((len) & 8ULL ? LITE3_SUM_8(s, LITE3_OFFSET_BIT3((len)), (len) - 1 - LITE3_OFFSET_BIT3((len))) : 0ULL) + \
        ((len) & 16ULL ? LITE3_SUM_16(s, LITE3_OFFSET_BIT4((len)), (len) - 1 - LITE3_OFFSET_BIT4((len))) : 0ULL) + \
        ((len) & 32ULL ? LITE3_SUM_32(s, LITE3_OFFSET_BIT5((len)), (len) - 1 - LITE3_OFFSET_BIT5((len))) : 0ULL)

#define LITE3_STRLEN(s) (sizeof(s) - 1)

typedef struct {
        uint32_t hash;
        uint32_t size;
} lite3_key_data;

static inline lite3_key_data lite3_get_key_data(const char *key) {
        lite3_key_data key_data;
        const char *key_cursor = key;
        key_data.hash = LITE3_DJB2_HASH_SEED;
        while (*key_cursor)
                key_data.hash = ((key_data.hash << 5) + key_data.hash) + (uint8_t)(*key_cursor++);
        key_data.size = (uint32_t)(key_cursor - key) + 1;
        return key_data;
}

#define LITE3_KEY_DATA(s) ( \
        __builtin_constant_p(s) ? \
                ((LITE3_STRLEN(s) < 64) ? \
                        (lite3_key_data){ \
                                .hash = (uint32_t)((LITE3_POLY_HASH(s, LITE3_STRLEN(s)) + LITE3_DJB2_HASH_SEED * LITE3_POW33(LITE3_STRLEN(s))) % LITE3_MOD), \
                                .size = (sizeof(s)), \
                        } \
                : lite3_get_key_data(s)) \
        : lite3_get_key_data(__lite3_key__) \
)
#else
#define LITE3_KEY_DATA(s) lite3_get_key_data(__lite3_key__)
#endif
#endif // DOXYGEN_IGNORE
/// @} lite3_config



/**
Custom types of the Lite³ library.

@defgroup lite3_types Library Custom Types
@{
*/
/**
`enum` containing all Lite³ types

Lite³ prefixes all values with a 1-byte type tag, similar to tagged unions.
*/
enum lite3_type {
        LITE3_TYPE_NULL,        ///< maps to 'null' type in JSON
        LITE3_TYPE_BOOL,        ///< maps to 'boolean' type in JSON; underlying datatype: `bool`
        LITE3_TYPE_I64,         ///< maps to 'number' type in JSON; underlying datatype: `int64_t`
        LITE3_TYPE_F64,         ///< maps to 'number' type in JSON; underlying datatype: `double`
        LITE3_TYPE_BYTES,       ///< coverted to base64 string in JSON
        LITE3_TYPE_STRING,      ///< maps to 'string' type in JSON
        LITE3_TYPE_OBJECT,      ///< maps to 'object' type in JSON
        LITE3_TYPE_ARRAY,       ///< maps to 'array' type in JSON
        LITE3_TYPE_INVALID,     ///< any type value equal or greater than this is considered invalid
        LITE3_TYPE_COUNT,       ///< not an actual type, only used for counting
};

/**
Struct representing a value inside a Lite³ buffer

Lite³ prefixes all values with a 1-byte type tag, similar to tagged unions.
To discover types inside a message, compare against the `lite3_val.type` field.

See @ref lite3_val_fns.
*/
typedef struct {
        uint8_t type;
        uint8_t val[];
} lite3_val;

#ifndef DOXYGEN_IGNORE
#define LITE3_VAL_SIZE sizeof(lite3_val)
#endif // DOXYGEN_IGNORE
static_assert(LITE3_VAL_SIZE <= sizeof(size_t), "LITE3_VAL_SIZE must be <= sizeof(size_t)");

#ifndef DOXYGEN_IGNORE
static const size_t lite3_type_sizes[] = {
        0,                                      // LITE3_TYPE_NULL
        1,                                      // LITE3_TYPE_BOOL
        8,                                      // LITE3_TYPE_I64
        8,                                      // LITE3_TYPE_F64
        4,                                      // LITE3_TYPE_BYTES     (this value must be <= sizeof(size_t))
        4,                                      // LITE3_TYPE_STRING    (this value must be <= sizeof(size_t))
        LITE3_NODE_SIZE - LITE3_VAL_SIZE,       // LITE3_TYPE_OBJECT    (`type` field is contained inside node->gen_type)
        LITE3_NODE_SIZE - LITE3_VAL_SIZE,       // LITE3_TYPE_ARRAY     (`type` field is contained inside node->gen_type)
        0,                                      // LITE3_TYPE_INVALID
};
#endif // DOXYGEN_IGNORE
static_assert((sizeof(lite3_type_sizes) / sizeof(size_t)) == LITE3_TYPE_COUNT, "lite3_type_sizes[] element count != LITE3_TYPE_COUNT");
static_assert(4 <= sizeof(size_t), "lite3_type_sizes[LITE3_TYPE_BYTES] and lite3_type_sizes[LITE3_TYPE_STRING] must fit inside size_t");

/**
Struct holding a reference to a bytes value inside a Lite³ buffer

Returned by `lite3_get_bytes()` and `lite3_ctx_get_bytes()`.

Lite³ buffers store an internal 'generation count'. Any mutations to the buffer will increment the count.

This struct contains a `gen` field equal to the generation of the Lite³ buffer when this struct was returned.
When the `gen` field of the reference does not match the buffer's generation count, this means the reference is invalid.
We can mitigate 'dangling pointer' scenarios by wrapping the reference with a macro: `LITE3_BYTES(buf, lite3_bytes)`.
This macro checks if the buffer's generation count still matches the reference. If so, we return a direct pointer. Otherwise, we receive a NULL pointer. 

@important
Never dereference `.ptr` directly! Always use `LITE3_BYTES(buf, lite3_bytes)` macro wrapper for safe access!
*/
typedef struct {
        uint32_t gen;                   ///< generation of the Lite³ buffer when this struct was returned
        uint32_t len;                   ///< byte array length (bytes)
        const unsigned char *ptr;       ///< byte array pointer to bytes inside Lite³ buffer
} lite3_bytes;

#ifndef DOXYGEN_IGNORE
#define LITE3_BYTES_LEN_SIZE sizeof(((lite3_bytes *)0)->len)
#endif // DOXYGEN_IGNORE
static_assert(LITE3_BYTES_LEN_SIZE <= sizeof(size_t), "lite3_val_bytes() expects LITE3_BYTES_LEN_SIZE to be <= sizeof(size_t)");

/**
Struct holding a reference to a string inside a Lite³ buffer

Returned by `lite3_get_str()` and `lite3_ctx_get_str()`.

Lite³ buffers store an internal 'generation count'. Any mutations to the buffer will increment the count.

This struct contains a `gen` field equal to the generation of the Lite³ buffer when this struct was returned.
When the `gen` field of the reference does not match the buffer's generation count, this means the reference is invalid.
We can mitigate 'dangling pointer' scenarios by wrapping the reference with a macro: `LITE3_STR(buf, lite3_str)`.
This macro checks if the buffer's generation count still matches the reference. If so, we return a direct pointer. Otherwise, we receive a NULL pointer. 

@warning
The `len` struct member is exclusive of the NULL-terminator.

@important
Never dereference `.ptr` directly! Always use `LITE3_STR(buf, lite3_str)` macro wrapper for safe access!
*/
typedef struct {
        uint32_t gen;           ///< generation of the Lite³ buffer when this struct was returned
        uint32_t len;           ///< char array length (characters, exclusive of NULL-terminator)
        const char *ptr;        ///< char array pointer to string inside Lite³ buffer
} lite3_str;

#ifndef DOXYGEN_IGNORE
#define LITE3_STR_LEN_SIZE sizeof(((lite3_str *)0)->len)
#endif // DOXYGEN_IGNORE
static_assert(LITE3_STR_LEN_SIZE <= sizeof(size_t), "lite3_val_str() expects LITE3_STR_LEN_SIZE to be <= sizeof(size_t)");

/**
Generational pointer / safe access wrapper

Every Lite³ buffer stores a generation count which is incremented on every mutation.
lite3_bytes accessed through the LITE3_BYTES() macro will return a direct pointer if `lite3_bytes.gen` matches the generation count of the buffer.
Otherwise, it returns NULL.

When a Lite³ structure is modified via `set()` or `delete()`, pointed to data could be moved or deleted;
therefore it is no longer safe to dereference previously obtained pointers. This macro prevents dangerous situations with dangling pointers.

Example usage:
```C
lite3_str data;
if (lite3_get_bytes(buf, buflen, 0, "data_field", &data) < 0)
        return 1;

memcpy(dest_ptr, data.ptr, data.len); // ❌ Unsafe! Always use wrapper macro!

memcpy(dest_ptr, LITE3_BYTES(buf, data), data.len); // ✅ Safe dereference
// For context API: LITE3_BYTES(ctx->buf, data)
```

@param[in] buf the Lite³ buffer to which lite3_bytes is pointing
@param[in] val the lite3_bytes struct (passed by value, not by reference)
*/
#define LITE3_BYTES(buf, val) (const unsigned char *)_lite3_ptr_suppress_nonnull_warning( \
        (uint32_t)(val).gen == *(uint32_t *)(buf) ? (val).ptr : NULL \
)

/**
Generational pointer / safe access wrapper

Every Lite³ buffer stores a generation count which is incremented on every mutation.
lite3_str accessed through the LITE3_STR() macro will return a direct pointer if `lite3_str.gen` matches the generation count of the buffer.
Otherwise, it returns NULL.

When a Lite³ structure is modified via `set()` or `delete()`, pointed to data could be moved or deleted;
therefore it is no longer safe to dereference previously obtained pointers. This macro prevents dangerous situations with dangling pointers.

Example usage:
```C
lite3_str str;
if (lite3_get_str(buf, buflen, 0, "str_field", &str) < 0)
        return 1;

memcpy(dest_ptr, str.ptr, str.len); // ❌ Unsafe! Always use wrapper macro!

memcpy(dest_ptr, LITE3_STR(buf, str), str.len); // ✅ Safe dereference
// For context API: LITE3_STR(ctx->buf, str)
```

@param[in] buf the Lite³ buffer to which lite3_str is pointing
@param[in] val the lite3_str struct (passed by value, not by reference)
*/
#define LITE3_STR(buf, val) (const char *)_lite3_ptr_suppress_nonnull_warning( \
        (uint32_t)(val).gen == *(uint32_t *)(buf) ? (val).ptr : NULL \
)

#ifndef DOXYGEN_IGNORE
static inline __attribute__((always_inline)) const void *_lite3_ptr_suppress_nonnull_warning(const void *p) { return p; }
#endif // DOXYGEN_IGNORE
/// @} lite3_types



/**
Object / array initialization

The JSON standard requires that the root-level type always be an 'object' or 'array'. This also applies to Lite³.

Before data can be inserted into an empty buffer, it must first be initialized as object or array.

@defgroup lite3_init Object / Array Initialization
@ingroup lite3_buffer_api
@{
*/
/**
Initialize a Lite³ buffer as an object

The number of bytes written to `*buf` never exceeds `bufsz`.
The available buffer space must be at least `LITE3_NODE_SIZE`.

@return 0 on success
@return < 0 on error

@note
This function can also be used to reset an existing Lite³ message; the root node is simply replaced with an empty object.
*/
int lite3_init_obj(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict out_buflen,  ///< [out] buffer used length
        size_t bufsz                    ///< [in] buffer max size
);

/**
Initialize a Lite³ buffer as an array

The number of bytes written to `*buf` never exceeds `bufsz`.
The available buffer space must be at least `LITE3_NODE_SIZE`.

@return 0 on success
@return < 0 on error

@note
This function can also be used to reset an existing Lite³ message; the root node is simply replaced with an empty array.
*/
int lite3_init_arr(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict out_buflen,  ///< [out] buffer used length
        size_t bufsz                    ///< [in] buffer max size
);
/// @} lite3_init



/**
Set key-value pair in object

An empty buffer must first be initialized using `lite3_init_obj()` or `lite3_init_arr()` before insertion. See @ref lite3_init.

Set functions read `*inout_buflen` to know the currently used portion of the buffer. After modifications, the new length will be written back.
The caller must provide sufficient buffer space via `bufsz` or the call will fail and set `errno` to `ENOBUFS`. Retrying is up to the caller.

The number of bytes written to `*buf` never exceeds `bufsz`.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@note
Inserting a value with an existing key will override the current value.

@warning
1. Insertions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase `*inout_buflen`.
3. Overriding any value with an existing key can still grow the buffer and increase `*inout_buflen`.
4. Overriding a **variable-length value (string/bytes)** will require extra buffer space if the new value is larger than the old.
The overridden space is never recovered, causing buffer size to grow indefinitely.

@defgroup lite3_obj_set Object Set
@ingroup lite3_buffer_api
@{
*/
#ifndef DOXYGEN_IGNORE
static inline int _lite3_verify_set(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz)
{
        (void)buf;
        if (LITE3_UNLIKELY(bufsz > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(*inout_buflen > bufsz)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: inout_buflen > bufsz\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(LITE3_NODE_SIZE > *inout_buflen || ofs > *inout_buflen - LITE3_NODE_SIZE)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: START OFFSET OUT OF BOUNDS\n");
                errno = EINVAL;
                return -1;
        }
        return 0;
}

static inline int _lite3_verify_obj_set(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key)
{
        int ret;
        if ((ret = _lite3_verify_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        if (LITE3_UNLIKELY(*(buf + ofs) != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING OBJECT TYPE\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(!key)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING NON-NULL KEY\n");
                errno = EINVAL;
                return -1;
        }
        return ret;
}

static inline int _lite3_verify_arr_set(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz)
{
        int ret;
        if ((ret = _lite3_verify_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        if (LITE3_UNLIKELY(*(buf + ofs) != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY TYPE\n");
                errno = EINVAL;
                return -1;
        }
        return ret;
}

// Private function
int lite3_set_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, size_t val_len, lite3_val **out);
#endif // DOXYGEN_IGNORE

/**
Set null in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key

@return 0 on success
@return < 0 on error
*/
#define lite3_set_null(buf, inout_buflen, ofs, bufsz, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_null_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_null_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set boolean in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      value (`bool`) boolean value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_set_bool(buf, inout_buflen, ofs, bufsz, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_bool_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_bool_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, bool value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BOOL;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_BOOL]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set integer in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      value (`uint64_t`) integer value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_set_i64(buf, inout_buflen, ofs, bufsz, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_i64_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_i64_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, int64_t value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_I64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_I64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set floating point in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      value (`double`) floating point value to set

@return 0 on success
@return < 0 on error
*/
#define lite3_set_f64(buf, inout_buflen, ofs, bufsz, key, value) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_f64_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), value); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_f64_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, double value)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_F64;
        memcpy(val->val, &value, lite3_type_sizes[LITE3_TYPE_F64]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set bytes in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      bytes (`const unsigned char *`) bytes pointer
@param[in]      bytes_len (`size_t`) bytes amount

@return 0 on success
@return < 0 on error
*/
#define lite3_set_bytes(buf, inout_buflen, ofs, bufsz, key, bytes, bytes_len) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_bytes_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), bytes, bytes_len); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_bytes_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, const unsigned char *__restrict bytes, size_t bytes_len)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_BYTES;
        memcpy(val->val, &bytes_len, lite3_type_sizes[LITE3_TYPE_BYTES]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_BYTES], bytes, bytes_len);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set string in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      str (`const char *`) string pointer

@return 0 on success
@return < 0 on error

@note
This function must call `strlen()` to learn the size of the string.
If you know the length beforehand, it is more efficient to call `lite3_set_str_n()`.
*/
#define lite3_set_str(buf, inout_buflen, ofs, bufsz, key, str) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_str_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), str); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_str_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, const char *__restrict str)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_size);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set string in object by length

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[in]      str (`const char *`) string pointer
@param[in]      str_len (`size_t`) string length, exclusive of NULL-terminator.

@return 0 on success
@return < 0 on error

@warning
`str_len` is exclusive of the NULL-terminator.
*/
#define lite3_set_str_n(buf, inout_buflen, ofs, bufsz, key, str, str_len) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_set_str_n_impl(buf, inout_buflen, ofs, bufsz, __lite3_key__, LITE3_KEY_DATA(key), str, str_len); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_str_n_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, const char *__restrict str, size_t str_len)
{
        int ret;
        if ((ret = _lite3_verify_obj_set(buf, inout_buflen, ofs, bufsz, key)) < 0)
                return ret;
        lite3_val *val;
        size_t str_size = str_len + 1;
        if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_STRING;
        memcpy(val->val, &str_size, lite3_type_sizes[LITE3_TYPE_STRING]);
        memcpy(val->val + lite3_type_sizes[LITE3_TYPE_STRING], str, str_len);
        *(val->val + lite3_type_sizes[LITE3_TYPE_STRING] + str_len) = 0x00; // Insert NULL-terminator
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Set object in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[out]     out_ofs (`size_t *`) offset of the newly inserted object (if not needed, pass `NULL`)

@return 0 on success
@return < 0 on error
*/
#define lite3_set_obj(buf, inout_buflen, ofs, bufsz, key, out_ofs) ({ \
        unsigned char *__lite3_buf__ = (buf); \
        size_t *__lite3_inout_buflen__ = (inout_buflen); \
        size_t __lite3_ofs__ = (ofs); \
        size_t __lite3_bufsz__ = (bufsz); \
        const char *__lite3_key__ = (key); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_obj_set( \
                __lite3_buf__, \
                __lite3_inout_buflen__, \
                __lite3_ofs__, \
                __lite3_bufsz__, \
                __lite3_key__)) < 0) \
                return __lite3_ret__; \
        \
        lite3_set_obj_impl( \
                __lite3_buf__, \
                __lite3_inout_buflen__, \
                __lite3_ofs__, \
                __lite3_bufsz__, \
                __lite3_key__, \
                LITE3_KEY_DATA(key), \
                out_ofs); \
})
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_set_obj_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE

/**
Set array in object

@param[in]      buf (`unsigned char *`) buffer pointer
@param[in,out]  inout_buflen (`size_t *`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      bufsz (`size_t`) buffer max size
@param[in]      key (`const char *`) key
@param[out]     out_ofs (`size_t *`) offset of the newly inserted array (if not needed, pass `NULL`)

@return 0 on success
@return < 0 on error
*/
#define lite3_set_arr(buf, inout_buflen, ofs, bufsz, key, out_ofs) ({ \
        unsigned char *__lite3_buf__ = (buf); \
        size_t *__lite3_inout_buflen__ = (inout_buflen); \
        size_t __lite3_ofs__ = (ofs); \
        size_t __lite3_bufsz__ = (bufsz); \
        const char *__lite3_key__ = (key); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_obj_set( \
                __lite3_buf__, \
                __lite3_inout_buflen__, \
                __lite3_ofs__, \
                __lite3_bufsz__, \
                __lite3_key__)) < 0) \
                return __lite3_ret__; \
        \
        lite3_set_arr_impl( \
                __lite3_buf__, \
                __lite3_inout_buflen__, \
                __lite3_ofs__, \
                __lite3_bufsz__, \
                __lite3_key__, \
                LITE3_KEY_DATA(key), \
                out_ofs); \
})
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_set_arr_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE
/// @} lite3_obj_set



/**
Append value to array

An empty buffer must first be initialized using `lite3_init_obj()` or `lite3_init_arr()` before insertion. See @ref lite3_init.

Set functions read `*inout_buflen` to know the currently used portion of the buffer. After modifications, the new length will be written back.
The caller must provide sufficient buffer space via `bufsz` or the call will fail and set `errno` to `ENOBUFS`. Retrying is up to the caller.

The number of bytes written to `*buf` never exceeds `bufsz`.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
1. Append actions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase `*inout_buflen`.

@defgroup lite3_arr_append Array Append
@ingroup lite3_buffer_api
@{
*/
#ifndef DOXYGEN_IGNORE
static inline int _lite3_set_by_index(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t val_len, lite3_val **out)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf,  inout_buflen, ofs, bufsz)) < 0)
                return ret;
        uint32_t size = (*(uint32_t *)(buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        if (LITE3_UNLIKELY(index > size)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
                errno = EINVAL;
                return -1;
        }
        lite3_key_data key_data = {
                .hash = index,
                .size = 0,
        };
        return lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, val_len, out);
}

static inline int _lite3_set_by_append(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, size_t val_len, lite3_val **out)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        uint32_t size = (*(uint32_t *)(buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        lite3_key_data key_data = {
                .hash = size,
                .size = 0,
        };
        return lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, val_len, out);
}
#endif // DOXYGEN_IGNORE

/**
Append null to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_append_null(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz)                   ///< [in] buffer max size
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}

/**
Append boolean to array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_append_bool(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        bool value)                     ///< [in] boolean value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0)
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
static inline int lite3_arr_append_i64(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        int64_t value)                  ///< [in] integer value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0)
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
static inline int lite3_arr_append_f64(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        double value)                   ///< [in] floating point value to append
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0)
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
static inline int lite3_arr_append_bytes(
        unsigned char *buf,                     ///< [in] buffer pointer
        size_t *__restrict inout_buflen,        ///< [in,out] buffer used length
        size_t ofs,                             ///< [in] start offset (0 == root)
        size_t bufsz,                           ///< [in] buffer max size
        const unsigned char *__restrict bytes,  ///< [in] bytes pointer
        size_t bytes_len)                       ///< [in] bytes amount
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0)
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
static inline int lite3_arr_append_str(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        const char *__restrict str)     ///< [in] string pointer
{
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
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
static inline int lite3_arr_append_str_n(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        const char *__restrict str,     ///< [in] string pointer
        size_t str_len)                 ///< [in] string length, exclusive of NULL-terminator.
{
        lite3_val *val;
        size_t str_size = str_len + 1;
        int ret;
        if ((ret = _lite3_set_by_append(buf, inout_buflen, ofs, bufsz, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
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
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_arr_append_obj_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE

static inline int lite3_arr_append_obj(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        size_t *__restrict out_ofs)     ///< [out] offset of the newly appended object (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        return lite3_arr_append_obj_impl(buf, inout_buflen, ofs, bufsz, out_ofs);
}


/**
Append array to array

@return 0 on success
@return < 0 on error
*/
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_arr_append_arr_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE

static inline int lite3_arr_append_arr(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        size_t *__restrict out_ofs)     ///< [out] offset of the newly appended array (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        return lite3_arr_append_arr_impl(buf, inout_buflen, ofs, bufsz, out_ofs);
}

/// @} lite3_arr_append



/**
Set value in array

An empty buffer must first be initialized using `lite3_init_obj()` or `lite3_init_arr()` before insertion. See @ref lite3_init.

Set functions read `*inout_buflen` to know the currently used portion of the buffer. After modifications, the new length will be written back.
The caller must provide sufficient buffer space via `bufsz` or the call will fail and set `errno` to `ENOBUFS`. Retrying is up to the caller.

The number of bytes written to `*buf` never exceeds `bufsz`.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@note
Setting a value at an existing index will override the current value.

@warning
1. Insertions, like any other buffer mutations, are not thread-safe. The caller must manually synchronize access to the buffer.
2. A failed call with return value < 0 can still write to the buffer and increase `*inout_buflen`.
3. Overriding any value with an existing key can still grow the buffer and increase `*inout_buflen`.
4. Overriding a **variable-length value (string/bytes)** will require extra buffer space if the new value is larger than the old.
The overridden space is never recovered, causing buffer size to grow indefinitely.

@defgroup lite3_arr_set Array Set
@ingroup lite3_buffer_api
@{
*/
/**
Set null in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_set_null(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index)                 ///< [in] array index
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_NULL], &val)) < 0)
                return ret;
        val->type = (uint8_t)LITE3_TYPE_NULL;
        return ret;
}

/**
Set boolean in array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_set_bool(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        bool value)                     ///< [in] boolean value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_BOOL], &val)) < 0)
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
static inline int lite3_arr_set_i64(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        int64_t value)                  ///< [in] integer value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_I64], &val)) < 0)
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
static inline int lite3_arr_set_f64(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        double value)                   ///< [in] floating point value to set
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_F64], &val)) < 0)
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
static inline int lite3_arr_set_bytes(
        unsigned char *buf,                     ///< [in] buffer pointer
        size_t *__restrict inout_buflen,        ///< [in,out] buffer used length
        size_t ofs,                             ///< [in] start offset (0 == root)
        size_t bufsz,                           ///< [in] buffer max size
        uint32_t index,                         ///< [in] array index
        const unsigned char *__restrict bytes,  ///< [in] bytes pointer
        size_t bytes_len)                         ///< [in] bytes amount
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_BYTES] + bytes_len, &val)) < 0)
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
static inline int lite3_arr_set_str(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        const char *__restrict str)     ///< [in] string pointer
{
        lite3_val *val;
        size_t str_size = strlen(str) + 1;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
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
static inline int lite3_arr_set_str_n(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        const char *__restrict str,     ///< [in] string pointer
        size_t str_len)                 ///< [in] string length, exclusive of NULL-terminator.
{
        lite3_val *val;
        size_t str_size = str_len + 1;
        int ret;
        if ((ret = _lite3_set_by_index(buf, inout_buflen, ofs, bufsz, index, lite3_type_sizes[LITE3_TYPE_STRING] + str_size, &val)) < 0)
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
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_arr_set_obj_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE

static inline int lite3_arr_set_obj(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] offset of the newly inserted object (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        return lite3_arr_set_obj_impl(buf, inout_buflen, ofs, bufsz, index, out_ofs);
}


/**
Set array in array

@return 0 on success
@return < 0 on error
*/
#ifndef DOXYGEN_IGNORE
// Private function
int lite3_arr_set_arr_impl(unsigned char *buf, size_t *__restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *__restrict out_ofs);
#endif // DOXYGEN_IGNORE

static inline int lite3_arr_set_arr(
        unsigned char *buf,             ///< [in] buffer pointer
        size_t *__restrict inout_buflen,///< [in,out] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t bufsz,                   ///< [in] buffer max size
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] offset of the newly inserted array (if not needed, pass `NULL`)
{
        int ret;
        if ((ret = _lite3_verify_arr_set(buf, inout_buflen, ofs, bufsz)) < 0)
                return ret;
        return lite3_arr_set_arr_impl(buf, inout_buflen, ofs, bufsz, index, out_ofs);
}
/// @} lite3_arr_set



/**
Utility functions

Get functions read `buflen` to know the currently used portion of the buffer.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@warning
Read-only operations are thread-safe. This includes all utility functions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_utility Utility Functions
@ingroup lite3_buffer_api
@{
*/
#ifndef DOXYGEN_IGNORE
static inline int _lite3_verify_get(const unsigned char *buf, size_t buflen, size_t ofs)
{
        (void)buf;
        if (LITE3_UNLIKELY(buflen > LITE3_BUF_SIZE_MAX)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: buflen > LITE3_BUF_SIZE_MAX\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(LITE3_NODE_SIZE > buflen || ofs > buflen - LITE3_NODE_SIZE)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: START OFFSET OUT OF BOUNDS\n");
                errno = EINVAL;
                return -1;
        }
        return 0;
}

static inline int _lite3_verify_obj_get(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key)
{
        if (_lite3_verify_get(buf, buflen, ofs) < 0)
                return -1;
        if (LITE3_UNLIKELY(*(buf + ofs) != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING OBJECT TYPE\n");
                errno = EINVAL;
                return -1;
        }
        if (LITE3_UNLIKELY(!key)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING NON-NULL KEY\n");
                errno = EINVAL;
                return -1;
        }
        return 0;
}

static inline int _lite3_verify_arr_get(const unsigned char *buf, size_t buflen, size_t ofs)
{
        if (_lite3_verify_get(buf, buflen, ofs) < 0)
                return -1;
        if (LITE3_UNLIKELY(*(buf + ofs) != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY TYPE\n");
                errno = EINVAL;
                return -1;
        }
        return 0;
}

// Private function
int lite3_get_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, lite3_val **out);

static inline int _lite3_get_by_index(const unsigned char *buf, size_t buflen, size_t ofs, uint32_t index, lite3_val **out)
{
        int ret;
        if ((ret = _lite3_verify_arr_get(buf, buflen, ofs)) < 0)
                return ret;
        uint32_t size = (*(uint32_t *)(buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        if (LITE3_UNLIKELY(index >= size)) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
                errno = EINVAL;
                return -1;
        }
        lite3_key_data key_data = {
                .hash = index,
                .size = 0,
        };
        return lite3_get_impl(buf, buflen, ofs, NULL, key_data, out);
}
#endif // DOXYGEN_IGNORE

/**
Get the root type of a Lite³ buffer

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length

@return `lite3_type` on success (`LITE3_TYPE_OBJECT` or `LITE3_TYPE_ARRAY`)
@return `LITE3_TYPE_INVALID` on error (empty or invalid buffer)
*/
static inline enum lite3_type lite3_get_root_type(const unsigned char *buf, size_t buflen)
{
        if (_lite3_verify_get(buf, buflen, 0) < 0)
                return LITE3_TYPE_INVALID;
        return (enum lite3_type)(*buf);
}

/**
Find value by key and return value type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `lite3_type` on success
@return `LITE3_TYPE_INVALID` on error (key cannot be found)
*/
#define lite3_get_type(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_type_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline enum lite3_type _lite3_get_type_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return LITE3_TYPE_INVALID;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return LITE3_TYPE_INVALID;
        return (enum lite3_type)val->type;
}
#endif // DOXYGEN_IGNORE

/**
Find value by index and return value type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      index (`uint32_t`) array index

@return `lite3_type` on success
@return `LITE3_TYPE_INVALID` on error (index out of bounds)
*/
static inline enum lite3_type lite3_arr_get_type(const unsigned char *buf, size_t buflen, size_t ofs, uint32_t index)
{
        if (_lite3_verify_arr_get(buf, buflen, ofs) < 0)
                return LITE3_TYPE_INVALID;
        lite3_val *val;
        if (_lite3_get_by_index(buf, buflen, ofs, index, &val) < 0)
                return LITE3_TYPE_INVALID;
        return (enum lite3_type)val->type;
}

/**
Find value by key and write back type size

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size *`) type size

@return 0 on success
@return < 0 on error

@note
For variable sized types like `LITE3_TYPE_BYTES` or `LITE3_TYPE_STRING`, the number of bytes (including NULL-terminator for string) are written back.
*/
#define lite3_get_type_size(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_type_size_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_type_size_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
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

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` on success
@return `false` on failure
*/
#define lite3_exists(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_exists_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_exists_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
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
static inline int lite3_count(
        unsigned char *buf,     ///< [in] buffer pointer
        size_t buflen,          ///< [in] buffer used length
        size_t ofs,             ///< [in] start offset (0 == root)
        uint32_t *out)          ///< [out] number of object entries or array elements
{
        int ret;
        if ((ret = _lite3_verify_get(buf, buflen, ofs)) < 0)
                return ret;
        enum lite3_type type = (enum lite3_type)(*(buf + ofs));
        if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
                LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
                errno = EINVAL;
                return -1;
        }
        *out = (*(uint32_t *)(buf + ofs + LITE3_NODE_SIZE_KC_OFFSET)) >> LITE3_NODE_SIZE_SHIFT;
        return ret;
}

/**
Find value by key and test for null type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_null(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_null_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_null_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_NULL;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for bool type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_bool(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_bool_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_bool_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_BOOL;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for integer type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_i64(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_i64_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_i64_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_I64;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for floating point type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_f64(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_f64_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_f64_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_F64;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for bytes type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_bytes(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_bytes_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_bytes_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_BYTES;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for string type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_str(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_str_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_str_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_STRING;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for object type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_obj(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_obj_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_obj_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_OBJECT;
}
#endif // DOXYGEN_IGNORE

/**
Find value by key and test for array type

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key

@return `true` if the value matches the type
@return `false` if the type does not match or the key cannot be found
*/
#define lite3_is_arr(buf, buflen, ofs, key) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_is_arr_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key)); \
})
#ifndef DOXYGEN_IGNORE
static inline bool _lite3_is_arr_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data)
{
        if (_lite3_verify_obj_get(buf, buflen, ofs, key) < 0)
                return false;
        lite3_val *val;
        if (lite3_get_impl(buf, buflen, ofs, key, key_data, &val) < 0)
                return false;
        return val->type == LITE3_TYPE_ARRAY;
}
#endif // DOXYGEN_IGNORE
/// @} lite3_utility



/**
Get value from object by key

Get functions read `buflen` to know the currently used portion of the buffer.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
Read-only operations are thread-safe. This includes all `lite3_get_xxx()` functions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_get Object Get
@ingroup lite3_buffer_api
@{
*/
/**
Get value from object

Unlike other `lite3_get_xxx()` functions, this function does not get a specific type.
Instead, it produces a generic `lite3_val` pointer, which points to a value inside the Lite³ buffer.
This can be useful in cases where you don't know the exact type of a value beforehand. See @ref lite3_val_fns.

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_val *`) opaque value pointer

@return 0 on success
@return < 0 on error
*/
#define lite3_get(buf, buflen, ofs, key, out) ({ \
        const unsigned char *__lite3_buf__ = (buf); \
        size_t __lite3_buflen__ = (buflen); \
        size_t __lite3_ofs__ = (ofs); \
        int __lite3_ret__; \
        if ((__lite3_ret__ = _lite3_verify_get(__lite3_buf__, __lite3_buflen__, __lite3_ofs__) < 0)) \
                return __lite3_ret__; \
        const char *__lite3_key__ = (key); \
        lite3_get_impl(__lite3_buf__, __lite3_buflen__, __lite3_ofs__, __lite3_key__, LITE3_KEY_DATA(key), out); \
})

/**
Get boolean value by key

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`bool *`) boolean value

@return 0 on success
@return < 0 on error
*/
#define lite3_get_bool(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_bool_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_bool_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, bool *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
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

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`int64_t *`) integer value

@return 0 on success
@return < 0 on error
*/
#define lite3_get_i64(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_i64_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_i64_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, int64_t *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
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

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`double *`) floating point value

@return 0 on success
@return < 0 on error
*/
#define lite3_get_f64(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_f64_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_f64_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, double *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
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

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_bytes *`) bytes value

@return 0 on success
@return < 0 on error
*/
#define lite3_get_bytes(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_bytes_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_bytes_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, lite3_bytes *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BYTES)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BYTES\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_bytes){
                .gen = *(uint32_t *)buf,
                .len = 0,
                .ptr = val->val + lite3_type_sizes[LITE3_TYPE_BYTES]
        };
        memcpy(&out->len, val->val, lite3_type_sizes[LITE3_TYPE_BYTES]);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get string value by key

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`lite3_str *`) string value

@return 0 on success
@return < 0 on error
*/
#define lite3_get_str(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_str_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_str_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, lite3_str *out)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_STRING)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_STRING\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_str){
                .gen = *(uint32_t *)buf,
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

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size_t *`) object offset

@return 0 on success
@return < 0 on error
*/
#define lite3_get_obj(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_obj_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_obj_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_OBJECT\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - buf);
        return ret;
}
#endif // DOXYGEN_IGNORE

/**
Get array by key

@param[in]      buf (`const unsigned char *`) buffer pointer
@param[in]      buflen (`size_t`) buffer used length
@param[in]      ofs (`size_t`) start offset (0 == root)
@param[in]      key (`const char *`) key
@param[out]     out (`size_t *`) array offset

@return 0 on success
@return < 0 on error
*/
#define lite3_get_arr(buf, buflen, ofs, key, out) ({ \
        const char *__lite3_key__ = (key); \
        _lite3_get_arr_impl(buf, buflen, ofs, __lite3_key__, LITE3_KEY_DATA(key), out); \
})
#ifndef DOXYGEN_IGNORE
static inline int _lite3_get_arr_impl(const unsigned char *buf, size_t buflen, size_t ofs, const char *__restrict key, lite3_key_data key_data, size_t *__restrict out_ofs)
{
        int ret;
        if ((ret = _lite3_verify_obj_get(buf, buflen, ofs, key)) < 0)
                return ret;
        lite3_val *val;
        if ((ret = lite3_get_impl(buf, buflen, ofs, key, key_data, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_ARRAY\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - buf);
        return ret;
}
#endif // DOXYGEN_IGNORE
/// @} lite3_get



/**
Get value from array by index

Get functions read `buflen` to know the currently used portion of the buffer.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@par Returns
- Returns 0 on success
- Returns < 0 on error

@warning
Read-only operations are thread-safe. This includes all `lite3_arr_get_xxx()` functions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_arr_get Array Get
@ingroup lite3_buffer_api
@{
*/
/**
Get boolean value by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_get_bool(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        bool *out)                      ///< [out] boolean value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
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
static inline int lite3_arr_get_i64(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        int64_t *out)                   ///< [out] integer value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
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
static inline int lite3_arr_get_f64(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        double *out)                    ///< [out] floating point value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
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
static inline int lite3_arr_get_bytes(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        lite3_bytes *out)               ///< [out] bytes value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_BYTES)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_BYTES\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_bytes){
                .gen = *(uint32_t *)buf,
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
static inline int lite3_arr_get_str(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        lite3_str *out)                 ///< [out] string value
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_STRING)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_STRING\n");
                errno = EINVAL;
                return -1;
        }
        *out = (lite3_str){
                .gen = *(uint32_t *)buf,
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
static inline int lite3_arr_get_obj(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] object offset
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_OBJECT)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_OBJECT\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - buf);
        return ret;
}

/**
Get array by index

@return 0 on success
@return < 0 on error
*/
static inline int lite3_arr_get_arr(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        uint32_t index,                 ///< [in] array index
        size_t *__restrict out_ofs)     ///< [out] array offset
{
        lite3_val *val;
        int ret;
        if ((ret = _lite3_get_by_index(buf, buflen, ofs, index, &val)) < 0)
                return ret;
        if (LITE3_UNLIKELY(val->type != LITE3_TYPE_ARRAY)) {
                LITE3_PRINT_ERROR("VALUE TYPE != LITE3_TYPE_ARRAY\n");
                errno = EINVAL;
                return -1;
        }
        *out_ofs = (size_t)((uint8_t *)val - buf);
        return ret;
}
/// @} lite3_arr_get



/**
Create and use iterators for objects/arrays

Iter functions read `buflen` to know the currently used portion of the buffer.

The `ofs` (offset) field is used to target an object or array inside the Lite³ buffer. To target the root-level object/array, use `ofs == 0`.

@warning
Read-only operations are thread-safe. This includes all iterator functions. Mixing reads and writes however is not thread-safe.

@defgroup lite3_iter Iterators
@ingroup lite3_buffer_api
@ingroup lite3_context_api
@{
*/
/// Return value of `lite3_iter_next()`; iterator produced an item, continue;
#define LITE3_ITER_ITEM 1
/// Return value of `lite3_iter_next()`; iterator finished; stop.
#define LITE3_ITER_DONE 0

/**
Struct containing iterator state.
See @ref lite3_iter.
The iterator struct is meant to be opaque, but is included in the header to support stack allocation and `sizeof()`.
@ingroup lite3_types
*/
typedef struct {
        uint32_t gen;
        uint32_t node_ofs[LITE3_TREE_HEIGHT_MAX + 1];
        uint8_t  depth;
        uint8_t  node_i[LITE3_TREE_HEIGHT_MAX + 1];
} lite3_iter;

#ifndef DOXYGEN_IGNORE
// Private function
int lite3_iter_create_impl(const unsigned char *buf, size_t buflen, size_t ofs, lite3_iter *out);
#endif // DOXYGEN_IGNORE

/**
Create a lite3 iterator for the given object or array

@return 0 on success
@return < 0 on error
*/
static inline int lite3_iter_create(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        size_t ofs,                     ///< [in] start offset (0 == root)
        lite3_iter *out)                ///< [out] iterator struct pointer
{
        int ret;
        if ((ret = _lite3_verify_get(buf, buflen, ofs)) < 0)
                return ret;
        return lite3_iter_create_impl(buf, buflen, ofs, out);
}

/**
Get the next item from a lite3 iterator

To use in conjunctions with @ref lite3_val_fns, the `*out_val_ofs` can be cast to `(lite3_val *)`.

@return `LITE3_ITER_ITEM` (== 1) on item produced
@return `LITE3_ITER_DONE` (== 0) on success (no more items)
@return < 0 on error

@note
If the user does not want a key or value, simply pass NULL. For array iterators, `out_key` should always be passed as `NULL`.

@warning
Iterators are read-only. Any attempt to write to the buffer using `lite3_set_xxxx()` will immediately invalidate the iterator.
If you need to make changes to the buffer, first prepare your changes, then apply them afterwards in one batch.
*/
int lite3_iter_next(
        const unsigned char *buf,       ///< [in] buffer pointer
        size_t buflen,                  ///< [in] buffer used length
        lite3_iter *iter,               ///< [in] iterator struct pointer
        lite3_str *out_key,             ///< [out] current key (if not needed, pass `NULL`)
        size_t *out_val_ofs             ///< [out] current value offset (if not needed, pass `NULL`)
);
/// @} lite3_iter



/**
Functions to deal with opaque values

The lite3_val struct represents a value inside a Lite³ buffer.

@defgroup lite3_val_fns lite3_val functions
@{
*/
/**
Returns the value type of `*val`
*/
static inline enum lite3_type lite3_val_type(lite3_val *val)
{
        enum lite3_type type = (enum lite3_type)val->type;
        return type < LITE3_TYPE_INVALID ? type : LITE3_TYPE_INVALID;
}

/**
Returns the size of the value type

@note
For variable sized types like LITE3_TYPE_BYTES or LITE3_TYPE_STRING, the number of bytes (including NULL-terminator for string) are written back.

@warning
This function assumes you have a valid `lite3_val`. Passing an invalid value will return an invalid size.
*/
static inline size_t lite3_val_type_size(lite3_val *val)
{
        enum lite3_type type = (enum lite3_type)val->type;
        if (type == LITE3_TYPE_STRING || type == LITE3_TYPE_BYTES) {
                size_t tmp = 0;
                memcpy(&tmp, val->val, lite3_type_sizes[LITE3_TYPE_BYTES]);
                return tmp;
        }
        return lite3_type_sizes[val->type];
}

static inline bool lite3_val_is_null(lite3_val *val) { return val->type == LITE3_TYPE_NULL; }
static inline bool lite3_val_is_bool(lite3_val *val) { return val->type == LITE3_TYPE_BOOL; }
static inline bool lite3_val_is_i64(lite3_val *val) { return val->type == LITE3_TYPE_I64; }
static inline bool lite3_val_is_f64(lite3_val *val) { return val->type == LITE3_TYPE_F64; }
static inline bool lite3_val_is_bytes(lite3_val *val) { return val->type == LITE3_TYPE_BYTES; }
static inline bool lite3_val_is_str(lite3_val *val) { return val->type == LITE3_TYPE_STRING; }
static inline bool lite3_val_is_obj(lite3_val *val) { return val->type == LITE3_TYPE_OBJECT; }
static inline bool lite3_val_is_arr(lite3_val *val) { return val->type == LITE3_TYPE_ARRAY; }

static inline bool lite3_val_bool(lite3_val *val)
{
        return *(bool *)(val->val);
}

static inline int64_t lite3_val_i64(lite3_val *val)
{
        int64_t tmp;
        memcpy(&tmp, val->val, sizeof(tmp));
        return tmp;
}

static inline double lite3_val_f64(lite3_val *val)
{
        double tmp;
        memcpy(&tmp, val->val, sizeof(tmp));
        return tmp;
}

static inline const char *lite3_val_str(lite3_val *val)
{
        return (const char *)val->val + LITE3_STR_LEN_SIZE;
}

/**
@warning
`*out_len` is exclusive of the NULL-terminator.
*/
static inline const char *lite3_val_str_n(lite3_val *val, size_t *out_len)
{
        *out_len = 0;
        memcpy(out_len, val->val, LITE3_STR_LEN_SIZE);
        *out_len -= 1; // Lite³ stores string size including NULL-terminator. Correction required for public API.
        return (const char *)val->val + LITE3_STR_LEN_SIZE;
}

static inline const unsigned char *lite3_val_bytes(lite3_val *val, size_t *out_len)
{
        *out_len = 0;
        memcpy(out_len, val->val, LITE3_BYTES_LEN_SIZE);
        return (const unsigned char *)val->val + LITE3_BYTES_LEN_SIZE;
}
/// @} lite3_val_fns



/**
Conversion between Lite³ and JSON

All JSON functionality is enabled internally by the yyjson library.

JSON encode functions read `buflen` to know the currently used portion of the buffer.

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
Read-only operations are thread-safe. This includes all JSON encode functions.
Decoding JSON however is not thread-safe for the given Lite³ buffer and requires manual locking.
Mixing reads and writes on the same Lite³ buffer is also not thread-safe.

@defgroup lite3_json JSON Conversion
@ingroup lite3_buffer_api
@{
*/
/**
@ingroup lite3_config
Enable JSON-related functions for conversion between Lite³ and JSON
*/
#define LITE3_JSON

#if defined(DOXYGEN_ONLY) && !defined(LITE3_JSON)
#define LITE3_JSON
#endif // DOXYGEN_ONLY

#ifdef LITE3_JSON
#include <stdio.h>

/**
Maximum nesting limit for JSON documents being encoded or decoded.
Default: 32

The conversion process is recursive and could otherwise risk a stack overflow with too many nesting layers.

@ingroup lite3_config
*/
#define LITE3_JSON_NESTING_DEPTH_MAX 32

/**
Convert JSON string to Lite³

The number of bytes written to `*buf` never exceeds `bufsz`.

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
int lite3_json_dec(
        unsigned char *buf,             ///< [in] Lite³ buffer pointer
        size_t *__restrict out_buflen,  ///< [out] buffer used length (bytes, out value)
        size_t bufsz,                   ///< [in] buffer max size (bytes)
        const char *__restrict json_str,///< [in] JSON input string (string)
        size_t json_len                 ///< [in] JSON input string length (bytes, including or excluding NULL-terminator)
);

/**
Convert JSON from file path to Lite³

The number of bytes written to `*buf` never exceeds `bufsz`.

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
int lite3_json_dec_file(
        unsigned char *buf,             ///< [in] Lite³ buffer pointer
        size_t *__restrict out_buflen,  ///< [out] buffer used length (bytes, out value)
        size_t bufsz,                   ///< [in] buffer max size (bytes)
        const char *__restrict path     ///< [in] JSON file path (string)
);

/**
Convert JSON from file pointer to Lite³

The number of bytes written to `*buf` never exceeds `bufsz`.

@return 0 on success
@return < 0 on error

@note
This function performs internal memory allocation using `malloc()`.
*/
int lite3_json_dec_fp(
        unsigned char *buf,             ///< [in] Lite³ buffer pointer
        size_t *__restrict out_buflen,  ///< [out] buffer used length (bytes, out value)
        size_t bufsz,                   ///< [in] buffer max size (bytes)
        FILE *fp                        ///< [in] JSON file pointer
);

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
int lite3_json_print(
        const unsigned char *buf,       ///< [in] Lite³ buffer
        size_t buflen,                  ///< [in] buffer used length (bytes)
        size_t ofs                      ///< [in] start offset (0 == root)
);

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
char *lite3_json_enc(
        const unsigned char *buf,       ///< [in] Lite³ buffer
        size_t buflen,                  ///< [in] buffer used length (bytes)
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_len      ///< [out] string length excluding NULL-terminator (if not needed, pass `NULL`)
);

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
char *lite3_json_enc_pretty(
        const unsigned char *buf,       ///< [in] Lite³ buffer
        size_t buflen,                  ///< [in] Lite³ buffer used length (bytes)
        size_t ofs,                     ///< [in] start offset (0 == root)
        size_t *__restrict out_len      ///< [out] string length excluding NULL-terminator (if not needed, pass `NULL`)
);

/**
Convert Lite³ to JSON and write to output buffer

@return >= 0 on success (number of bytes written)
@return < 0 on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.
*/
int64_t lite3_json_enc_buf(
        const unsigned char *buf,       ///< [in] Lite³ buffer
        size_t buflen,                  ///< [in] Lite³ buffer used length (bytes)
        size_t ofs,                     ///< [in] start offset (0 == root)
        char *__restrict json_buf,      ///< [in] JSON output buffer
        size_t json_bufsz               ///< [in] JSON output buffer max size (bytes)
);

/**
Convert Lite³ to prettified JSON and write to output buffer

The prettified string uses a tab space indent of 4.

@return >= 0 on success (number of bytes written)
@return < 0 on error

@note
1. This function performs internal memory allocation using `malloc()`.
2. Because JSON does not support encoding of raw bytes, `LITE3_TYPE_BYTES` are automatically converted to a base64 string.
*/
int64_t lite3_json_enc_pretty_buf(
        const unsigned char *buf,       ///< [in] Lite³ buffer
        size_t buflen,                  ///< [in] Lite³ buffer used length (bytes)
        size_t ofs,                     ///< [in] start offset (0 == root)
        char *__restrict json_buf,      ///< [in] JSON output buffer
        size_t json_bufsz               ///< [in] JSON output buffer max size (bytes)
);
#else
static inline int lite3_json_dec(unsigned char *buf, size_t *__restrict out_buflen, size_t bufsz, const char *__restrict json_str, size_t json_len)
{
        (void)buf; (void)out_buflen; (void)bufsz; (void)json_str; (void)json_len;
        return -1;
}

static inline int lite3_json_dec_file(unsigned char *buf, size_t *__restrict out_buflen, size_t bufsz, const char *__restrict path)
{
        (void)buf; (void)out_buflen; (void)bufsz; (void)path;
        return -1;
}

static inline int lite3_json_dec_fp(unsigned char *buf, size_t *__restrict out_buflen, size_t bufsz, FILE *fp)
{
        (void)buf; (void)out_buflen; (void)bufsz; (void)fp;
        return -1;
}

static inline int lite3_json_print(const unsigned char *buf, size_t buflen, size_t ofs)
{
        (void)buf; (void)buflen; (void)ofs;
        return 0;
}

static inline char *lite3_json_enc(const unsigned char *buf, size_t buflen, size_t ofs, size_t *out_len)
{
        (void)buf; (void)buflen; (void)ofs; (void)out_len;
        return NULL;
}

static inline char *lite3_json_enc_pretty(const unsigned char *buf, size_t buflen, size_t ofs, size_t *out_len)
{
        (void)buf; (void)buflen; (void)ofs; (void)out_len;
        return NULL;
}

static inline int64_t lite3_json_enc_buf(const unsigned char *buf, size_t buflen, size_t ofs, char *__restrict json_buf, size_t json_bufsz)
{
        (void)buf; (void)buflen; (void)ofs; (void)json_buf; (void)json_bufsz;
        return -1;
}

static inline int64_t lite3_json_enc_pretty_buf(const unsigned char *buf, size_t buflen, size_t ofs, char *__restrict json_buf, size_t json_bufsz)
{
        (void)buf; (void)buflen; (void)ofs; (void)json_buf; (void)json_bufsz;
        return -1;
}
#endif
/// @} lite3_json

#ifdef __cplusplus
}
#endif

#endif // LITE3_H