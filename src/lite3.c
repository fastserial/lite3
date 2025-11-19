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

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>



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



/*
        B-tree node struct
                Advanced users can adjust the B-tree node size to experiment with different performance characteristics.
                Larger node sizes will bloat message size, but also reduce the tree height, thus also reducing node walks.
                Actual effects are dependent on the architecture and workload. To be sure, experiment with different settings and profile.

                IMPORTANT RULES TO OBEY:
                - hashes[] and kv_ofs[] should always have an element count exactly equal to LITE3_NODE_KEY_COUNT_MASK
                - hashes[] and kv_ofs[] should always have an uneven element count
                - hashes[] and kv_ofs[] should always have equal element count
                - child_ofs[] should always have element count of exactly 1 greater than hashes[] and kv_ofs[]

        How to change:
                1) uncomment `LITE3_NODE_KEY_COUNT_MASK` to the preferred setting
                2) adjust the member array sizes inside the `struct node` definition
                3) uncomment `LITE3_NODE_SIZE` to the correct size inside the header
                4) uncomment LITE3_NODE_SIZE_KC_OFFSET` to the correct size inside the header
                5) uncomment `LITE3_TREE_HEIGHT_MAX` to the correct value inside the header
        
        [ WARNING ] If you change this setting, everyone you communicate with must also change it.
                    Unless you control all communicating parties, you probably should not touch this.
*/
struct node {
        u32	gen_type;       // upper 24 bits: gen           lower 8 bits: lite3_type
        u32	hashes[7];
        u32	size_kc;        // upper 26 bits: size          lower 6 bits: key_count
        u32	kv_ofs[7];
        u32	child_ofs[8];
};
static_assert(sizeof(struct node) == LITE3_NODE_SIZE, "sizeof(struct node) must equal LITE3_NODE_SIZE");
static_assert(offsetof(struct node, gen_type) == 0, "Runtime type checks and LITE3_BYTES() & LITE3_STR() macros expect to read (struct node).gen_type field at offset 0");
static_assert(sizeof(((struct node *)0)->gen_type) == sizeof(uint32_t), "LITE3_BYTES() & LITE3_STR() macros expect to read (struct node).gen_type as uint32_t");
static_assert(offsetof(struct node, size_kc) == LITE3_NODE_SIZE_KC_OFFSET, "Offset of (struct node).size_kc must equal LITE3_NODE_SIZE_KC_OFFSET");
static_assert(sizeof(((struct node *)0)->size_kc) == sizeof(uint32_t), "Node size checks expect to read (struct node).size_kc as uint32_t");
static_assert(sizeof(((struct node *)0)->gen_type) == sizeof(((lite3_iter *)0)->gen), "Iterator expects to read (struct node).gen_type as uint32_t");

#define LITE3_NODE_TYPE_SHIFT 0
#define LITE3_NODE_TYPE_MASK ((u32)((1 << 8) - 1))  // 8 LSB

#define LITE3_NODE_GEN_SHIFT  8
#define LITE3_NODE_GEN_MASK  ((u32)~((1 << 8) - 1)) // 24 MSB

#define LITE3_NODE_KEY_COUNT_MAX ((int)(sizeof(((struct node *)0)->hashes) / sizeof(u32)))
#define LITE3_NODE_KEY_COUNT_MIN ((int)(LITE3_NODE_KEY_COUNT_MAX / 2))

#define LITE3_NODE_KEY_COUNT_SHIFT 0
// #define LITE3_NODE_KEY_COUNT_MASK ((u32)((1 << 2) - 1))  // 2 LSB	key_count: 0-3          hashes[3]       kv_ofs[3]       child_ofs[4]	LITE3_NODE_SIZE: 48 (0.75 cache lines)
#define LITE3_NODE_KEY_COUNT_MASK ((u32)((1 << 3) - 1))  // 3 LSB	key_count: 0-7          hashes[7]       kv_ofs[7]       child_ofs[8]	LITE3_NODE_SIZE: 96 (1.5 cache lines)
// #define LITE3_NODE_KEY_COUNT_MASK ((u32)((1 << 4) - 1))  // 4 LSB	key_count: 0-15         hashes[15]      kv_ofs[15]      child_ofs[16]	LITE3_NODE_SIZE: 192 (3 cache lines)
// #define LITE3_NODE_KEY_COUNT_MASK ((u32)((1 << 5) - 1))  // 5 LSB	key_count: 0-31         hashes[31]      kv_ofs[31]      child_ofs[32]	LITE3_NODE_SIZE: 384 (6 cache lines)
// #define LITE3_NODE_KEY_COUNT_MASK ((u32)((1 << 6) - 1))  // 6 LSB	key_count: 0-63         hashes[63]      kv_ofs[63]      child_ofs[64]	LITE3_NODE_SIZE: 768 (12 cache lines)



#define LITE3_KEY_TAG_SIZE_MIN 1
#define LITE3_KEY_TAG_SIZE_MAX 4
#define LITE3_KEY_TAG_SIZE_MASK ((1 << 2) - 1)
#define LITE3_KEY_TAG_SIZE_SHIFT 0

#define LITE3_KEY_TAG_KEY_SIZE_MASK (~((1 << 2) - 1))
#define LITE3_KEY_TAG_KEY_SIZE_SHIFT 2
/*
        Verify a key inside the buffer to ensure readers don't go out of bounds.
                Optionally compare the existing key to an input key; a mismatch implies a hash collision.
                - Returns 0 on success
                - Returns < 0 on failure
        
        [ NOTE ] For internal use only.
*/
static inline int _verify_key(
	const u8 *buf,                  	// buffer pointer
	size_t buflen,                  	// buffer length (bytes)
	const char *restrict key,       	// key string (string, optionally call with NULL)
	size_t key_size,                	// key size (bytes including null-terminator, optionally call with 0)
	size_t key_tag_size,            	// key tag size (bytes, optionally call with 0)
	size_t *restrict inout_ofs,     	// key entry offset (relative to *buf)
	size_t *restrict out_key_tag_size)	// key tag size (optionally call with NULL)
{
	if (LITE3_UNLIKELY(LITE3_KEY_TAG_SIZE_MAX > buflen || *inout_ofs > buflen - LITE3_KEY_TAG_SIZE_MAX)) {
		LITE3_PRINT_ERROR("KEY ENTRY OUT OF BOUNDS\n");
		errno = EFAULT;
		return -1;
	}
	size_t _key_tag_size = (size_t)((*((u8 *)(buf + *inout_ofs)) & LITE3_KEY_TAG_SIZE_MASK) + 1);
	if (key_tag_size) {
		if (key_tag_size != _key_tag_size) {
			LITE3_PRINT_ERROR("KEY TAG SIZE DOES NOT MATCH\n");
			errno = EINVAL;
			return -1;
		}
	}
	size_t _key_size = 0;
	memcpy(&_key_size, buf + *inout_ofs, _key_tag_size);
	_key_size >>= LITE3_KEY_TAG_KEY_SIZE_SHIFT;
	*inout_ofs += _key_tag_size;

	if (LITE3_UNLIKELY(_key_size > buflen || *inout_ofs > buflen - _key_size)) {
		LITE3_PRINT_ERROR("KEY ENTRY OUT OF BOUNDS\n");
		errno = EFAULT;
		return -1;
	}
	if (key_size) {
		int cmp = memcmp(
			(const char *)(buf + *inout_ofs),
			key,
			(key_size < _key_size) ? key_size : _key_size
		);
		if (LITE3_UNLIKELY(cmp != 0)) {
			LITE3_PRINT_ERROR("HASH COLLISION\n");
			errno = EINVAL;
			return -1;
		}
	}
	*inout_ofs += _key_size;
	if (out_key_tag_size)
		*out_key_tag_size = _key_tag_size;
	return 0;
}

/*
        Verify a value inside the buffer to ensure readers don't go out of bounds.
                - Returns 0 on success
                - Returns < 0 on failure
        
        [ NOTE ] For internal use only.
*/
static inline int _verify_val(
	const u8 *buf,                  // buffer pointer
	size_t buflen,                  // buffer length (bytes)
	size_t *restrict inout_ofs)     // val entry offset (relative to *buf)
{	
	if (LITE3_UNLIKELY(LITE3_VAL_SIZE > buflen || *inout_ofs > buflen - LITE3_VAL_SIZE)) {
		LITE3_PRINT_ERROR("VALUE OUT OF BOUNDS\n");
		errno = EFAULT;
		return -1;
	}
	enum lite3_type type = (enum lite3_type)(*(buf + *inout_ofs));

	if (LITE3_UNLIKELY(type >= LITE3_TYPE_INVALID)) {
		LITE3_PRINT_ERROR("VALUE TYPE INVALID\n");
		errno = EINVAL;
		return -1;
	}
	size_t _val_entry_size = LITE3_VAL_SIZE + lite3_type_sizes[type];

	if (LITE3_UNLIKELY(_val_entry_size > buflen || *inout_ofs > buflen - _val_entry_size)) {
		LITE3_PRINT_ERROR("VALUE OUT OF BOUNDS\n");
		errno = EFAULT;
		return -1;
	}
	if (type == LITE3_TYPE_STRING || type == LITE3_TYPE_BYTES) {			// extra check required for str/bytes
		size_t byte_count = 0;
		memcpy(&byte_count, buf + *inout_ofs + LITE3_VAL_SIZE, lite3_type_sizes[LITE3_TYPE_BYTES]);
		_val_entry_size += byte_count;
		if (LITE3_UNLIKELY(_val_entry_size > buflen || *inout_ofs > buflen - _val_entry_size)) {
			LITE3_PRINT_ERROR("VALUE OUT OF BOUNDS\n");
			errno = EFAULT;
			return -1;
		}
	}
	*inout_ofs += _val_entry_size;
	return 0;
}

int lite3_get_impl(
	const unsigned char *buf,       // buffer pointer
	size_t buflen,                  // buffer length (bytes)
	size_t ofs,			// start offset (0 == root)
	const char *restrict key,       // key pointer (string)
	lite3_key_data key_data,        // key data struct
	lite3_val **out)                // value entry pointer (out pointer)
{
	#ifdef LITE3_DEBUG
	if (*(buf + ofs) == LITE3_TYPE_OBJECT) {
		LITE3_PRINT_DEBUG("GET\tkey: %s\n", key);
	} else if (*(buf + ofs) == LITE3_TYPE_ARRAY) {
		LITE3_PRINT_DEBUG("GET\tindex: %u\n", key_data.hash);
	} else {
		LITE3_PRINT_DEBUG("GET INVALID: EXEPCTING ARRAY OR OBJECT TYPE\n");
	}
	#endif

	size_t key_tag_size = (size_t)((!!(key_data.size >> (16 - LITE3_KEY_TAG_KEY_SIZE_SHIFT)) << 1)
					+ !!(key_data.size >> (8 - LITE3_KEY_TAG_KEY_SIZE_SHIFT))
					+ !!key_data.size);

	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);
	assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);			// *buf + ofs must be aligned to LITE3_NODE_ALIGNMENT

	int key_count;
	int i;
	int node_walks = 0;
	while (1) {
		key_count = node->size_kc & LITE3_NODE_KEY_COUNT_MASK;
		i = 0;
		while (i < key_count && node->hashes[i] < key_data.hash)
			i++;
		if (i < key_count && node->hashes[i] == key_data.hash) {		// target key found
			size_t target_ofs = node->kv_ofs[i];
			if (key && _verify_key(buf, buflen, key, (size_t)key_data.size, key_tag_size, &target_ofs, NULL) < 0)
				return -1;
			size_t val_start_ofs = target_ofs;
			if (_verify_val(buf, buflen, &target_ofs) < 0)
				return -1;
			*out = (lite3_val *)(buf + val_start_ofs);
			return 0;
		}
		if (node->child_ofs[0]) {						// if children, walk to next node
			size_t next_node_ofs = (size_t)node->child_ofs[i];
			node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
			assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);

			if (LITE3_UNLIKELY(next_node_ofs > buflen - LITE3_NODE_SIZE)) {
				LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
				errno = EFAULT;
				return -1;
			}
			if (LITE3_UNLIKELY(++node_walks > LITE3_TREE_HEIGHT_MAX)) {
				LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
				errno = EBADMSG;
				return -1;
			}
		} else {
			LITE3_PRINT_ERROR("KEY NOT FOUND\n");
			errno = ENOENT;
			return -1;
		}
	}
}

int lite3_iter_create_impl(const unsigned char *buf, size_t buflen, size_t ofs, lite3_iter *out)
{
	LITE3_PRINT_DEBUG("CREATE ITER\n");

	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);
	assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);			// node must be aligned to LITE3_NODE_ALIGNMENT

	enum lite3_type type = node->gen_type & LITE3_NODE_TYPE_MASK;
	if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
		errno = EINVAL;
		return -1;
	}
	out->gen = ((struct node *)buf)->gen_type;
	out->depth = 0;
	out->node_ofs[0] = (u32)ofs;
	out->node_i[0] = 0;

	while (node->child_ofs[0]) {							// has children, travel down
		u32 next_node_ofs = node->child_ofs[0];

		node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
		assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);

		if (LITE3_UNLIKELY(++out->depth > LITE3_TREE_HEIGHT_MAX)) {
			LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY((size_t)next_node_ofs > buflen - LITE3_NODE_SIZE)) {
			LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
			errno = EFAULT;
			return -1;
		}
		out->node_ofs[out->depth] = next_node_ofs;
		out->node_i[out->depth] = 0;
	}
	#ifdef LITE3_PREFETCHING
	__builtin_prefetch(buf + node->kv_ofs[0],      0, 0); // prefetch first few items
	__builtin_prefetch(buf + node->kv_ofs[0] + 64, 0, 0);
	__builtin_prefetch(buf + node->kv_ofs[1],      0, 0);
	__builtin_prefetch(buf + node->kv_ofs[1] + 64, 0, 0);
	__builtin_prefetch(buf + node->kv_ofs[2],      0, 0);
	__builtin_prefetch(buf + node->kv_ofs[2] + 64, 0, 0);
	#endif
	return 0;
}

int lite3_iter_next(const unsigned char *buf, size_t buflen, lite3_iter *iter, lite3_str *out_key, size_t *out_val_ofs)
{
	if (LITE3_UNLIKELY(iter->gen != ((struct node *)buf)->gen_type)) {
		LITE3_PRINT_ERROR("ITERATOR INVALID: iter->gen != node->gen_type (BUFFER MUTATION INVALIDATES ITERATORS)\n");
		errno = EINVAL;
		return -1;
	}

	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + iter->node_ofs[iter->depth]), LITE3_NODE_ALIGNMENT);
	assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);			// node must be aligned to LITE3_NODE_ALIGNMENT

	enum lite3_type type = node->gen_type & LITE3_NODE_TYPE_MASK;
	if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY ORlite3_iter_nex OBJECT TYPE\n");
		errno = EINVAL;
		return -1;
	}
	if (iter->depth == 0 && (iter->node_i[iter->depth] == (node->size_kc & LITE3_NODE_KEY_COUNT_MASK))) { // key_count reached, done
		return LITE3_ITER_DONE;
	}
	size_t target_ofs = node->kv_ofs[iter->node_i[iter->depth]];

	int ret;
	if (type == LITE3_TYPE_OBJECT && out_key) {					// write back key if not NULL
		size_t key_tag_size;
		size_t key_start_ofs = target_ofs;
		if ((ret = _verify_key(buf, buflen, NULL, 0, 0, &target_ofs, &key_tag_size)) < 0)
			return ret;
		out_key->gen = iter->gen;
		out_key->len = 0;
		memcpy(&out_key->len, buf + key_start_ofs, key_tag_size);
		--out_key->len; // Lite³ stores string size including NULL-terminator. Correction required for public API.
		out_key->ptr = (const char *)(buf + key_start_ofs + key_tag_size);
	}
	if (out_val_ofs) {								// write back val if not NULL
		size_t val_start_ofs = target_ofs;
		if ((ret = _verify_val(buf, buflen, &target_ofs)) < 0)
			return ret;
		*out_val_ofs = val_start_ofs;
	}

	++iter->node_i[iter->depth];

	while (node->child_ofs[iter->node_i[iter->depth]]) {				// has children, travel down
		u32 next_node_ofs = node->child_ofs[iter->node_i[iter->depth]];

		node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
		assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);

		if (LITE3_UNLIKELY(++iter->depth > LITE3_TREE_HEIGHT_MAX)) {
			LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY((size_t)next_node_ofs > buflen - LITE3_NODE_SIZE)) {
			LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
			errno = EFAULT;
			return -1;
		}
		iter->node_ofs[iter->depth] = next_node_ofs;
		iter->node_i[iter->depth] = 0;
	}
	while (iter->depth > 0 && (iter->node_i[iter->depth] == (node->size_kc & LITE3_NODE_KEY_COUNT_MASK))) { // key_count reached, go up
		--iter->depth;
		node = __builtin_assume_aligned((struct node *)(buf + iter->node_ofs[iter->depth]), LITE3_NODE_ALIGNMENT);
		assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);
		#ifdef LITE3_PREFETCHING
		__builtin_prefetch(buf + node->child_ofs[(iter->node_i[iter->depth] + 1) & LITE3_NODE_KEY_COUNT_MASK],      0, 2); // prefetch next nodes
		__builtin_prefetch(buf + node->child_ofs[(iter->node_i[iter->depth] + 1) & LITE3_NODE_KEY_COUNT_MASK] + 64, 0, 2);
		__builtin_prefetch(buf + node->child_ofs[(iter->node_i[iter->depth] + 2) & LITE3_NODE_KEY_COUNT_MASK],      0, 2);
		__builtin_prefetch(buf + node->child_ofs[(iter->node_i[iter->depth] + 2) & LITE3_NODE_KEY_COUNT_MASK] + 64, 0, 2);
		#endif
	}
	#ifdef LITE3_PREFETCHING
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth]],          0, 0); // prefetch next items
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth]]     + 64, 0, 0);
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth] + 1],      0, 0);
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth] + 1] + 64, 0, 0);
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth] + 2],      0, 0);
	__builtin_prefetch(buf + node->kv_ofs[iter->node_i[iter->depth] + 2] + 64, 0, 0);
	#endif
	return LITE3_ITER_ITEM;
}


static inline void _lite3_init_impl(unsigned char *buf, size_t ofs, enum lite3_type type)
{
	LITE3_PRINT_DEBUG("INITIALIZE %s\n", type == LITE3_TYPE_OBJECT ? "OBJECT" : "ARRAY");

	struct node *node = (struct node *)(buf + ofs);
	assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);			// node must be aligned to LITE3_NODE_ALIGNMENT
	node->gen_type = type & LITE3_NODE_TYPE_MASK;
	node->size_kc = 0x00;
	#ifdef LITE3_ZERO_MEM_EXTRA
		memset(node->hashes, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->hashes));
		memset(node->kv_ofs, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->kv_ofs));
	#endif
	memset(node->child_ofs, 0x00, sizeof(((struct node *)0)->child_ofs));
}

int lite3_init_obj(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz)
{
	if (LITE3_UNLIKELY(bufsz < LITE3_NODE_SIZE)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < LITE3_NODE_SIZE\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_OBJECT);
	*out_buflen = LITE3_NODE_SIZE;
	return 0;
}

int lite3_init_arr(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz)
{
	if (LITE3_UNLIKELY(bufsz < LITE3_NODE_SIZE)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < LITE3_NODE_SIZE\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_ARRAY);
	*out_buflen = LITE3_NODE_SIZE;
	return 0;
}

/*
        Inserts entry into the Lite³ structure to prepare for writing of the actual value.
                - Returns 0 on success
                - Returns < 0 on failure

        [ NOTE ] This function expects the caller to write to:
                        1) `val->type`: the value type (bytes written should equal to `LITE3_VAL_SIZE`)
                        2) `val->val`: the actual value (bytes written should equal `val_len`)
                 This has the advantage that the responsibility of type-specific logic is also moved to the caller.
                 Otherwise, this function would have to contain branches to account for all types.
*/
int lite3_set_impl(
	unsigned char *buf,             // buffer pointer
	size_t *restrict inout_buflen,  // buffer used length (bytes, inout value)
	size_t ofs,                     // start offset (0 == root)
	size_t bufsz,                   // buffer max size (bytes)
	const char *restrict key,       // key string (string, pass NULL when inserting in array)
	lite3_key_data key_data,        // key data struct
	size_t val_len,                 // value length (bytes)
	lite3_val **out)                // value entry pointer (out pointer)
{
	#ifdef LITE3_DEBUG
	if (*(buf + ofs) == LITE3_TYPE_OBJECT) {
		LITE3_PRINT_DEBUG("SET\tkey: %s\n", key);
	} else if (*(buf + ofs) == LITE3_TYPE_ARRAY) {
		LITE3_PRINT_DEBUG("SET\tindex: %u\n", key_data.hash);
	} else {
		LITE3_PRINT_DEBUG("SET INVALID: EXEPCTING ARRAY OR OBJECT TYPE\n");
	}
	#endif
	size_t key_tag_size = (size_t)((!!(key_data.size >> (16 - LITE3_KEY_TAG_KEY_SIZE_SHIFT)) << 1)
					+ !!(key_data.size >> (8 - LITE3_KEY_TAG_KEY_SIZE_SHIFT))
					+ !!key_data.size);
	size_t entry_size = key_tag_size + (size_t)key_data.size + LITE3_VAL_SIZE + val_len;

	struct node *restrict parent = NULL;
	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);
	assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);				// *buf + ofs must be aligned to LITE3_NODE_ALIGNMENT

	u32 gen = node->gen_type >> LITE3_NODE_GEN_SHIFT;
	++gen;
	node->gen_type = (node->gen_type & ~LITE3_NODE_GEN_MASK) | (gen << LITE3_NODE_GEN_SHIFT);
	
	int key_count;
	int i;
	int node_walks = 0;
	while (1) {
		if ((node->size_kc & LITE3_NODE_KEY_COUNT_MASK) == LITE3_NODE_KEY_COUNT_MAX) {	// node full, need to split

			size_t buflen_aligned = (*inout_buflen + LITE3_NODE_ALIGNMENT_MASK) & ~(size_t)LITE3_NODE_ALIGNMENT_MASK; // next multiple of LITE3_NODE_ALIGNMENT
			size_t new_node_size = parent ? LITE3_NODE_SIZE : 2 * LITE3_NODE_SIZE;

			if (LITE3_UNLIKELY(new_node_size > bufsz || buflen_aligned > bufsz - new_node_size)) {
				LITE3_PRINT_ERROR("NO BUFFER SPACE FOR NODE SPLIT\n");
				errno = ENOBUFS;
				return -1;
			}
			*inout_buflen = buflen_aligned;
			// TODO: add lost bytes from alignment to GC index
			if (!parent) {								// if root split, create new root
				LITE3_PRINT_DEBUG("NEW ROOT\n");
				memcpy(buf + *inout_buflen, node, LITE3_NODE_SIZE);
				node = __builtin_assume_aligned((struct node *)(buf + *inout_buflen), LITE3_NODE_ALIGNMENT);
				assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);
				parent = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);
				assert(((uintptr_t)parent & LITE3_NODE_ALIGNMENT_MASK) == 0);
				#ifdef LITE3_ZERO_MEM_EXTRA
					memset(parent->hashes, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->hashes));
					memset(parent->kv_ofs, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->kv_ofs));
					memset(parent->child_ofs, 0x00,          sizeof(((struct node *)0)->child_ofs));
				#endif
				parent->size_kc &= ~LITE3_NODE_KEY_COUNT_MASK;			// set key_count to 0
				parent->child_ofs[0] = (u32)*inout_buflen;			// insert node as child of new root
				*inout_buflen += LITE3_NODE_SIZE;
				key_count = 0;
				i = 0;
			}
			LITE3_PRINT_DEBUG("SPLIT NODE\n");
			for (int j = key_count; j > i; j--) {					// shift parent array before separator insert
				parent->hashes[j] =        parent->hashes[j - 1];
				parent->kv_ofs[j] =        parent->kv_ofs[j - 1];
				parent->child_ofs[j + 1] = parent->child_ofs[j];
			}
			parent->hashes[i] = node->hashes[LITE3_NODE_KEY_COUNT_MIN];		// insert new separator key in parent
			parent->kv_ofs[i] = node->kv_ofs[LITE3_NODE_KEY_COUNT_MIN];
			parent->child_ofs[i + 1] = (u32)*inout_buflen;				// insert sibling as child in parent
			parent->size_kc = (parent->size_kc & ~LITE3_NODE_KEY_COUNT_MASK)
			                    | ((parent->size_kc + 1) & LITE3_NODE_KEY_COUNT_MASK); // key_count++
			#ifdef LITE3_ZERO_MEM_EXTRA
				node->hashes[LITE3_NODE_KEY_COUNT_MIN] = LITE3_ZERO_MEM_32;
				node->kv_ofs[LITE3_NODE_KEY_COUNT_MIN] = LITE3_ZERO_MEM_32;
			#endif
			struct node *restrict sibling = __builtin_assume_aligned((struct node *)(buf + *inout_buflen), LITE3_NODE_ALIGNMENT);
			assert(((uintptr_t)sibling & LITE3_NODE_ALIGNMENT_MASK) == 0);
			#ifdef LITE3_ZERO_MEM_EXTRA
				memset(sibling->hashes, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->hashes));
				memset(sibling->kv_ofs, LITE3_ZERO_MEM_8, sizeof(((struct node *)0)->kv_ofs));
			#endif
			sibling->gen_type = ((struct node *)(buf + ofs))->gen_type & LITE3_NODE_TYPE_MASK;
			sibling->size_kc = 	LITE3_NODE_KEY_COUNT_MIN & LITE3_NODE_KEY_COUNT_MASK;
			node->size_kc = 	LITE3_NODE_KEY_COUNT_MIN & LITE3_NODE_KEY_COUNT_MASK;
			memset(sibling->child_ofs, 0x00, sizeof(((struct node *)0)->child_ofs));
			sibling->child_ofs[0] = node->child_ofs[LITE3_NODE_KEY_COUNT_MIN + 1];	// take child from node
			                        node->child_ofs[LITE3_NODE_KEY_COUNT_MIN + 1] = 0x00;
			for (int j = 0; j < LITE3_NODE_KEY_COUNT_MIN; j++) {			// copy half of node's keys to sibling
				sibling->hashes[j] =        node->hashes[j + LITE3_NODE_KEY_COUNT_MIN + 1];
				sibling->kv_ofs[j] =        node->kv_ofs[j + LITE3_NODE_KEY_COUNT_MIN + 1];
				sibling->child_ofs[j + 1] = node->child_ofs[j + LITE3_NODE_KEY_COUNT_MIN + 2];
				#ifdef LITE3_ZERO_MEM_EXTRA
					node->hashes[j + LITE3_NODE_KEY_COUNT_MIN + 1] =    LITE3_ZERO_MEM_32;
					node->kv_ofs[j + LITE3_NODE_KEY_COUNT_MIN + 1] =    LITE3_ZERO_MEM_32;
					node->child_ofs[j + LITE3_NODE_KEY_COUNT_MIN + 2] = 0x00000000;
				#endif
			}
			if (key_data.hash > parent->hashes[i]) {				// sibling has target key? then we follow
				node = __builtin_assume_aligned(sibling, LITE3_NODE_ALIGNMENT);
				assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);
			}
			*inout_buflen += LITE3_NODE_SIZE;
		}

		key_count = node->size_kc & LITE3_NODE_KEY_COUNT_MASK;
		i = 0;
		while (i < key_count && node->hashes[i] < key_data.hash)
			i++;
		if (i < key_count && node->hashes[i] == key_data.hash) {			// matching key found, already exists?
			size_t target_ofs = node->kv_ofs[i];
			size_t key_start_ofs = target_ofs;
			if (key && _verify_key(buf, *inout_buflen, key, (size_t)key_data.size, key_tag_size, &target_ofs, NULL) < 0)
				return -1;
			size_t val_start_ofs = target_ofs;
			if (_verify_val(buf, *inout_buflen, &target_ofs) < 0)
				return -1;
			if (val_len >= target_ofs - val_start_ofs) {				// value is too large, we must append
				size_t alignment_mask = val_len == lite3_type_sizes[LITE3_TYPE_OBJECT] ? (size_t)LITE3_NODE_ALIGNMENT_MASK : 0;
				size_t unaligned_val_ofs = *inout_buflen + key_tag_size + (size_t)key_data.size;
				size_t alignment_padding = ((unaligned_val_ofs + alignment_mask) & ~alignment_mask) - unaligned_val_ofs;
				entry_size += alignment_padding;
				if (LITE3_UNLIKELY(entry_size > bufsz || *inout_buflen > bufsz - entry_size)) {
					LITE3_PRINT_ERROR("NO BUFFER SPACE FOR ENTRY INSERTION\n");
					errno = ENOBUFS;
					return -1;
				}
				#ifdef LITE3_ZERO_MEM_DELETED
					memset(buf + node->kv_ofs[i], LITE3_ZERO_MEM_8, target_ofs - key_start_ofs); // zero out key + value
				#endif
				(void)key_start_ofs;						// silence unused variable warning
				*inout_buflen += alignment_padding;
				node->kv_ofs[i] = (u32)*inout_buflen;
				goto insert_append;
				// TODO: add lost bytes to GC index
			}
			#ifdef LITE3_ZERO_MEM_DELETED
				memset(buf + val_start_ofs, LITE3_ZERO_MEM_8, target_ofs - val_start_ofs); // zero out value
			#endif
			*out = (lite3_val *)(buf + val_start_ofs);				// caller overwrites value in place
			// TODO: add lost bytes to GC index
			return 0;
		}
		if (node->child_ofs[0]) {							// if children, walk to next node
			size_t next_node_ofs = (size_t)node->child_ofs[i];

			parent = __builtin_assume_aligned(node, LITE3_NODE_ALIGNMENT);
			assert(((uintptr_t)parent & LITE3_NODE_ALIGNMENT_MASK) == 0);
			node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
			assert(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) == 0);

			if (LITE3_UNLIKELY(next_node_ofs > *inout_buflen - LITE3_NODE_SIZE)) {
				LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
				errno = EFAULT;
				return -1;
			}
			if (LITE3_UNLIKELY(++node_walks > LITE3_TREE_HEIGHT_MAX)) {
				LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
				errno = EBADMSG;
				return -1;
			}
		} else {									// insert the kv-pair
			size_t alignment_mask = val_len == lite3_type_sizes[LITE3_TYPE_OBJECT] ? (size_t)LITE3_NODE_ALIGNMENT_MASK : 0;
			size_t unaligned_val_ofs = *inout_buflen + key_tag_size + (size_t)key_data.size;
			size_t alignment_padding = ((unaligned_val_ofs + alignment_mask) & ~alignment_mask) - unaligned_val_ofs;
			entry_size += alignment_padding;
			if (LITE3_UNLIKELY(entry_size > bufsz || *inout_buflen > bufsz - entry_size)) {
				LITE3_PRINT_ERROR("NO BUFFER SPACE FOR ENTRY INSERTION\n");
				errno = ENOBUFS;
				return -1;
			}
			for (int j = key_count; j > i; j--) {
				node->hashes[j] = node->hashes[j - 1];
				node->kv_ofs[j] = node->kv_ofs[j - 1];
			}
			node->hashes[i] = key_data.hash;
			node->size_kc = (node->size_kc & ~LITE3_NODE_KEY_COUNT_MASK)
			                  | ((node->size_kc + 1) & LITE3_NODE_KEY_COUNT_MASK);	// key_count++
			*inout_buflen += alignment_padding;
			node->kv_ofs[i] = (u32)*inout_buflen;

			node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT); // set node to root
			u32 size = node->size_kc >> LITE3_NODE_SIZE_SHIFT;
			++size;
			node->size_kc = (node->size_kc & ~LITE3_NODE_SIZE_MASK) | (size << LITE3_NODE_SIZE_SHIFT); // node size++
			goto insert_append;
		}
	}
insert_append:
	size_t key_size_tmp = (key_data.size << LITE3_KEY_TAG_KEY_SIZE_SHIFT) | (key_tag_size - 1);
	memcpy(buf + *inout_buflen, &key_size_tmp, key_tag_size);
	*inout_buflen += key_tag_size;
	memcpy(buf + *inout_buflen, key, (size_t)key_data.size);
	*inout_buflen += (size_t)key_data.size;
	*out = (lite3_val *)(buf + *inout_buflen);
	*inout_buflen += LITE3_VAL_SIZE + val_len;
	return 0;
}

int lite3_set_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, size_t *restrict out_ofs)
{
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_OBJECT], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_OBJECT);
	return ret;
}

int lite3_set_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, size_t *restrict out_ofs)
{
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, lite3_type_sizes[LITE3_TYPE_ARRAY], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_ARRAY);
	return ret;
}

int lite3_arr_append_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t *restrict out_ofs)
{
	u32 size = ((struct node *)(buf + ofs))->size_kc >> LITE3_NODE_SIZE_SHIFT;
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, lite3_type_sizes[LITE3_TYPE_OBJECT], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_OBJECT);
	return ret;
}

int lite3_arr_append_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t *restrict out_ofs)
{
	u32 size = ((struct node *)(buf + ofs))->size_kc >> LITE3_NODE_SIZE_SHIFT;
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, lite3_type_sizes[LITE3_TYPE_ARRAY], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_ARRAY);
	return ret;
}

int lite3_arr_set_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *restrict out_ofs)
{
	u32 size = ((struct node *)(buf + ofs))->size_kc >> LITE3_NODE_SIZE_SHIFT;
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, lite3_type_sizes[LITE3_TYPE_OBJECT], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_OBJECT);
	return ret;
}

int lite3_arr_set_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *restrict out_ofs)
{
	u32 size = ((struct node *)(buf + ofs))->size_kc >> LITE3_NODE_SIZE_SHIFT;
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	lite3_val *val;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, NULL, key_data, lite3_type_sizes[LITE3_TYPE_ARRAY], &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, LITE3_TYPE_ARRAY);
	return ret;
}