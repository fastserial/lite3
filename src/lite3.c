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
        B-tree node layout helpers.
        Nodes store their configuration id in gen_type (see lite3.h). Arrays are computed from the selected layout.
*/
struct node {
        u32	gen_type;       // upper bits: generation       lower bits: type + cfg id
};
static_assert(offsetof(struct node, gen_type) == 0, "Runtime type checks and LITE3_BYTES() & LITE3_STR() macros expect to read (struct node).gen_type field at offset 0");
static_assert(sizeof(((struct node *)0)->gen_type) == sizeof(uint32_t), "LITE3_BYTES() & LITE3_STR() macros expect to read (struct node).gen_type as uint32_t");
static_assert(sizeof(((struct node *)0)->gen_type) == sizeof(((lite3_iter *)0)->gen), "Iterator expects to read (struct node).gen_type as uint32_t");

static inline bool _lite3_node_aligned(const void *ptr) {
        return (((uintptr_t)ptr & LITE3_NODE_ALIGNMENT_MASK) == 0);
}

static inline const lite3_node_cfg *_lite3_node_cfg_from_gen_type(u32 gen_type)
{
        return lite3_node_cfg_from_id((enum lite3_node_cfg_id)((gen_type & LITE3_NODE_CFG_MASK) >> LITE3_NODE_CFG_SHIFT));
}

static inline const lite3_node_cfg *_lite3_node_cfg_from_node(const struct node *node)
{
        return _lite3_node_cfg_from_gen_type(node->gen_type);
}

static inline u32 *_lite3_node_hashes(const lite3_node_cfg *cfg, struct node *node)
{
        (void)cfg;
        return (u32 *)((u8 *)node + sizeof(u32));
}

static inline const u32 *_lite3_node_hashes_c(const lite3_node_cfg *cfg, const struct node *node)
{
        (void)cfg;
        return (const u32 *)((const u8 *)node + sizeof(u32));
}

static inline u32 *_lite3_node_size_kc(const lite3_node_cfg *cfg, struct node *node)
{
        return (u32 *)((u8 *)node + cfg->size_kc_offset);
}

static inline const u32 *_lite3_node_size_kc_c(const lite3_node_cfg *cfg, const struct node *node)
{
        return (const u32 *)((const u8 *)node + cfg->size_kc_offset);
}

static inline u32 *_lite3_node_kv_ofs(const lite3_node_cfg *cfg, struct node *node)
{
        return (u32 *)((u8 *)node + cfg->size_kc_offset + sizeof(u32));
}

static inline const u32 *_lite3_node_kv_ofs_c(const lite3_node_cfg *cfg, const struct node *node)
{
        return (const u32 *)((const u8 *)node + cfg->size_kc_offset + sizeof(u32));
}

static inline u32 *_lite3_node_child_ofs(const lite3_node_cfg *cfg, struct node *node)
{
        return _lite3_node_kv_ofs(cfg, node) + cfg->key_count_max;
}

static inline const u32 *_lite3_node_child_ofs_c(const lite3_node_cfg *cfg, const struct node *node)
{
        return _lite3_node_kv_ofs_c(cfg, node) + cfg->key_count_max;
}

static inline u32 _lite3_node_key_count(const lite3_node_cfg *cfg, u32 size_kc)
{
        return size_kc & cfg->key_count_mask;
}

static inline u32 _lite3_node_size(const lite3_node_cfg *cfg, u32 size_kc)
{
        (void)cfg;
        return size_kc >> LITE3_NODE_SIZE_SHIFT;
}

static inline u32 _lite3_pack_size_kc(u32 size, u32 key_count)
{
	return (size << LITE3_NODE_SIZE_SHIFT) | (key_count & LITE3_NODE_KEY_COUNT_MASK);
}

static inline size_t _lite3_alignment_mask_for_val_len(size_t val_len)
{
	for (size_t i = 0; i < LITE3_NODE_CFG_COUNT; i++) {
		size_t candidate = lite3_node_cfg_table[i].node_size - LITE3_VAL_SIZE;
		if (candidate == val_len)
			return (size_t)LITE3_NODE_ALIGNMENT_MASK;
	}
	return 0;
}

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

	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, buflen, ofs, &cfg, NULL) < 0)
		return -1;

	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);

	if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
		LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
		errno = EBADMSG;
		return -1;
	}

	int key_count;
	int i;
	int node_walks = 0;
	while (1) {
		const u32 *hashes = _lite3_node_hashes_c(cfg, node);
		const u32 *kv_ofs = _lite3_node_kv_ofs_c(cfg, node);
		const u32 *child_ofs = _lite3_node_child_ofs_c(cfg, node);
		const u32 *size_kc_ptr = _lite3_node_size_kc_c(cfg, node);
		key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);
		i = 0;
		while (i < key_count && hashes[i] < key_data.hash)
			i++;
		if (i < key_count && hashes[i] == key_data.hash) {		// target key found
			size_t target_ofs = kv_ofs[i];
			if (key && _verify_key(buf, buflen, key, (size_t)key_data.size, key_tag_size, &target_ofs, NULL) < 0)
				return -1;
			size_t val_start_ofs = target_ofs;
			if (_verify_val(buf, buflen, &target_ofs) < 0)
				return -1;
			*out = (lite3_val *)(buf + val_start_ofs);
			return 0;
		}
		if (child_ofs[0]) {						// if children, walk to next node
			size_t next_node_ofs = (size_t)child_ofs[i];
			node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);

			if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
				LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
				errno = EBADMSG;
				return -1;
			}
			if (LITE3_UNLIKELY(next_node_ofs > buflen - cfg->node_size)) {
				LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
				errno = EFAULT;
				return -1;
			}
			if (LITE3_UNLIKELY(++node_walks > cfg->tree_height_max)) {
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

	const lite3_node_cfg *cfg = NULL;
	if (_lite3_cfg_for_offset(buf, buflen, ofs, &cfg, NULL) < 0)
		return -1;

	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);

	if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
		LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
		errno = EBADMSG;
		return -1;
	}

	enum lite3_type type = (enum lite3_type)(node->gen_type & LITE3_NODE_TYPE_MASK);
	if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
		errno = EINVAL;
		return -1;
	}
	out->gen = ((struct node *)buf)->gen_type;
	out->cfg_id = cfg->id;
	out->key_count_mask = cfg->key_count_mask;
	out->tree_height_max = cfg->tree_height_max;
	out->depth = 0;
	out->node_ofs[0] = (u32)ofs;
	out->node_i[0] = 0;

	const u32 *child_ofs = _lite3_node_child_ofs_c(cfg, node);
	while (child_ofs[0]) {							// has children, travel down
		u32 next_node_ofs = child_ofs[0];

		node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
		
		if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
			LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY(++out->depth > cfg->tree_height_max || out->depth > LITE3_TREE_HEIGHT_MAX_STATIC)) {
			LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY((size_t)next_node_ofs > buflen - cfg->node_size)) {
			LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
			errno = EFAULT;
			return -1;
		}
		out->node_ofs[out->depth] = next_node_ofs;
		out->node_i[out->depth] = 0;
		child_ofs = _lite3_node_child_ofs_c(cfg, node);
	}
	#ifdef LITE3_PREFETCHING
	const u32 *kv_ofs = _lite3_node_kv_ofs_c(cfg, node);
	__builtin_prefetch(buf + kv_ofs[0],      0, 0); // prefetch first few items
	__builtin_prefetch(buf + kv_ofs[0] + 64, 0, 0);
	__builtin_prefetch(buf + kv_ofs[1 & cfg->key_count_mask],      0, 0);
	__builtin_prefetch(buf + kv_ofs[1 & cfg->key_count_mask] + 64, 0, 0);
	__builtin_prefetch(buf + kv_ofs[2 & cfg->key_count_mask],      0, 0);
	__builtin_prefetch(buf + kv_ofs[2 & cfg->key_count_mask] + 64, 0, 0);
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

	if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
		LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
		errno = EBADMSG;
		return -1;
	}
	u32 node_cfg_id = (node->gen_type & LITE3_NODE_CFG_MASK) >> LITE3_NODE_CFG_SHIFT;
	if (LITE3_UNLIKELY(node_cfg_id >= LITE3_NODE_CFG_COUNT)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: NODE CFG ID OUT OF RANGE\n");
		errno = EBADMSG;
		return -1;
	}
	if (LITE3_UNLIKELY(node_cfg_id != iter->cfg_id)) {
		LITE3_PRINT_ERROR("ITERATOR INVALID: cfg mismatch\n");
		errno = EBADMSG;
		return -1;
	}
	const lite3_node_cfg *cfg = &lite3_node_cfg_table[node_cfg_id];
	if (LITE3_UNLIKELY(iter->depth > cfg->tree_height_max || iter->depth > LITE3_TREE_HEIGHT_MAX_STATIC)) {
		LITE3_PRINT_ERROR("ITERATOR INVALID: depth exceeds maximum\n");
		errno = EINVAL;
		return -1;
	}
	enum lite3_type type = node->gen_type & LITE3_NODE_TYPE_MASK;
	if (LITE3_UNLIKELY(!(type == LITE3_TYPE_OBJECT || type == LITE3_TYPE_ARRAY))) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: EXPECTING ARRAY OR OBJECT TYPE\n");
		errno = EINVAL;
		return -1;
	}
	if (LITE3_UNLIKELY(buflen < iter->node_ofs[iter->depth] + cfg->node_size)) {
		LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
		errno = EFAULT;
		return -1;
	}
	const u32 *size_kc_ptr = _lite3_node_size_kc_c(cfg, node);
	int key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);
	if (iter->depth == 0 && (iter->node_i[iter->depth] == key_count)) { // key_count reached, done
		return LITE3_ITER_DONE;
	}
	if (LITE3_UNLIKELY(key_count == 0))
		return LITE3_ITER_DONE;
	const u32 *kv_ofs = _lite3_node_kv_ofs_c(cfg, node);
	const u32 *child_ofs = _lite3_node_child_ofs_c(cfg, node);
	size_t target_ofs = kv_ofs[iter->node_i[iter->depth] & cfg->key_count_mask];

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

	while (child_ofs[iter->node_i[iter->depth] & cfg->key_count_mask]) {				// has children, travel down
		u32 next_node_ofs = child_ofs[iter->node_i[iter->depth] & cfg->key_count_mask];

		node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);
		
		if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
			LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
			errno = EBADMSG;
			return -1;
		}
		node_cfg_id = (node->gen_type & LITE3_NODE_CFG_MASK) >> LITE3_NODE_CFG_SHIFT;
		if (LITE3_UNLIKELY(node_cfg_id >= LITE3_NODE_CFG_COUNT || node_cfg_id != iter->cfg_id)) {
			LITE3_PRINT_ERROR("ITERATOR INVALID: cfg mismatch\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY(++iter->depth > cfg->tree_height_max || iter->depth > LITE3_TREE_HEIGHT_MAX_STATIC)) {
			LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
			errno = EBADMSG;
			return -1;
		}
		if (LITE3_UNLIKELY((size_t)next_node_ofs > buflen - cfg->node_size)) {
			LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
			errno = EFAULT;
			return -1;
		}
		iter->node_ofs[iter->depth] = next_node_ofs;
		iter->node_i[iter->depth] = 0;
		size_kc_ptr = _lite3_node_size_kc_c(cfg, node);
		key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);
		kv_ofs = _lite3_node_kv_ofs_c(cfg, node);
		child_ofs = _lite3_node_child_ofs_c(cfg, node);
	}
	while (iter->depth > 0 && (iter->node_i[iter->depth] == key_count)) { // key_count reached, go up
		--iter->depth;
		node = __builtin_assume_aligned((struct node *)(buf + iter->node_ofs[iter->depth]), LITE3_NODE_ALIGNMENT);
		
		if (LITE3_UNLIKELY(((uintptr_t)node & LITE3_NODE_ALIGNMENT_MASK) != 0)) {
			LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
			errno = EBADMSG;
			return -1;
		}
		node_cfg_id = (node->gen_type & LITE3_NODE_CFG_MASK) >> LITE3_NODE_CFG_SHIFT;
		if (LITE3_UNLIKELY(node_cfg_id >= LITE3_NODE_CFG_COUNT || node_cfg_id != iter->cfg_id)) {
			LITE3_PRINT_ERROR("ITERATOR INVALID: cfg mismatch\n");
			errno = EBADMSG;
			return -1;
		}
		size_kc_ptr = _lite3_node_size_kc_c(cfg, node);
		key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);
		kv_ofs = _lite3_node_kv_ofs_c(cfg, node);
		child_ofs = _lite3_node_child_ofs_c(cfg, node);
		#ifdef LITE3_PREFETCHING
		__builtin_prefetch(buf + child_ofs[(iter->node_i[iter->depth] + 1) & cfg->key_count_mask],      0, 2); // prefetch next nodes
		__builtin_prefetch(buf + child_ofs[(iter->node_i[iter->depth] + 1) & cfg->key_count_mask] + 64, 0, 2);
		__builtin_prefetch(buf + child_ofs[(iter->node_i[iter->depth] + 2) & cfg->key_count_mask],      0, 2);
		__builtin_prefetch(buf + child_ofs[(iter->node_i[iter->depth] + 2) & cfg->key_count_mask] + 64, 0, 2);
		#endif
	}
	#ifdef LITE3_PREFETCHING
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 0) & cfg->key_count_mask],      0, 0); // prefetch next items
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 0) & cfg->key_count_mask] + 64, 0, 0);
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 1) & cfg->key_count_mask],      0, 0);
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 1) & cfg->key_count_mask] + 64, 0, 0);
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 2) & cfg->key_count_mask],      0, 0);
	__builtin_prefetch(buf + kv_ofs[(iter->node_i[iter->depth] + 2) & cfg->key_count_mask] + 64, 0, 0);
	#endif
	return LITE3_ITER_ITEM;
}


static inline void _lite3_init_impl(unsigned char *buf, size_t ofs, enum lite3_type type, const lite3_node_cfg *cfg)
{
	LITE3_PRINT_DEBUG("INITIALIZE %s\n", type == LITE3_TYPE_OBJECT ? "OBJECT" : "ARRAY");

	struct node *node = (struct node *)(buf + ofs);
	node->gen_type = ((u32)(cfg->id) << LITE3_NODE_CFG_SHIFT) | (u32)(type & LITE3_NODE_TYPE_MASK);
	u32 *size_kc = _lite3_node_size_kc(cfg, node);
	*size_kc = 0x00;
	#ifdef LITE3_ZERO_MEM_EXTRA
		memset(_lite3_node_hashes(cfg, node), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
		memset(_lite3_node_kv_ofs(cfg, node), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
	#endif
	memset(_lite3_node_child_ofs(cfg, node), 0x00, cfg->child_count * sizeof(u32));
}

int lite3_init_obj(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz)
{
	const lite3_node_cfg *cfg = lite3_node_cfg_default();
	if (LITE3_UNLIKELY(bufsz < cfg->node_size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < node size\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_OBJECT, cfg);
	*out_buflen = cfg->node_size;
	return 0;
}

int lite3_init_arr(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz)
{
	const lite3_node_cfg *cfg = lite3_node_cfg_default();
	if (LITE3_UNLIKELY(bufsz < cfg->node_size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < node size\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_ARRAY, cfg);
	*out_buflen = cfg->node_size;
	return 0;
}

int lite3_init_obj_cfg(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, enum lite3_node_cfg_id cfg_id)
{
	const lite3_node_cfg *cfg = lite3_node_cfg_from_id(cfg_id);
	if (LITE3_UNLIKELY(!cfg)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: NODE CFG ID OUT OF RANGE\n");
		errno = EINVAL;
		return -1;
	}
	if (LITE3_UNLIKELY(bufsz < cfg->node_size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < node size\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_OBJECT, cfg);
	*out_buflen = cfg->node_size;
	return 0;
}

int lite3_init_arr_cfg(unsigned char *buf, size_t *restrict out_buflen, size_t bufsz, enum lite3_node_cfg_id cfg_id)
{
	const lite3_node_cfg *cfg = lite3_node_cfg_from_id(cfg_id);
	if (LITE3_UNLIKELY(!cfg)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: NODE CFG ID OUT OF RANGE\n");
		errno = EINVAL;
		return -1;
	}
	if (LITE3_UNLIKELY(bufsz < cfg->node_size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: bufsz < node size\n");
		errno = EINVAL;
		return -1;
	}
	_lite3_init_impl(buf, 0, LITE3_TYPE_ARRAY, cfg);
	*out_buflen = cfg->node_size;
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

	const lite3_node_cfg *cfg = NULL;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;

	struct node *restrict parent = NULL;
	struct node *restrict node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);

	if (LITE3_UNLIKELY(!_lite3_node_aligned(node))) {
		LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
		errno = EBADMSG;
		return -1;
	}

	u32 gen = node->gen_type >> LITE3_NODE_GEN_SHIFT;
	++gen;
	node->gen_type = (node->gen_type & ~LITE3_NODE_GEN_MASK) | (gen << LITE3_NODE_GEN_SHIFT);

	int node_walks = 0;
	while (1) {
		u32 *hashes = _lite3_node_hashes(cfg, node);
		u32 *kv_ofs = _lite3_node_kv_ofs(cfg, node);
		u32 *child_ofs = _lite3_node_child_ofs(cfg, node);
		u32 *size_kc_ptr = _lite3_node_size_kc(cfg, node);
		int key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);

		if (key_count == cfg->key_count_max) {	// node full, need to split

			size_t buflen_aligned = (*inout_buflen + LITE3_NODE_ALIGNMENT_MASK) & ~(size_t)LITE3_NODE_ALIGNMENT_MASK; // next multiple of LITE3_NODE_ALIGNMENT
			size_t new_node_size = parent ? cfg->node_size : 2 * cfg->node_size;

			if (LITE3_UNLIKELY(new_node_size > bufsz || buflen_aligned > bufsz - new_node_size)) {
				LITE3_PRINT_ERROR("NO BUFFER SPACE FOR NODE SPLIT\n");
				errno = ENOBUFS;
				return -1;
			}
			*inout_buflen = buflen_aligned;
			if (!parent) {								// if root split, create new root
				LITE3_PRINT_DEBUG("NEW ROOT\n");
				memcpy(buf + *inout_buflen, node, cfg->node_size);
				node = __builtin_assume_aligned((struct node *)(buf + *inout_buflen), LITE3_NODE_ALIGNMENT);

				if (LITE3_UNLIKELY(!_lite3_node_aligned(node))) {
					LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
					errno = EBADMSG;
					return -1;
				}
				parent = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT);

				if (LITE3_UNLIKELY(!_lite3_node_aligned(parent))) {
					LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
					errno = EBADMSG;
					return -1;
				}
				#ifdef LITE3_ZERO_MEM_EXTRA
					memset(_lite3_node_hashes(cfg, parent), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
					memset(_lite3_node_kv_ofs(cfg, parent), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
				#endif
				memset(_lite3_node_child_ofs(cfg, parent), 0x00, cfg->child_count * sizeof(u32));
				u32 *parent_size_kc_root = _lite3_node_size_kc(cfg, parent);
				u32 parent_size_root = _lite3_node_size(cfg, *parent_size_kc_root);
				*parent_size_kc_root = _lite3_pack_size_kc(parent_size_root, 0);
				_lite3_node_child_ofs(cfg, parent)[0] = (u32)*inout_buflen;			// insert node as child of new root
				*inout_buflen += cfg->node_size;
				hashes = _lite3_node_hashes(cfg, node);
				kv_ofs = _lite3_node_kv_ofs(cfg, node);
				child_ofs = _lite3_node_child_ofs(cfg, node);
				size_kc_ptr = _lite3_node_size_kc(cfg, node);
				key_count = (int)_lite3_node_key_count(cfg, *size_kc_ptr);
			}
			LITE3_PRINT_DEBUG("SPLIT NODE\n");
			u32 *parent_hashes = _lite3_node_hashes(cfg, parent);
			u32 *parent_kv_ofs = _lite3_node_kv_ofs(cfg, parent);
			u32 *parent_child_ofs = _lite3_node_child_ofs(cfg, parent);
			u32 *parent_size_kc_ptr = _lite3_node_size_kc(cfg, parent);
			u32 parent_size = _lite3_node_size(cfg, *parent_size_kc_ptr);
			u32 parent_key_count = _lite3_node_key_count(cfg, *parent_size_kc_ptr);

			// Insert median into parent based on existing parent keys
			int parent_insert_idx = 0;
			u32 median_hash = hashes[cfg->key_count_min];
			while (parent_insert_idx < (int)parent_key_count && parent_hashes[parent_insert_idx] < median_hash)
				parent_insert_idx++;

			for (int j = (int)parent_key_count; j > parent_insert_idx; j--) {					// shift parent array before separator insert
				parent_hashes[j] =        parent_hashes[j - 1];
				parent_kv_ofs[j] =        parent_kv_ofs[j - 1];
				parent_child_ofs[j + 1] = parent_child_ofs[j];
			}
			parent_hashes[parent_insert_idx] = median_hash;		// insert new separator key in parent
			parent_kv_ofs[parent_insert_idx] = kv_ofs[cfg->key_count_min];
			parent_child_ofs[parent_insert_idx + 1] = (u32)*inout_buflen;				// insert sibling as child in parent
			*parent_size_kc_ptr = _lite3_pack_size_kc(parent_size, parent_key_count + 1);
			#ifdef LITE3_ZERO_MEM_EXTRA
				hashes[cfg->key_count_min] = LITE3_ZERO_MEM_32;
				kv_ofs[cfg->key_count_min] = LITE3_ZERO_MEM_32;
			#endif
			struct node *restrict sibling = __builtin_assume_aligned((struct node *)(buf + *inout_buflen), LITE3_NODE_ALIGNMENT);

			if (LITE3_UNLIKELY(!_lite3_node_aligned(sibling))) {
				LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
				errno = EBADMSG;
				return -1;
			}
			#ifdef LITE3_ZERO_MEM_EXTRA
				memset(_lite3_node_hashes(cfg, sibling), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
				memset(_lite3_node_kv_ofs(cfg, sibling), LITE3_ZERO_MEM_8, cfg->key_count_max * sizeof(u32));
			#endif
			u32 root_gen_type = ((struct node *)(buf + ofs))->gen_type;
			sibling->gen_type = root_gen_type & (LITE3_NODE_GEN_MASK | LITE3_NODE_CFG_MASK | LITE3_NODE_TYPE_MASK);
			u32 *sibling_size_kc = _lite3_node_size_kc(cfg, sibling);
			*sibling_size_kc = 	_lite3_pack_size_kc(0, cfg->key_count_min);
			*size_kc_ptr = 	_lite3_pack_size_kc(_lite3_node_size(cfg, *size_kc_ptr), cfg->key_count_min);
			u32 *sibling_child_ofs = _lite3_node_child_ofs(cfg, sibling);
			sibling_child_ofs[0] = child_ofs[cfg->key_count_min + 1];	// take child from node
			                        child_ofs[cfg->key_count_min + 1] = 0x00;
			u32 *sibling_hashes = _lite3_node_hashes(cfg, sibling);
			u32 *sibling_kv_ofs = _lite3_node_kv_ofs(cfg, sibling);
			for (int j = 0; j < cfg->key_count_min; j++) {			// copy half of node's keys to sibling
				int src_idx = j + cfg->key_count_min + 1;
				sibling_hashes[j] =        hashes[src_idx];
				sibling_kv_ofs[j] =        kv_ofs[src_idx];
				sibling_child_ofs[j + 1] = child_ofs[src_idx + 1];
				#ifdef LITE3_ZERO_MEM_EXTRA
					hashes[src_idx] =    LITE3_ZERO_MEM_32;
					kv_ofs[src_idx] =    LITE3_ZERO_MEM_32;
					child_ofs[src_idx + 1] = 0x00000000;
				#endif
			}
			if (key_data.hash >= parent_hashes[parent_insert_idx]) {				// sibling has target key? then we follow
				node = __builtin_assume_aligned(sibling, LITE3_NODE_ALIGNMENT);
			}
			*inout_buflen += cfg->node_size;
			continue;
		}

		int i = 0;
		while (i < key_count && hashes[i] < key_data.hash)
			i++;
		if (i < key_count && hashes[i] == key_data.hash) {			// matching key found, already exists?
			size_t target_ofs = kv_ofs[i];
			size_t key_start_ofs = target_ofs;
			if (key && _verify_key(buf, *inout_buflen, key, (size_t)key_data.size, key_tag_size, &target_ofs, NULL) < 0)
				return -1;
			size_t val_start_ofs = target_ofs;
			if (_verify_val(buf, *inout_buflen, &target_ofs) < 0)
				return -1;
			size_t alignment_mask = _lite3_alignment_mask_for_val_len(val_len);
			if (val_len >= target_ofs - val_start_ofs) {				// value is too large, we must append
				size_t unaligned_val_ofs = *inout_buflen + key_tag_size + (size_t)key_data.size;
				size_t alignment_padding = ((unaligned_val_ofs + alignment_mask) & ~alignment_mask) - unaligned_val_ofs;
				entry_size += alignment_padding;
				if (LITE3_UNLIKELY(entry_size > bufsz || *inout_buflen > bufsz - entry_size)) {
					LITE3_PRINT_ERROR("NO BUFFER SPACE FOR ENTRY INSERTION\n");
					errno = ENOBUFS;
					return -1;
				}
				#ifdef LITE3_ZERO_MEM_DELETED
					memset(buf + kv_ofs[i], LITE3_ZERO_MEM_8, target_ofs - key_start_ofs); // zero out key + value
				#endif
				(void)key_start_ofs;						// silence unused variable warning
				*inout_buflen += alignment_padding;
				kv_ofs[i] = (u32)*inout_buflen;
				goto insert_append;
			}
			#ifdef LITE3_ZERO_MEM_DELETED
				memset(buf + val_start_ofs, LITE3_ZERO_MEM_8, target_ofs - val_start_ofs); // zero out value
			#endif
			*out = (lite3_val *)(buf + val_start_ofs);				// caller overwrites value in place
			return 0;
		}
		if (child_ofs[0]) {							// if children, walk to next node
			size_t next_node_ofs = (size_t)child_ofs[i];

			parent = __builtin_assume_aligned(node, LITE3_NODE_ALIGNMENT);
			node = __builtin_assume_aligned((struct node *)(buf + next_node_ofs), LITE3_NODE_ALIGNMENT);

			if (LITE3_UNLIKELY(!_lite3_node_aligned(node))) {
				LITE3_PRINT_ERROR("NODE OFFSET NOT ALIGNED TO LITE3_NODE_ALIGNMENT\n");
				errno = EBADMSG;
				return -1;
			}
			if (LITE3_UNLIKELY(next_node_ofs > *inout_buflen - cfg->node_size)) {
				LITE3_PRINT_ERROR("NODE WALK OFFSET OUT OF BOUNDS\n");
				errno = EFAULT;
				return -1;
			}
			if (LITE3_UNLIKELY(++node_walks > cfg->tree_height_max)) {
				LITE3_PRINT_ERROR("NODE WALKS EXCEEDED LITE3_TREE_HEIGHT_MAX\n");
				errno = EBADMSG;
				return -1;
			}
		} else {									// insert the kv-pair
			size_t alignment_mask = _lite3_alignment_mask_for_val_len(val_len);
			size_t unaligned_val_ofs = *inout_buflen + key_tag_size + (size_t)key_data.size;
			size_t alignment_padding = ((unaligned_val_ofs + alignment_mask) & ~alignment_mask) - unaligned_val_ofs;
			entry_size += alignment_padding;
			if (LITE3_UNLIKELY(entry_size > bufsz || *inout_buflen > bufsz - entry_size)) {
				LITE3_PRINT_ERROR("NO BUFFER SPACE FOR ENTRY INSERTION\n");
				errno = ENOBUFS;
				return -1;
			}
			for (int j = key_count; j > i; j--) {
				hashes[j] = hashes[j - 1];
				kv_ofs[j] = kv_ofs[j - 1];
			}
			hashes[i] = key_data.hash;
			u32 current_size = _lite3_node_size(cfg, *size_kc_ptr);
			*size_kc_ptr = _lite3_pack_size_kc(current_size, (u32)key_count + 1);	// key_count++
			*inout_buflen += alignment_padding;
			kv_ofs[i] = (u32)*inout_buflen;

			node = __builtin_assume_aligned((struct node *)(buf + ofs), LITE3_NODE_ALIGNMENT); // set node to root
			u32 *root_size_kc = _lite3_node_size_kc(cfg, node);
			u32 size = _lite3_node_size(cfg, *root_size_kc);
			++size;
			*root_size_kc = (_lite3_pack_size_kc(size, _lite3_node_key_count(cfg, *root_size_kc)));
			goto insert_append;
		}
	}
insert_append:
	if (key) {
		size_t key_size_tmp = (key_data.size << LITE3_KEY_TAG_KEY_SIZE_SHIFT) | (key_tag_size - 1);
		memcpy(buf + *inout_buflen, &key_size_tmp, key_tag_size);
		*inout_buflen += key_tag_size;
		memcpy(buf + *inout_buflen, key, (size_t)key_data.size);
		*inout_buflen += (size_t)key_data.size;
	}
	*out = (lite3_val *)(buf + *inout_buflen);
	*inout_buflen += LITE3_VAL_SIZE + val_len;
	return 0;
}

static inline int _lite3_set_node_value(
	unsigned char *buf,
	size_t *restrict inout_buflen,
	size_t ofs,
	size_t bufsz,
	const char *restrict key,
	lite3_key_data key_data,
	enum lite3_type type,
	const lite3_node_cfg *child_cfg,
	size_t *restrict out_ofs)
{
	if (LITE3_UNLIKELY(!child_cfg)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: NODE CFG ID OUT OF RANGE\n");
		errno = EINVAL;
		return -1;
	}
	lite3_val *val;
	size_t val_len = child_cfg->node_size - LITE3_VAL_SIZE;
	int ret;
	if ((ret = lite3_set_impl(buf, inout_buflen, ofs, bufsz, key, key_data, val_len, &val)) < 0)
		return ret;
	size_t init_ofs = (size_t)((u8 *)val - buf);
	if (out_ofs)
		*out_ofs = init_ofs;
	_lite3_init_impl(buf, init_ofs, type, child_cfg);
	return ret;
}

static inline u32 _lite3_array_size(const unsigned char *buf, size_t buflen, size_t ofs, const lite3_node_cfg *cfg)
{
	(void)buflen;
	const struct node *root = (const struct node *)(buf + ofs);
	const u32 *size_kc = _lite3_node_size_kc_c(cfg, root);
	return (u32)_lite3_node_size(cfg, *size_kc);
}

int lite3_set_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, size_t *restrict out_ofs)
{
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, key, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_default(), out_ofs);
}

int lite3_set_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, size_t *restrict out_ofs)
{
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, key, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_default(), out_ofs);
}

int lite3_set_obj_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, key, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_from_id(cfg_id), out_ofs);
}

int lite3_set_arr_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, const char *restrict key, lite3_key_data key_data, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, key, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_from_id(cfg_id), out_ofs);
}

int lite3_arr_append_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_default(), out_ofs);
}

int lite3_arr_append_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_default(), out_ofs);
}

int lite3_arr_set_obj_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_default(), out_ofs);
}

int lite3_arr_set_arr_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_default(), out_ofs);
}

int lite3_arr_append_obj_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_from_id(cfg_id), out_ofs);
}

int lite3_arr_append_arr_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	lite3_key_data key_data = {
		.hash = size,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_from_id(cfg_id), out_ofs);
}

int lite3_arr_set_obj_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_OBJECT, lite3_node_cfg_from_id(cfg_id), out_ofs);
}

int lite3_arr_set_arr_cfg_impl(unsigned char *buf, size_t *restrict inout_buflen, size_t ofs, size_t bufsz, uint32_t index, enum lite3_node_cfg_id cfg_id, size_t *restrict out_ofs)
{
	const lite3_node_cfg *cfg;
	if (_lite3_cfg_for_offset(buf, *inout_buflen, ofs, &cfg, NULL) < 0)
		return -1;
	u32 size = _lite3_array_size(buf, *inout_buflen, ofs, cfg);
	if (LITE3_UNLIKELY(index > size)) {
		LITE3_PRINT_ERROR("INVALID ARGUMENT: ARRAY INDEX %u OUT OF BOUNDS (size == %u)\n", index, size);
		errno = EINVAL;
		return -1;
	}
	lite3_key_data key_data = {
		.hash = index,
		.size = 0,
	};
	return _lite3_set_node_value(buf, inout_buflen, ofs, bufsz, NULL, key_data, LITE3_TYPE_ARRAY, lite3_node_cfg_from_id(cfg_id), out_ofs);
}
