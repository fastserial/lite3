#include <assert.h>

#include "lite3.h"

static void expect_count(unsigned char *buf, size_t len, size_t ofs, uint32_t expected)
{
	uint32_t count;
	assert(lite3_count(buf, len, ofs, &count) == 0);
	assert(count == expected);
}

static void expect_i64(const unsigned char *buf, size_t len, size_t ofs, uint32_t index, int64_t expected)
{
	int64_t value;
	assert(lite3_arr_get_i64(buf, len, ofs, index, &value) == 0);
	assert(value == expected);
}

static void mixed_and_nested(void)
{
	_Alignas(LITE3_NODE_ALIGNMENT) unsigned char buf[131072] = {0};
	size_t len, nested, object, ofs;
	lite3_iter iter;
	assert(lite3_init_arr(buf, &len, sizeof(buf)) == 0);
	expect_count(buf, len, 0, 0);
	assert(lite3_iter_create(buf, len, 0, &iter) == 0);
	assert(lite3_iter_next(buf, len, &iter, NULL, NULL) == LITE3_ITER_DONE);

	assert(lite3_arr_append_null(buf, &len, 0, sizeof(buf)) == 0);
	assert(lite3_arr_append_bool(buf, &len, 0, sizeof(buf), true) == 0);
	assert(lite3_arr_append_str(buf, &len, 0, sizeof(buf), "text") == 0);
	assert(lite3_arr_append_arr(buf, &len, 0, sizeof(buf), &nested) == 0);
	assert(lite3_arr_append_obj(buf, &len, 0, sizeof(buf), &object) == 0);
	assert(lite3_set_i64(buf, &len, object, sizeof(buf), "number", 42) == 0);

	for (uint32_t i = 0; i < 1024; ++i) {
		assert(lite3_arr_append_i64(buf, &len, 0, sizeof(buf), i) == 0);
		assert(lite3_arr_append_i64(buf, &len, nested, sizeof(buf), -(int64_t)i) == 0);
	}

	expect_count(buf, len, 0, 1029);
	expect_count(buf, len, nested, 1024);
	assert(lite3_arr_get_type(buf, len, 0, 0) == LITE3_TYPE_NULL);
	bool flag;
	assert(lite3_arr_get_bool(buf, len, 0, 1, &flag) == 0 && flag);
	lite3_str str;
	assert(lite3_arr_get_str(buf, len, 0, 2, &str) == 0);
	assert(strcmp(LITE3_STR(buf, str), "text") == 0);
	assert(lite3_arr_get_arr(buf, len, 0, 3, &ofs) == 0 && ofs == nested);
	assert(lite3_arr_get_obj(buf, len, 0, 4, &ofs) == 0 && ofs == object);
	int64_t number;
	assert(lite3_get_i64(buf, len, object, "number", &number) == 0 && number == 42);

	for (uint32_t i = 0; i < 1024; ++i) {
		expect_i64(buf, len, 0, i + 5, i);
		expect_i64(buf, len, nested, i, -(int64_t)i);
	}

	assert(lite3_arr_set_i64(buf, &len, nested, sizeof(buf), 31, 99) == 0);
	assert(lite3_arr_set_i64(buf, &len, nested, sizeof(buf), 1025, 0) < 0 && errno == EINVAL);
	assert(lite3_iter_create(buf, len, 0, &iter) == 0);
	assert(lite3_arr_set_i64(buf, &len, nested, sizeof(buf), 1024, 100) == 0);
	assert(lite3_iter_next(buf, len, &iter, NULL, NULL) < 0 && errno == EINVAL);
	expect_count(buf, len, nested, 1025);

	assert(lite3_iter_create(buf, len, nested, &iter) == 0);
	for (uint32_t i = 0; i < 1024; ++i) {
		assert(lite3_iter_next(buf, len, &iter, NULL, &ofs) == LITE3_ITER_ITEM);
		assert(lite3_val_i64((lite3_val *)(buf + ofs)) == (i == 31 ? 99 : -(int64_t)i));
	}
	assert(lite3_iter_next(buf, len, &iter, NULL, &ofs) == LITE3_ITER_ITEM);
	assert(lite3_val_i64((lite3_val *)(buf + ofs)) == 100);
	assert(lite3_iter_next(buf, len, &iter, NULL, NULL) == LITE3_ITER_DONE);
}

static void retry_after_root_split(void)
{
	_Alignas(LITE3_NODE_ALIGNMENT) unsigned char buf[8 * LITE3_NODE_SIZE] = {0};
	size_t len;
	uint32_t index = 0;
	assert(lite3_init_arr(buf, &len, sizeof(buf)) == 0);

	// Leave less space than a new node needs, so a full root stops the append.
	const size_t capacity = 2 * LITE3_NODE_SIZE - 1;
	while (lite3_arr_append_i64(buf, &len, 0, capacity, index) == 0)
		++index;
	assert(errno == ENOBUFS);
	expect_count(buf, len, 0, index);

	// Allow the split to finish, but leave no space for the value.
	const size_t split_end = ((len + LITE3_NODE_ALIGNMENT_MASK) & ~LITE3_NODE_ALIGNMENT_MASK)
		+ 2 * LITE3_NODE_SIZE;
	assert(lite3_arr_append_i64(buf, &len, 0, split_end, index) < 0 && errno == ENOBUFS);
	assert(len == split_end);
	expect_count(buf, len, 0, index);
	for (uint32_t i = 0; i < index; ++i)
		expect_i64(buf, len, 0, i, i);

	assert(lite3_arr_append_i64(buf, &len, 0, sizeof(buf), index) == 0);
	expect_count(buf, len, 0, index + 1);
	for (uint32_t i = 0; i <= index; ++i)
		expect_i64(buf, len, 0, i, i);
}

int main(void)
{
	mixed_and_nested();
	retry_after_root_split();
	return 0;
}
