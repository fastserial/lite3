/* Exercise short, full, split and exhausted nodes with either prefetch configuration.
 * Compile the library with -DLITE3_PREFETCHING and run under UBSan to check
 * speculative lookahead; also exercise the default hint-free configuration. */
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lite3.h"

static void check_container(unsigned count, int object)
{
    unsigned char __attribute__((aligned(LITE3_NODE_ALIGNMENT))) storage[32768];
    size_t length = 0;
    assert((object ? lite3_init_obj(storage, &length, sizeof(storage))
                   : lite3_init_arr(storage, &length, sizeof(storage))) == 0);
    for (unsigned i = 0; i < count; i++) {
        char key[32];
        snprintf(key, sizeof(key), "field_%u", i);
        int result = object
            ? lite3_set_i64(storage, &length, 0, sizeof(storage), key, i)
            : lite3_arr_append_i64(storage, &length, 0, sizeof(storage), i);
        assert(result == 0);
    }
    /* Keep only the message allocation, without spare capacity at the end. */
    unsigned char *buf = malloc(length);
    assert(buf);
    memcpy(buf, storage, length);
    lite3_iter iter;
    assert(lite3_iter_create(buf, length, 0, &iter) == 0);
    size_t offset;
    unsigned seen = 0;
    int64_t sum = 0;
    int result;
    while ((result = lite3_iter_next(buf, length, &iter, NULL, &offset)) == LITE3_ITER_ITEM) {
        assert(buf[offset] == LITE3_TYPE_I64);
        sum += lite3_val_i64((lite3_val *)(buf + offset));
        seen++;
    }
    assert(result == LITE3_ITER_DONE);
    assert(seen == count);
    assert(sum == (int64_t)count * ((int64_t)count - 1) / 2);
    assert(lite3_iter_next(buf, length, &iter, NULL, &offset) == LITE3_ITER_DONE);
    free(buf);
}

static void check_invalid_child(void)
{
    _Alignas(LITE3_NODE_ALIGNMENT) unsigned char buf[8192];
    size_t length = 0;
    assert(lite3_init_arr(buf, &length, sizeof(buf)) == 0);
    for (int i = 0; i < 10; i++)
        assert(lite3_arr_append_i64(buf, &length, 0, sizeof(buf), i) == 0);

    /* The last eight 32-bit words in a node hold child offsets. */
    size_t child_slot = LITE3_NODE_SIZE - 8 * sizeof(uint32_t);
    uint32_t child;
    memcpy(&child, buf + child_slot, sizeof(child));
    assert(child != 0);
    lite3_iter iter;
    assert(lite3_iter_create(buf, length, 0, &iter) == 0);
    uint32_t bad = child + 1;
    memcpy(buf + child_slot, &bad, sizeof(bad));
    assert(lite3_iter_create(buf, length, 0, &iter) < 0 && errno == EBADMSG);

    bad = (uint32_t)length;
    memcpy(buf + child_slot, &bad, sizeof(bad));
    assert(lite3_iter_create(buf, length, 0, &iter) < 0 && errno == EFAULT);
}

int main(void)
{
    check_invalid_child();
    for (unsigned count = 0; count <= 128; count++) {
        check_container(count, 0);
        check_container(count, 1);
    }
    return 0;
}
