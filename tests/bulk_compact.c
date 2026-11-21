#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lite3.h"
#include "lite3_context_api.h"

static int64_t values[65536];

static void check_array(unsigned char *buf, size_t len, uint32_t count)
{
    uint32_t n;
    assert(lite3_count(buf, len, 0, &n) == 0 && n == count);
    lite3_iter iter;
    assert(lite3_iter_create(buf, len, 0, &iter) == 0);
    for (uint32_t i = 0; i < count; i++) {
        size_t ofs;
        assert(lite3_iter_next(buf, len, &iter, NULL, &ofs) == LITE3_ITER_ITEM);
        assert(buf[ofs] == LITE3_TYPE_I64);
        assert(lite3_val_i64((lite3_val *)(buf + ofs)) == values[i]);
    }
    assert(lite3_iter_next(buf, len, &iter, NULL, NULL) == LITE3_ITER_DONE);
}

static void check_bulk(unsigned char *buf, size_t capacity, uint32_t count)
{
    size_t len = 0;
    assert(lite3_init_i64_array(buf, &len, capacity, values, count) == 0);
    check_array(buf, len, count);

    size_t used = len;
    unsigned char first = buf[0];
    assert(lite3_init_i64_array(buf, &len, used - 1, values, count) < 0);
    assert(errno == ENOBUFS && len == used && buf[0] == first);
    assert(lite3_init_i64_array(buf, &len, used, values, count) == 0);
    if (count)
        assert(lite3_arr_set_i64(buf, &len, 0, capacity, count / 2, -88) == 0);
    for (uint32_t i = 0; i < 16; i++)
        assert(lite3_arr_append_i64(buf, &len, 0, capacity, -99) == 0);
    uint32_t n;
    assert(lite3_count(buf, len, 0, &n) == 0 && n == count + 16);
    int64_t v;
    assert(lite3_arr_get_i64(buf, len, 0, count + 15, &v) == 0 && v == -99);
    if (count)
        assert(lite3_arr_get_i64(buf, len, 0, count / 2, &v) == 0 && v == -88);
}

static void check_compact(unsigned char *buf, unsigned char *dst, size_t capacity)
{
    size_t len, nested;
    assert(lite3_init_obj(buf, &len, capacity) == 0);
    assert(lite3_set_arr_impl(buf, &len, 0, capacity, "list",
                              lite3_get_key_data("list"), &nested) == 0);
    for (int i = 0; i < 100; i++)
        assert(lite3_arr_append_i64(buf, &len, nested, capacity, -i) == 0);
    assert(lite3_arr_append_f64(buf, &len, nested, capacity, -0.0) == 0);
    unsigned char raw[] = {0, 255, 0, 1};
    assert(lite3_set_bytes(buf, &len, 0, capacity, "raw", raw, sizeof(raw)) == 0);
    char text[513];
    memset(text, 'a', sizeof(text));
    text[512] = '\0';
    for (int i = 0; i < 40; i++) {
        assert(lite3_set_str(buf, &len, 0, capacity, "text", ".") == 0);
        assert(lite3_set_str(buf, &len, 0, capacity, "text", text) == 0);
    }
    unsigned char *saved = malloc(len);
    assert(saved);
    memcpy(saved, buf, len);
    size_t out = 0;
    assert(lite3_compact_copy(buf, len, dst, &out, LITE3_NODE_SIZE) < 0);
    assert(errno == ENOBUFS && out == 0 && memcmp(saved, buf, len) == 0);
    assert(lite3_compact_copy(buf, len, dst, &out, capacity) == 0);
    assert(out < len / 2 && memcmp(saved, buf, len) == 0);
    free(saved);

    size_t list;
    double minus_zero;
    assert(lite3_get_arr(dst, out, 0, "list", &list) == 0);
    assert(lite3_arr_get_f64(dst, out, list, 100, &minus_zero) == 0);
    assert(minus_zero == 0 && signbit(minus_zero));
    lite3_bytes bytes;
    assert(lite3_get_bytes(dst, out, 0, "raw", &bytes) == 0);
    assert(bytes.len == sizeof(raw) && memcmp(bytes.ptr, raw, sizeof(raw)) == 0);

    size_t size1, size2;
    char *a = lite3_json_enc(buf, len, 0, &size1);
    char *b = lite3_json_enc(dst, out, 0, &size2);
    assert(a && b && size1 == size2 && memcmp(a, b, size1) == 0);
    free(a);
    free(b);
}

static void check_split_padding(void)
{
    _Alignas(LITE3_NODE_ALIGNMENT) unsigned char buf[1024];
    memset(buf, 0xA5, sizeof(buf));
    size_t len;
    assert(lite3_init_arr(buf, &len, sizeof(buf)) == 0);
    for (int i = 0; i < 7; i++)
        assert(lite3_arr_append_i64(buf, &len, 0, sizeof(buf), i) == 0);
    size_t before = len;
    assert(lite3_arr_append_i64(buf, &len, 0, sizeof(buf), 7) == 0);
    size_t aligned = (before + LITE3_NODE_ALIGNMENT_MASK) & ~LITE3_NODE_ALIGNMENT_MASK;
    assert(aligned > before);
    for (size_t i = before; i < aligned; i++)
        assert(buf[i] == 0);
}

static void check_growth(void)
{
    char json[8192];
    memset(json, 'x', sizeof(json));
    memcpy(json, "{\"text\":\"", 9);
    memcpy(json + sizeof(json) - 3, "\"}", 3);
    lite3_ctx *ctx = lite3_ctx_create_with_size(1024);
    assert(ctx);
    size_t capacity = ctx->bufsz;
    assert(lite3_ctx_json_dec(ctx, json, sizeof(json) - 1) == 0);
    assert(ctx->bufsz > capacity);
    lite3_str val;
    assert(lite3_get_str(ctx->buf, ctx->buflen, 0, "text", &val) == 0);
    assert(val.len == sizeof(json) - 12);
    size_t used = ctx->buflen;
    assert(lite3_ctx_json_dec(ctx, "{", 1) < 0 && errno == EINVAL);
    assert(ctx->buflen == used);
    assert(lite3_ctx_json_dec(ctx, "{}", 2) == 0);
    lite3_ctx_destroy(ctx);
}

int main(void)
{
    for (uint32_t i = 0; i < 65536; i++)
        values[i] = (int64_t)i - 40000;
    values[0] = INT64_MIN;
    values[65535] = INT64_MAX;
    const size_t cap = 4 * 1024 * 1024;
    unsigned char *buf = malloc(cap);
    unsigned char *dst = malloc(cap);
    assert(buf && dst);
    for (uint32_t i = 0; i < 600; i++)
        check_bulk(buf, cap, i);
    check_bulk(buf, cap, 4095);
    check_bulk(buf, cap, 4096);
    check_bulk(buf, cap, 65536);
    check_compact(buf, dst, cap);
    check_growth();
    check_split_padding();
    free(buf);
    free(dst);
    return 0;
}
