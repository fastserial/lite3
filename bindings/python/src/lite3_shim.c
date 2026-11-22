#include <stdlib.h>
#include "lite3_context_api.h"
#include "lite3_shim.h"

lite3_ctx *l3_create(size_t bufsz) { return lite3_ctx_create_with_size(bufsz); }
void       l3_destroy(lite3_ctx *c) { lite3_ctx_destroy(c); }
int        l3_init_obj(lite3_ctx *c) { return lite3_ctx_init_obj(c); }
int        l3_init_arr(lite3_ctx *c) { return lite3_ctx_init_arr(c); }

int   l3_import_from_buf(lite3_ctx *c, const uint8_t *buf, size_t buflen) { return lite3_ctx_import_from_buf(c, buf, buflen); }
int   l3_from_json(lite3_ctx *c, const char *json, size_t len) { return lite3_ctx_json_dec(c, json, len); }
char *l3_to_json(lite3_ctx *c, size_t ofs) { size_t n; return lite3_ctx_json_enc(c, ofs, &n); }
void  l3_free(void *p) { free(p); }

const uint8_t *l3_buf(lite3_ctx *c) { return c->buf; }
size_t         l3_buflen(lite3_ctx *c) { return c->buflen; }

int  l3_root_type(lite3_ctx *c) { return (int)lite3_ctx_get_root_type(c); }

long l3_count(lite3_ctx *c, size_t ofs) {
    uint32_t n;
    if (lite3_ctx_count(c, ofs, &n) < 0) return -1;
    return (long)n;
}

/* by-key */
int l3_get_type(lite3_ctx *c, size_t ofs, const char *key) { return (int)lite3_ctx_get_type(c, ofs, key); }
int l3_get_bool(lite3_ctx *c, size_t ofs, const char *key, int *out) {
    bool b; int r = lite3_ctx_get_bool(c, ofs, key, &b); if (r == 0) *out = b; return r;
}
int l3_get_i64(lite3_ctx *c, size_t ofs, const char *key, int64_t *out) { return lite3_ctx_get_i64(c, ofs, key, out); }
int l3_get_f64(lite3_ctx *c, size_t ofs, const char *key, double *out)  { return lite3_ctx_get_f64(c, ofs, key, out); }
int l3_get_str(lite3_ctx *c, size_t ofs, const char *key, const char **ptr, size_t *len) {
    lite3_str s; int r = lite3_ctx_get_str(c, ofs, key, &s); if (r < 0) return r;
    *ptr = LITE3_STR(c->buf, s); *len = s.len; return r;
}
int l3_get_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t **ptr, size_t *len) {
    lite3_bytes b; int r = lite3_ctx_get_bytes(c, ofs, key, &b); if (r < 0) return r;
    *ptr = LITE3_BYTES(c->buf, b); *len = b.len; return r;
}
int l3_get_obj(lite3_ctx *c, size_t ofs, const char *key, size_t *o) { return lite3_ctx_get_obj(c, ofs, key, o); }
int l3_get_arr(lite3_ctx *c, size_t ofs, const char *key, size_t *o) { return lite3_ctx_get_arr(c, ofs, key, o); }

/* by-index */
int l3_arr_get_type(lite3_ctx *c, size_t ofs, uint32_t i) { return (int)lite3_ctx_arr_get_type(c, ofs, i); }
int l3_arr_get_bool(lite3_ctx *c, size_t ofs, uint32_t i, int *out) {
    bool b; int r = lite3_ctx_arr_get_bool(c, ofs, i, &b); if (r == 0) *out = b; return r;
}
int l3_arr_get_i64(lite3_ctx *c, size_t ofs, uint32_t i, int64_t *out) { return lite3_ctx_arr_get_i64(c, ofs, i, out); }
int l3_arr_get_f64(lite3_ctx *c, size_t ofs, uint32_t i, double *out)  { return lite3_ctx_arr_get_f64(c, ofs, i, out); }
int l3_arr_get_str(lite3_ctx *c, size_t ofs, uint32_t i, const char **ptr, size_t *len) {
    lite3_str s; int r = lite3_ctx_arr_get_str(c, ofs, i, &s); if (r < 0) return r;
    *ptr = LITE3_STR(c->buf, s); *len = s.len; return r;
}
int l3_arr_get_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t **ptr, size_t *len) {
    lite3_bytes b; int r = lite3_ctx_arr_get_bytes(c, ofs, i, &b); if (r < 0) return r;
    *ptr = LITE3_BYTES(c->buf, b); *len = b.len; return r;
}
int l3_arr_get_obj(lite3_ctx *c, size_t ofs, uint32_t i, size_t *o) { return lite3_ctx_arr_get_obj(c, ofs, i, o); }
int l3_arr_get_arr(lite3_ctx *c, size_t ofs, uint32_t i, size_t *o) { return lite3_ctx_arr_get_arr(c, ofs, i, o); }

/* --- writes --- */
int l3_set_null (lite3_ctx *c, size_t ofs, const char *key) { return lite3_ctx_set_null(c, ofs, key); }
int l3_set_bool (lite3_ctx *c, size_t ofs, const char *key, int v) { return lite3_ctx_set_bool(c, ofs, key, v != 0); }
int l3_set_i64  (lite3_ctx *c, size_t ofs, const char *key, int64_t v) { return lite3_ctx_set_i64(c, ofs, key, v); }
int l3_set_f64  (lite3_ctx *c, size_t ofs, const char *key, double v) { return lite3_ctx_set_f64(c, ofs, key, v); }
int l3_set_str  (lite3_ctx *c, size_t ofs, const char *key, const char *s, size_t n) { return lite3_ctx_set_str_n(c, ofs, key, s, n); }
int l3_set_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t *b, size_t n) { return lite3_ctx_set_bytes(c, ofs, key, b, n); }
int l3_set_obj  (lite3_ctx *c, size_t ofs, const char *key, size_t *o) { return lite3_ctx_set_obj(c, ofs, key, o); }
int l3_set_arr  (lite3_ctx *c, size_t ofs, const char *key, size_t *o) { return lite3_ctx_set_arr(c, ofs, key, o); }

int l3_arr_set_null (lite3_ctx *c, size_t ofs, uint32_t i) { return lite3_ctx_arr_set_null(c, ofs, i); }
int l3_arr_set_bool (lite3_ctx *c, size_t ofs, uint32_t i, int v) { return lite3_ctx_arr_set_bool(c, ofs, i, v != 0); }
int l3_arr_set_i64  (lite3_ctx *c, size_t ofs, uint32_t i, int64_t v) { return lite3_ctx_arr_set_i64(c, ofs, i, v); }
int l3_arr_set_f64  (lite3_ctx *c, size_t ofs, uint32_t i, double v) { return lite3_ctx_arr_set_f64(c, ofs, i, v); }
int l3_arr_set_str  (lite3_ctx *c, size_t ofs, uint32_t i, const char *s, size_t n) { return lite3_ctx_arr_set_str_n(c, ofs, i, s, n); }
int l3_arr_set_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t *b, size_t n) { return lite3_ctx_arr_set_bytes(c, ofs, i, b, n); }
int l3_arr_set_obj  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *o) { return lite3_ctx_arr_set_obj(c, ofs, i, o); }
int l3_arr_set_arr  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *o) { return lite3_ctx_arr_set_arr(c, ofs, i, o); }

int l3_app_null (lite3_ctx *c, size_t ofs) { return lite3_ctx_arr_append_null(c, ofs); }
int l3_app_bool (lite3_ctx *c, size_t ofs, int v) { return lite3_ctx_arr_append_bool(c, ofs, v != 0); }
int l3_app_i64  (lite3_ctx *c, size_t ofs, int64_t v) { return lite3_ctx_arr_append_i64(c, ofs, v); }
int l3_app_f64  (lite3_ctx *c, size_t ofs, double v) { return lite3_ctx_arr_append_f64(c, ofs, v); }
int l3_app_str  (lite3_ctx *c, size_t ofs, const char *s, size_t n) { return lite3_ctx_arr_append_str_n(c, ofs, s, n); }
int l3_app_bytes(lite3_ctx *c, size_t ofs, const uint8_t *b, size_t n) { return lite3_ctx_arr_append_bytes(c, ofs, b, n); }
int l3_app_obj  (lite3_ctx *c, size_t ofs, size_t *o) { return lite3_ctx_arr_append_obj(c, ofs, o); }
int l3_app_arr  (lite3_ctx *c, size_t ofs, size_t *o) { return lite3_ctx_arr_append_arr(c, ofs, o); }

/* heap iterator: keeps ctx + lite3_iter together so Cython treats it opaquely */
struct l3_iter { lite3_ctx *c; lite3_iter it; };

l3_iter *l3_iter_new(lite3_ctx *c, size_t ofs) {
    l3_iter *h = malloc(sizeof *h);
    if (!h) return NULL;
    h->c = c;
    if (lite3_ctx_iter_create(c, ofs, &h->it) < 0) { free(h); return NULL; }
    return h;
}
int l3_iter_next(l3_iter *h, const char **key_ptr, size_t *key_len) {
    lite3_str key;
    int r = lite3_ctx_iter_next(h->c, &h->it, &key, NULL);
    if (r == LITE3_ITER_ITEM) { *key_ptr = LITE3_STR(h->c->buf, key); *key_len = key.len; }
    return r;
}
void l3_iter_free(l3_iter *h) { free(h); }
