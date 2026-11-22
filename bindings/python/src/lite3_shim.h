/* Lite3 Python binding shim — re-exposes the macro/inline Context API as plain
 * extern symbols so Cython/ctypes can reach it. The tricky key-hash + auto-grow
 * logic stays in C (it lives in the macros), never reimplemented in Python. */
#ifndef LITE3_SHIM_H
#define LITE3_SHIM_H
#include <stddef.h>
#include <stdint.h>

typedef struct lite3_ctx lite3_ctx;        /* opaque to Cython */
typedef struct l3_iter l3_iter;            /* opaque heap iterator */

/* lifecycle */
lite3_ctx *l3_create(size_t bufsz);
void       l3_destroy(lite3_ctx *c);
int        l3_init_obj(lite3_ctx *c);
int        l3_init_arr(lite3_ctx *c);

/* ingest a received Lite3 buffer (copies into the context) */
int l3_import_from_buf(lite3_ctx *c, const uint8_t *buf, size_t buflen);

/* JSON round-trip (reuses in-repo yyjson) */
int   l3_from_json(lite3_ctx *c, const char *json, size_t len);
char *l3_to_json(lite3_ctx *c, size_t ofs);   /* malloc'd NUL-string; l3_free it */
void  l3_free(void *p);

/* zero-copy buffer access (buffer protocol) */
const uint8_t *l3_buf(lite3_ctx *c);
size_t         l3_buflen(lite3_ctx *c);

/* structure introspection */
int  l3_root_type(lite3_ctx *c);                       /* lite3_type or INVALID */
long l3_count(lite3_ctx *c, size_t ofs);               /* entries/elems, -1 err */

/* by-key reads (objects) */
int l3_get_type(lite3_ctx *c, size_t ofs, const char *key);
int l3_get_bool(lite3_ctx *c, size_t ofs, const char *key, int *out);
int l3_get_i64 (lite3_ctx *c, size_t ofs, const char *key, int64_t *out);
int l3_get_f64 (lite3_ctx *c, size_t ofs, const char *key, double *out);
int l3_get_str (lite3_ctx *c, size_t ofs, const char *key, const char **ptr, size_t *len);
int l3_get_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t **ptr, size_t *len);
int l3_get_obj (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs);
int l3_get_arr (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs);

/* by-index reads (arrays) */
int l3_arr_get_type(lite3_ctx *c, size_t ofs, uint32_t i);
int l3_arr_get_bool(lite3_ctx *c, size_t ofs, uint32_t i, int *out);
int l3_arr_get_i64 (lite3_ctx *c, size_t ofs, uint32_t i, int64_t *out);
int l3_arr_get_f64 (lite3_ctx *c, size_t ofs, uint32_t i, double *out);
int l3_arr_get_str (lite3_ctx *c, size_t ofs, uint32_t i, const char **ptr, size_t *len);
int l3_arr_get_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t **ptr, size_t *len);
int l3_arr_get_obj (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs);
int l3_arr_get_arr (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs);

/* by-key writes (objects). str/bytes take explicit length. obj/arr return the
 * new container's offset so children can be written into it. */
int l3_set_null (lite3_ctx *c, size_t ofs, const char *key);
int l3_set_bool (lite3_ctx *c, size_t ofs, const char *key, int v);
int l3_set_i64  (lite3_ctx *c, size_t ofs, const char *key, int64_t v);
int l3_set_f64  (lite3_ctx *c, size_t ofs, const char *key, double v);
int l3_set_str  (lite3_ctx *c, size_t ofs, const char *key, const char *s, size_t n);
int l3_set_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t *b, size_t n);
int l3_set_obj  (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs);
int l3_set_arr  (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs);

/* array set-by-index (overwrite; index must be < length) */
int l3_arr_set_null (lite3_ctx *c, size_t ofs, uint32_t i);
int l3_arr_set_bool (lite3_ctx *c, size_t ofs, uint32_t i, int v);
int l3_arr_set_i64  (lite3_ctx *c, size_t ofs, uint32_t i, int64_t v);
int l3_arr_set_f64  (lite3_ctx *c, size_t ofs, uint32_t i, double v);
int l3_arr_set_str  (lite3_ctx *c, size_t ofs, uint32_t i, const char *s, size_t n);
int l3_arr_set_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t *b, size_t n);
int l3_arr_set_obj  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs);
int l3_arr_set_arr  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs);

/* array append */
int l3_app_null (lite3_ctx *c, size_t ofs);
int l3_app_bool (lite3_ctx *c, size_t ofs, int v);
int l3_app_i64  (lite3_ctx *c, size_t ofs, int64_t v);
int l3_app_f64  (lite3_ctx *c, size_t ofs, double v);
int l3_app_str  (lite3_ctx *c, size_t ofs, const char *s, size_t n);
int l3_app_bytes(lite3_ctx *c, size_t ofs, const uint8_t *b, size_t n);
int l3_app_obj  (lite3_ctx *c, size_t ofs, size_t *out_ofs);
int l3_app_arr  (lite3_ctx *c, size_t ofs, size_t *out_ofs);

/* object key enumeration (heap iterator) */
l3_iter *l3_iter_new(lite3_ctx *c, size_t ofs);
int      l3_iter_next(l3_iter *it, const char **key_ptr, size_t *key_len);  /* 1 item, 0 done, <0 err */
void     l3_iter_free(l3_iter *it);

#endif
