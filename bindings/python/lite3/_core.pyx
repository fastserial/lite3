# cython: language_level=3
"""Cython core for the Lite³ Python binding.

Design: lazy zero-copy reads. `Lite3` owns the C context; `_ObjView`/`_ArrView`
read fields out of the serialized buffer on demand (proxies, not hydrated dicts).
`to_dict()`/`from_dict()` and `to_json()`/`from_json()` are the explicit slow
paths — they round-trip through the in-repo yyjson C codec, never reimplemented.
"""
import json as _json
from libc.stdint cimport int64_t, uint8_t, uint32_t
from libc.stddef cimport size_t

cdef extern from "lite3_shim.h":
    ctypedef struct lite3_ctx
    ctypedef struct l3_iter
    lite3_ctx *l3_create(size_t bufsz)
    void       l3_destroy(lite3_ctx *c)
    int        l3_init_obj(lite3_ctx *c)
    int        l3_init_arr(lite3_ctx *c)
    int        l3_import_from_buf(lite3_ctx *c, const uint8_t *buf, size_t buflen)
    int        l3_from_json(lite3_ctx *c, const char *json, size_t length)
    char      *l3_to_json(lite3_ctx *c, size_t ofs)
    void       l3_free(void *p)
    const uint8_t *l3_buf(lite3_ctx *c)
    size_t     l3_buflen(lite3_ctx *c)
    int        l3_root_type(lite3_ctx *c)
    long       l3_count(lite3_ctx *c, size_t ofs)
    int  l3_get_type(lite3_ctx *c, size_t ofs, const char *key)
    int  l3_get_bool(lite3_ctx *c, size_t ofs, const char *key, int *out)
    int  l3_get_i64 (lite3_ctx *c, size_t ofs, const char *key, int64_t *out)
    int  l3_get_f64 (lite3_ctx *c, size_t ofs, const char *key, double *out)
    int  l3_get_str (lite3_ctx *c, size_t ofs, const char *key, const char **ptr, size_t *length)
    int  l3_get_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t **ptr, size_t *length)
    int  l3_get_obj (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs)
    int  l3_get_arr (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs)
    int  l3_arr_get_type(lite3_ctx *c, size_t ofs, uint32_t i)
    int  l3_arr_get_bool(lite3_ctx *c, size_t ofs, uint32_t i, int *out)
    int  l3_arr_get_i64 (lite3_ctx *c, size_t ofs, uint32_t i, int64_t *out)
    int  l3_arr_get_f64 (lite3_ctx *c, size_t ofs, uint32_t i, double *out)
    int  l3_arr_get_str (lite3_ctx *c, size_t ofs, uint32_t i, const char **ptr, size_t *length)
    int  l3_arr_get_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t **ptr, size_t *length)
    int  l3_arr_get_obj (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs)
    int  l3_arr_get_arr (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs)
    int  l3_set_null (lite3_ctx *c, size_t ofs, const char *key)
    int  l3_set_bool (lite3_ctx *c, size_t ofs, const char *key, int v)
    int  l3_set_i64  (lite3_ctx *c, size_t ofs, const char *key, int64_t v)
    int  l3_set_f64  (lite3_ctx *c, size_t ofs, const char *key, double v)
    int  l3_set_str  (lite3_ctx *c, size_t ofs, const char *key, const char *s, size_t n)
    int  l3_set_bytes(lite3_ctx *c, size_t ofs, const char *key, const uint8_t *b, size_t n)
    int  l3_set_obj  (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs)
    int  l3_set_arr  (lite3_ctx *c, size_t ofs, const char *key, size_t *out_ofs)
    int  l3_app_null (lite3_ctx *c, size_t ofs)
    int  l3_app_bool (lite3_ctx *c, size_t ofs, int v)
    int  l3_app_i64  (lite3_ctx *c, size_t ofs, int64_t v)
    int  l3_app_f64  (lite3_ctx *c, size_t ofs, double v)
    int  l3_app_str  (lite3_ctx *c, size_t ofs, const char *s, size_t n)
    int  l3_app_bytes(lite3_ctx *c, size_t ofs, const uint8_t *b, size_t n)
    int  l3_app_obj  (lite3_ctx *c, size_t ofs, size_t *out_ofs)
    int  l3_app_arr  (lite3_ctx *c, size_t ofs, size_t *out_ofs)
    int  l3_arr_set_null (lite3_ctx *c, size_t ofs, uint32_t i)
    int  l3_arr_set_bool (lite3_ctx *c, size_t ofs, uint32_t i, int v)
    int  l3_arr_set_i64  (lite3_ctx *c, size_t ofs, uint32_t i, int64_t v)
    int  l3_arr_set_f64  (lite3_ctx *c, size_t ofs, uint32_t i, double v)
    int  l3_arr_set_str  (lite3_ctx *c, size_t ofs, uint32_t i, const char *s, size_t n)
    int  l3_arr_set_bytes(lite3_ctx *c, size_t ofs, uint32_t i, const uint8_t *b, size_t n)
    int  l3_arr_set_obj  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs)
    int  l3_arr_set_arr  (lite3_ctx *c, size_t ofs, uint32_t i, size_t *out_ofs)
    l3_iter *l3_iter_new(lite3_ctx *c, size_t ofs)
    int      l3_iter_next(l3_iter *it, const char **key_ptr, size_t *key_len)
    void     l3_iter_free(l3_iter *it)

# lite3_type enum
cdef enum:
    T_NULL = 0
    T_BOOL = 1
    T_I64 = 2
    T_F64 = 3
    T_BYTES = 4
    T_STRING = 5
    T_OBJECT = 6
    T_ARRAY = 7
    T_INVALID = 8


cdef class Lite3:
    """Owns a Lite³ context. Index it like the root object/array for lazy reads."""
    cdef lite3_ctx *c

    def __cinit__(self):
        self.c = l3_create(1024)
        if self.c is NULL:
            raise MemoryError("lite3 context allocation failed")
        l3_init_obj(self.c)   # empty object by default; from_*/new_array overwrite

    def __dealloc__(self):
        if self.c is not NULL:
            l3_destroy(self.c)

    @classmethod
    def from_json(cls, data):
        cdef Lite3 self = cls()
        b = data.encode() if isinstance(data, str) else bytes(data)
        if l3_from_json(self.c, b, len(b)) < 0:
            raise ValueError("invalid JSON for Lite3")
        return self

    @classmethod
    def from_dict(cls, d):
        # Slow path: reuse the tested C JSON decoder rather than hand-building.
        return cls.from_json(_json.dumps(d))

    @classmethod
    def from_bytes(cls, data):
        """Ingest a received Lite³ buffer (copies it into a fresh context)."""
        cdef Lite3 self = cls()
        cdef const uint8_t[::1] mv = bytes(data)
        if mv.shape[0] == 0:
            raise ValueError("invalid Lite3 buffer")
        if l3_import_from_buf(self.c, &mv[0], mv.shape[0]) < 0:
            raise ValueError("invalid Lite3 buffer")
        return self

    @classmethod
    def new_array(cls):
        """Empty root array, ready for .root().append(...)."""
        cdef Lite3 self = cls()
        l3_init_arr(self.c)
        return self

    def to_bytes(self):
        """Copy of the wire buffer. For zero-copy send, use memoryview(self)."""
        return bytes(memoryview(self))

    def to_json(self):
        cdef char *s = l3_to_json(self.c, 0)
        if s is NULL:
            raise ValueError("JSON encode failed")
        try:
            return (<bytes>s).decode()
        finally:
            l3_free(s)

    def to_dict(self):
        return _json.loads(self.to_json())

    cdef object _view(self):
        cdef int t = l3_root_type(self.c)
        if t == T_OBJECT:
            return _ObjView(self, 0)
        if t == T_ARRAY:
            return _ArrView(self, 0)
        raise ValueError("uninitialized Lite3 buffer")

    def root(self):
        return self._view()

    def __getitem__(self, k):
        return self._view()[k]

    def __setitem__(self, k, v):
        self._view()[k] = v

    def __len__(self):
        return len(self._view())

    def __contains__(self, k):
        return k in self._view()

    def __iter__(self):
        return iter(self._view())

    def keys(self):
        return self._view().keys()

    # zero-copy buffer protocol: memoryview(lite3) is the wire bytes, no copy.
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef const uint8_t *p = l3_buf(self.c)
        cdef size_t n = l3_buflen(self.c)
        buffer.buf = <void *>p
        buffer.obj = self
        buffer.len = n
        buffer.itemsize = 1
        buffer.readonly = 1
        buffer.ndim = 1
        buffer.format = NULL
        buffer.shape = NULL
        buffer.strides = NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL


cdef object _scalar_by_key(Lite3 owner, size_t ofs, str key):
    cdef bytes kb = key.encode()
    cdef const char *k = kb
    cdef int t = l3_get_type(owner.c, ofs, k)
    cdef int64_t iv
    cdef double dv
    cdef int bv
    cdef const char *sp
    cdef const uint8_t *bp
    cdef size_t sl
    cdef size_t sub
    if t == T_NULL:
        return None
    if t == T_BOOL:
        l3_get_bool(owner.c, ofs, k, &bv); return bool(bv)
    if t == T_I64:
        l3_get_i64(owner.c, ofs, k, &iv); return iv
    if t == T_F64:
        l3_get_f64(owner.c, ofs, k, &dv); return dv
    if t == T_STRING:
        l3_get_str(owner.c, ofs, k, &sp, &sl); return sp[:sl].decode()
    if t == T_BYTES:
        l3_get_bytes(owner.c, ofs, k, &bp, &sl); return bytes(bp[:sl])
    if t == T_OBJECT:
        l3_get_obj(owner.c, ofs, k, &sub); return _ObjView(owner, sub)
    if t == T_ARRAY:
        l3_get_arr(owner.c, ofs, k, &sub); return _ArrView(owner, sub)
    raise KeyError(key)


cdef object _scalar_by_index(Lite3 owner, size_t ofs, uint32_t i):
    cdef int t = l3_arr_get_type(owner.c, ofs, i)
    cdef int64_t iv
    cdef double dv
    cdef int bv
    cdef const char *sp
    cdef const uint8_t *bp
    cdef size_t sl
    cdef size_t sub
    if t == T_NULL:
        return None
    if t == T_BOOL:
        l3_arr_get_bool(owner.c, ofs, i, &bv); return bool(bv)
    if t == T_I64:
        l3_arr_get_i64(owner.c, ofs, i, &iv); return iv
    if t == T_F64:
        l3_arr_get_f64(owner.c, ofs, i, &dv); return dv
    if t == T_STRING:
        l3_arr_get_str(owner.c, ofs, i, &sp, &sl); return sp[:sl].decode()
    if t == T_BYTES:
        l3_arr_get_bytes(owner.c, ofs, i, &bp, &sl); return bytes(bp[:sl])
    if t == T_OBJECT:
        l3_arr_get_obj(owner.c, ofs, i, &sub); return _ObjView(owner, sub)
    if t == T_ARRAY:
        l3_arr_get_arr(owner.c, ofs, i, &sub); return _ArrView(owner, sub)
    raise IndexError(i)


cdef int _set_value(Lite3 owner, size_t ofs, str key, object v) except -1:
    cdef bytes kb = key.encode()
    cdef const char *k = kb
    cdef bytes sb
    cdef size_t child
    if v is None:
        l3_set_null(owner.c, ofs, k)
    elif v is True or v is False:          # bool before int (bool is an int subclass)
        l3_set_bool(owner.c, ofs, k, 1 if v else 0)
    elif isinstance(v, int):
        l3_set_i64(owner.c, ofs, k, <int64_t>v)
    elif isinstance(v, float):
        l3_set_f64(owner.c, ofs, k, <double>v)
    elif isinstance(v, str):
        sb = v.encode()
        l3_set_str(owner.c, ofs, k, sb, len(sb))
    elif isinstance(v, (bytes, bytearray)):
        sb = bytes(v)
        l3_set_bytes(owner.c, ofs, k, <const uint8_t*>(<char*>sb), len(sb))
    elif isinstance(v, dict):
        l3_set_obj(owner.c, ofs, k, &child)
        for kk, vv in v.items():
            _set_value(owner, child, kk, vv)
    elif isinstance(v, (list, tuple)):
        l3_set_arr(owner.c, ofs, k, &child)
        for item in v:
            _append_value(owner, child, item)
    else:
        raise TypeError("unsupported value type: %r" % type(v))
    return 0


cdef int _append_value(Lite3 owner, size_t ofs, object v) except -1:
    cdef bytes sb
    cdef size_t child
    if v is None:
        l3_app_null(owner.c, ofs)
    elif v is True or v is False:
        l3_app_bool(owner.c, ofs, 1 if v else 0)
    elif isinstance(v, int):
        l3_app_i64(owner.c, ofs, <int64_t>v)
    elif isinstance(v, float):
        l3_app_f64(owner.c, ofs, <double>v)
    elif isinstance(v, str):
        sb = v.encode()
        l3_app_str(owner.c, ofs, sb, len(sb))
    elif isinstance(v, (bytes, bytearray)):
        sb = bytes(v)
        l3_app_bytes(owner.c, ofs, <const uint8_t*>(<char*>sb), len(sb))
    elif isinstance(v, dict):
        l3_app_obj(owner.c, ofs, &child)
        for kk, vv in v.items():
            _set_value(owner, child, kk, vv)
    elif isinstance(v, (list, tuple)):
        l3_app_arr(owner.c, ofs, &child)
        for item in v:
            _append_value(owner, child, item)
    else:
        raise TypeError("unsupported value type: %r" % type(v))
    return 0


# ponytail: third copy of the _set_value/_append_value type ladder (differs only
# in the l3_* family called). A new type means editing three sites. Collapse to one
# dispatcher parameterized by the leaf-writer if a fourth target ever appears.
cdef int _set_index_value(Lite3 owner, size_t ofs, uint32_t i, object v) except -1:
    cdef bytes sb
    cdef size_t child
    if v is None:
        l3_arr_set_null(owner.c, ofs, i)
    elif v is True or v is False:
        l3_arr_set_bool(owner.c, ofs, i, 1 if v else 0)
    elif isinstance(v, int):
        l3_arr_set_i64(owner.c, ofs, i, <int64_t>v)
    elif isinstance(v, float):
        l3_arr_set_f64(owner.c, ofs, i, <double>v)
    elif isinstance(v, str):
        sb = v.encode()
        l3_arr_set_str(owner.c, ofs, i, sb, len(sb))
    elif isinstance(v, (bytes, bytearray)):
        sb = bytes(v)
        l3_arr_set_bytes(owner.c, ofs, i, <const uint8_t*>(<char*>sb), len(sb))
    elif isinstance(v, dict):
        l3_arr_set_obj(owner.c, ofs, i, &child)
        for kk, vv in v.items():
            _set_value(owner, child, kk, vv)
    elif isinstance(v, (list, tuple)):
        l3_arr_set_arr(owner.c, ofs, i, &child)
        for item in v:
            _append_value(owner, child, item)
    else:
        raise TypeError("unsupported value type: %r" % type(v))
    return 0


cdef class _ObjView:
    """Lazy proxy over a Lite³ object at a buffer offset. Reads on demand.

    Writes (__setitem__) are supported, but mutating the buffer can relocate
    nodes — re-fetch any view from the root after a write rather than reusing
    a held proxy.
    """
    cdef Lite3 owner
    cdef size_t ofs

    def __cinit__(self, Lite3 owner, size_t ofs):
        self.owner = owner
        self.ofs = ofs

    def __getitem__(self, str key):
        if l3_get_type(self.owner.c, self.ofs, key.encode()) >= T_INVALID:
            raise KeyError(key)
        return _scalar_by_key(self.owner, self.ofs, key)

    def __setitem__(self, str key, value):
        _set_value(self.owner, self.ofs, key, value)

    def get(self, str key, default=None):
        if l3_get_type(self.owner.c, self.ofs, key.encode()) >= T_INVALID:
            return default
        return _scalar_by_key(self.owner, self.ofs, key)

    def __contains__(self, str key):
        return l3_get_type(self.owner.c, self.ofs, key.encode()) < T_INVALID

    def __len__(self):
        cdef long n = l3_count(self.owner.c, self.ofs)
        if n < 0:
            raise ValueError("count failed")
        return n

    def keys(self):
        cdef l3_iter *it = l3_iter_new(self.owner.c, self.ofs)
        cdef const char *kp
        cdef size_t kl
        cdef list out = []
        if it is NULL:
            raise MemoryError("iterator alloc failed")
        try:
            # keys are NUL-terminated C strings; key.len is not the byte length
            while l3_iter_next(it, &kp, &kl) == 1:
                out.append((<bytes>kp).decode())
        finally:
            l3_iter_free(it)
        return out

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, (_ObjView, _ArrView)) else v)
                for k, v in self.items()}


cdef class _ArrView:
    """Lazy proxy over a Lite³ array at a buffer offset."""
    cdef Lite3 owner
    cdef size_t ofs

    def __cinit__(self, Lite3 owner, size_t ofs):
        self.owner = owner
        self.ofs = ofs

    def __len__(self):
        cdef long n = l3_count(self.owner.c, self.ofs)
        if n < 0:
            raise ValueError("count failed")
        return n

    def __getitem__(self, Py_ssize_t i):
        cdef Py_ssize_t n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(i)
        return _scalar_by_index(self.owner, self.ofs, <uint32_t>i)

    def __setitem__(self, Py_ssize_t i, value):
        cdef Py_ssize_t n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(i)
        _set_index_value(self.owner, self.ofs, <uint32_t>i, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, value):
        _append_value(self.owner, self.ofs, value)

    def extend(self, values):
        for v in values:
            _append_value(self.owner, self.ofs, v)

    def to_list(self):
        return [(v.to_dict() if isinstance(v, (_ObjView, _ArrView)) else v) for v in self]

    to_dict = to_list
