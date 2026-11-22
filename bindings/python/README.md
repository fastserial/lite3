# lite3 — Python binding

Zero-copy reads over the Lite³ wire format, plus typed writes. Build or receive a
message, then index it like a `dict`/`list` — fields are read straight out of the
serialized buffer on demand (lazy proxies, never a hydrated copy).

## Install

Builds the C core (lite3 + bundled yyjson) as a Cython extension:

```sh
cd bindings/python
python3 -m venv .venv      # optional, but recommended
.venv/bin/pip install .    # or `-e .` for an editable dev install
```

Requires a C compiler, Python ≥ 3.11, and Cython ≥ 3 (pulled in by the build).
The build compiles the lite3 sources directly and does **not** use the repo
`Makefile`.

At build time the C core (`src/`, `lib/`, `include/` from the repo root) is
copied into a local `_vendor/` so the package is self-contained — `pip install .`,
`sdist`, and `cibuildwheel` all work without reaching outside the package. The
repo root stays the single source of truth; `_vendor/` is generated and gitignored.

## Quickstart

```python
from lite3 import Lite3

# construct
msg = Lite3.from_dict({"event": "ping", "headers": {"id": "req_9f"}, "tags": ["a", "b"]})
msg = Lite3()                 # empty object, then msg["k"] = v
arr = Lite3.new_array()       # empty array, then arr.root().append(v)

# wire protocol
wire = msg.to_bytes()         # copy of the buffer
sock.send(memoryview(msg))    # zero-copy send — no serialize step
got  = Lite3.from_bytes(wire) # ingest a received buffer

# read (lazy — only touched fields are read)
got["event"]                  # "ping"
got["headers"]["id"]          # "req_9f"  — nested proxy, still no copy
got["tags"][1]                # "b"       — array proxy

# write (typed): None/bool/int/float/str/bytes/dict/list
got["hops"] = 1               # then forward got.to_bytes()
arr.root().append({"k": "v"})
arr.root()[0] = 100           # array index overwrite
```

`memoryview(msg)` *is* the on-wire format. Send it on a socket, mmap it, hand it
to another process — no encode pass.

## API

Construct:

| | |
|---|---|
| `Lite3()` | empty object, ready for `msg[key] = v` |
| `Lite3.new_array()` | empty array, use `.root().append(...)` |
| `Lite3.from_dict(d)` | build from a Python dict/list (routes through JSON) |
| `Lite3.from_json(s)` | build from a JSON `str` or `bytes` |
| `Lite3.from_bytes(b)` | ingest a received Lite³ buffer (copies it in) |

Read (on `Lite3` and the nested `_ObjView` / `_ArrView` proxies):

| | |
|---|---|
| `msg[key]` / `msg[i]` | lazy field read; raises `KeyError`/`IndexError` if absent |
| `obj.get(key, default)` | object only; no raise |
| `key in obj` | membership |
| `len(x)` | entry/element count |
| `obj.keys()`, `obj.items()`, `iter(obj)` | object enumeration |
| `iter(arr)` | array iteration |

Write:

| | |
|---|---|
| `obj[key] = v` | set/overwrite object field (recurses for dict/list) |
| `arr[i] = v` | overwrite array element (index must be `< len`) |
| `arr.append(v)`, `arr.extend(vs)` | grow an array |

Serialize:

| | |
|---|---|
| `bytes(memoryview(msg))` | zero-copy wire bytes |
| `msg.to_bytes()` | copy of the wire bytes |
| `msg.to_dict()` / `msg.to_json()` | full hydration (slow paths) |

## Semantics & limits

- **Lifetime.** Proxies and the `memoryview` borrow the `Lite3` buffer — keep the
  `Lite3` alive while you hold them, or you read freed memory.
- **Writes can relocate buffer nodes.** A held `msg["a"]` proxy may go stale
  after a mutation — re-fetch views from the root after writing.
- **`bytes` is binary-path only.** JSON has no bytes type, so `from_dict`/
  `to_dict` base64-encode it. `bytes` round-trips losslessly only via typed
  write + `from_bytes`/typed read. `to_dict`/`to_json` are explicit slow paths.
- **Overwriting a longer string/bytes grows the buffer and never reclaims the
  old space** (a Lite³ property, not a binding bug).
- **Types**: `None`, `bool`, `int` (i64), `float` (f64), `str`, `bytes`, nested
  objects/arrays. Object keys must be strings.

## Tests

```sh
.venv/bin/python tests/test_roundtrip.py   # reads + dict/json round-trip + fuzz
.venv/bin/python tests/test_writes.py      # writes, from_bytes, fuzz
```

---

# Maintainer's guide

## Architecture

Three layers. The middle one — the C shim — is what makes the binding possible.

```
lite3/_core.pyx   Cython: Lite3 + _ObjView/_ArrView proxies, type dispatch
       │          (compiled to an importable .so)
src/lite3_shim.c  thin C file: #includes the headers so the macros expand HERE,
       │          re-exports them as plain extern functions
../../src/*.c     the Lite³ library, compiled in unchanged (+ ../../lib/**)
```

**Why the shim exists.** Lite³'s ergonomic API (`lite3_ctx_set_str`,
`lite3_ctx_get_i64`, the `LITE3_KEY_DATA` key hasher, the auto-grow retry loops)
is **C preprocessor macros and `static inline` functions** in
`include/lite3_context_api.h`. Those never become symbols in the compiled
`.a`/`.so` — they exist only at compile time. They are therefore unreachable
from `ctypes`/`cffi` over a prebuilt library, and the key-hash/grow logic must
not be reimplemented in Python — doing so reintroduces a class of bug Lite³ has
already had to fix. The shim is a C file, so when it calls `lite3_ctx_set_i64`
the macro expands normally; it wraps that in a real exported function the binding
calls. All the tricky logic stays in tested C.

## Changing the binding when the C API changes

Adding or fixing a wrapped call is mechanical — touch **three places in
lockstep**:

1. `src/lite3_shim.h` — declare the plain function, e.g.
   `int l3_get_i64(lite3_ctx *c, size_t ofs, const char *key, int64_t *out);`
2. `src/lite3_shim.c` — one-line body calling the macro/inline:
   `{ return lite3_ctx_get_i64(c, ofs, key, out); }`
3. `lite3/_core.pyx` — add the matching line inside `cdef extern from
   "lite3_shim.h"`, then call it from a `Lite3`/`_ObjView`/`_ArrView` method.

Then `.venv/bin/pip install -e .` to rebuild. If a C signature changes upstream,
fix it in steps 1–2 (and the `extern` in 3); the Python API stays stable.

Shim notes:
- `lite3_ctx_set_obj`/`set_arr` are macros containing `return` statements — they
  work inside a shim function returning `int` (early returns propagate).
- For strings/bytes the shim returns `(ptr, len)` via `LITE3_STR`/`LITE3_BYTES`
  (the generational-pointer safe-access macros); Cython copies into a Python
  object immediately.
- Object-key gotcha: iterator keys are NUL-terminated C strings; `key.len` is
  **not** the byte length. Read to the NUL (`<bytes>ptr`), not `ptr[:len]`.

## Version coupling — important for the wire format

This binding is **coupled to the Lite³ source it is compiled against**
(`setup.py` globs `../../src/*.c`). Proxies walk byte offsets in the buffer and
`from_bytes`/`to_bytes` are the raw binary format, so:

- Within one build, producer and consumer always match (same C compiled in).
- **Across versions, compatibility is not guaranteed.** The library README
  states the API is unstable, and roadmap items — *"built-in defragmentation
  with GC-index"* and *"write formal spec"* — could change the on-wire buffer
  layout. Bytes written by one version may not read under another.

Built against **Lite³ v1.0.0** (`lite3.pc`), repo commit `7b62398`. Record the
lite3 commit/version any distributed build corresponds to, so a format change
doesn't silently produce buffers an older consumer can't read.

## Publish to PyPI
Set up ``cibuildwheel`` CI.

