"""Lite³ — zero-copy serialization, Python binding.

Build / construct:
    from lite3 import Lite3
    msg = Lite3.from_dict({"event": "ping", "n": 3})  # via JSON (slow path)
    msg = Lite3()                 # empty object; then msg["k"] = v
    arr = Lite3.new_array()       # empty array; then arr.root().append(v)

Wire protocol:
    wire = msg.to_bytes()             # copy of the buffer
    sock.send(memoryview(msg))        # zero-copy send (no serialize step)
    got  = Lite3.from_bytes(wire)     # ingest a received buffer

Read (lazy, straight from the buffer — no parsing):
    got["event"]                      # -> "ping"
    got["headers"]["content-type"]    # nested proxy
    got["tags"][0]                    # array proxy

Write (typed; bytes/int/float/str/bool/None/dict/list):
    msg["hops"] = msg["hops"] + 1     # mutate, then forward msg.to_bytes()

Notes:
- to_dict()/to_json() are explicit slow paths (full hydration).
- JSON has no bytes type: `bytes` round-trips only via the binary path
  (typed write + from_bytes/typed read), not from_dict/to_dict.
- Mutating relocates buffer nodes — re-fetch views from the root after a write
  rather than reusing a held proxy.
"""
from ._core import Lite3

__all__ = ["Lite3"]
