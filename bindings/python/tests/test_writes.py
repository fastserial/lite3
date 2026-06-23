"""Write API + wire round-trip. Run with the read tests via:
   bindings/python/.venv/bin/python tests/test_writes.py"""
import random
import string

from lite3 import Lite3

# JSON-safe (from_dict/to_dict go through JSON, which has no bytes type)
SAMPLE = {
    "event": "http_request",
    "n": 47,
    "ok": True,
    "ratio": 1.5,
    "missing": None,
    "headers": {"content-type": "application/json", "id": "req_9f8e2a"},
    "tags": ["a", "b", "c"],
    "nested": {"deep": {"n": 1, "list": [1, 2, {"k": "v"}]}},
}


def test_from_bytes_roundtrip():
    # produce wire bytes, then ingest them as if received from a socket
    sent = Lite3.from_dict(SAMPLE)
    wire = sent.to_bytes()
    got = Lite3.from_bytes(wire)
    assert got.to_dict() == SAMPLE
    # zero-copy view equals the copy
    assert bytes(memoryview(got)) == wire


def test_typed_writes_scalar():
    m = Lite3()                 # empty object
    m["event"] = "ping"
    m["n"] = 7
    m["ok"] = False
    m["ratio"] = 2.5
    m["missing"] = None
    assert m.to_dict() == {
        "event": "ping", "n": 7, "ok": False, "ratio": 2.5, "missing": None,
    }


def test_typed_writes_nested():
    m = Lite3()
    m["headers"] = {"a": 1, "b": {"c": [1, 2, 3]}}
    m["list"] = [1, "two", {"three": 3}, [4, 5]]
    assert m.to_dict() == {
        "headers": {"a": 1, "b": {"c": [1, 2, 3]}},
        "list": [1, "two", {"three": 3}, [4, 5]],
    }


def test_array_root_and_append():
    a = Lite3.new_array()
    a.root().append(1)
    a.root().append("x")
    a.root().extend([{"k": "v"}, [9]])
    assert a.to_dict() == [1, "x", {"k": "v"}, [9]]


def test_array_index_overwrite():
    a = Lite3.new_array()
    a.root().extend([1, "two", 3, [9]])
    r = a.root()
    r[0] = 100              # scalar overwrite
    r[1] = "TWO"           # str (variable length)
    r[-1] = {"k": "v"}     # negative index, container overwrite
    assert a.to_dict() == [100, "TWO", 3, {"k": "v"}]
    try:
        r[10] = 1
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-bounds index did not raise")
    # nested array overwrite, then wire round-trip
    m = Lite3.from_dict({"xs": [1, 2, 3]})
    m["xs"][1] = 22
    assert Lite3.from_bytes(m.to_bytes()).to_dict() == {"xs": [1, 22, 3]}


def test_mutate_and_forward():
    # receive -> overwrite a field -> forward.  Re-fetch from root after write.
    msg = Lite3.from_bytes(Lite3.from_dict({"hops": 0, "path": "a"}).to_bytes())
    msg["hops"] = msg["hops"] + 1
    assert msg["hops"] == 1
    assert Lite3.from_bytes(msg.to_bytes()).to_dict() == {"hops": 1, "path": "a"}


def test_bytes_via_binary_path():
    # bytes survive only through the binary path (not JSON). Write typed, read typed.
    m = Lite3()
    m["blob"] = b"\x00\x01\xff\xfe"
    assert m["blob"] == b"\x00\x01\xff\xfe"
    # and across a wire round-trip
    assert Lite3.from_bytes(m.to_bytes())["blob"] == b"\x00\x01\xff\xfe"


# --- fuzz: build via typed writes (NOT from_dict), exercising the offset logic
def _rand(depth, rng):
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice([
            rng.randint(-10**12, 10**12),
            round(rng.uniform(-1e6, 1e6), 6),
            "".join(rng.choices(string.ascii_letters, k=rng.randint(0, 10))),
            rng.choice([True, False]),
            None,
        ])
    if rng.random() < 0.5:
        return [_rand(depth - 1, rng) for _ in range(rng.randint(0, 5))]
    return {"k%d" % i: _rand(depth - 1, rng) for i in range(rng.randint(0, 5))}


def test_fuzz_typed_writes():
    rng = random.Random(99)
    for _ in range(500):
        top = {"f%d" % i: _rand(3, rng) for i in range(rng.randint(1, 6))}
        m = Lite3()
        for k, v in top.items():
            m[k] = v
        assert m.to_dict() == top, top
        # and survives a wire round-trip
        assert Lite3.from_bytes(m.to_bytes()).to_dict() == top, top


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASSED")
