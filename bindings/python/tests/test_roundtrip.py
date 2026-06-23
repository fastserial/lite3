"""Run: bindings/python/.venv/bin/python -m pytest tests/  (or just run this file)."""
import json
import random
import string

from lite3 import Lite3

SAMPLE = {
    "event": "http_request",
    "method": "POST",
    "duration_ms": 47,
    "ok": True,
    "ratio": 1.5,
    "missing": None,
    "headers": {"content-type": "application/json", "x-request-id": "req_9f8e2a"},
    "tags": ["a", "b", "c"],
    "nested": {"deep": {"n": 1, "list": [1, 2, {"k": "v"}]}},
}


def test_dict_roundtrip():
    m = Lite3.from_dict(SAMPLE)
    assert m.to_dict() == SAMPLE


def test_lazy_reads():
    m = Lite3.from_dict(SAMPLE)
    assert m["method"] == "POST"
    assert m["duration_ms"] == 47
    assert m["ok"] is True
    assert m["ratio"] == 1.5
    assert m["missing"] is None
    assert m["headers"]["content-type"] == "application/json"   # nested proxy
    assert m["tags"][1] == "b"                                   # array proxy
    assert m["nested"]["deep"]["list"][2]["k"] == "v"            # deep mixed
    assert len(m["tags"]) == 3
    assert "headers" in m
    assert set(m["headers"].keys()) == {"content-type", "x-request-id"}


def test_zero_copy_buffer():
    m = Lite3.from_dict(SAMPLE)
    mv = memoryview(m)
    assert mv.readonly and len(mv) > 0
    # round-trips through the wire bytes via JSON oracle
    assert Lite3.from_json(m.to_json()).to_dict() == SAMPLE


def test_json_oracle_roundtrip():
    m = Lite3.from_dict(SAMPLE)
    assert json.loads(m.to_json()) == SAMPLE


# --- fuzz: random nested structures through from_dict -> to_dict and to/from_json
def _rand(depth, rng):
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice([
            rng.randint(-10**12, 10**12),
            round(rng.uniform(-1e6, 1e6), 6),
            "".join(rng.choices(string.ascii_letters + " _-", k=rng.randint(0, 12))),
            rng.choice([True, False]),
            None,
        ])
    if rng.random() < 0.5:
        return [_rand(depth - 1, rng) for _ in range(rng.randint(0, 5))]
    return {
        "k%d" % i: _rand(depth - 1, rng)
        for i in range(rng.randint(0, 5))
    }


def test_fuzz_roundtrip():
    rng = random.Random(1337)
    for _ in range(500):
        top = {"f%d" % i: _rand(3, rng) for i in range(rng.randint(1, 6))}
        m = Lite3.from_dict(top)
        assert m.to_dict() == top, top
        assert Lite3.from_json(m.to_json()).to_dict() == top, top


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASSED")
