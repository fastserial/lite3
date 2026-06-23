import shutil
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
VENDOR = HERE / "_vendor"

# The C core lives at the repo root (../../). That's outside this package, so it
# is invisible under build isolation and absent from an sdist. Vendor it in: copy
# when building from the repo, reuse the copy when building from an unpacked sdist.
# This is what makes `pip install .`, sdist, and cibuildwheel work.
# ponytail: wholesale dir copy, no per-file curation — robust to new C files.
def vendor():
    if (ROOT / "src" / "lite3.c").is_file():  # building from the repo
        for d in ("include", "lib", "src"):
            dst = VENDOR / d
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(ROOT / d, dst)
    elif not VENDOR.is_dir():  # not the repo and no bundled copy (sdist carries one)
        raise SystemExit("setup.py: C sources not found — build from the repo root or a bundled _vendor/")

vendor()

def vsrc(*parts):
    return [str(p.relative_to(HERE)) for p in (VENDOR / Path(*parts[:-1])).glob(parts[-1])]

sources = (
    ["lite3/_core.pyx", "src/lite3_shim.c"]
    + vsrc("src", "*.c")
    + vsrc("lib", "yyjson", "*.c")
    + vsrc("lib", "nibble_base64", "*.c")
)

ext = Extension(
    "lite3._core",
    sources=sources,
    include_dirs=[str(VENDOR / "include"), str(VENDOR / "lib"), "src"],
    extra_compile_args=["-O2", "-std=gnu11", "-DNDEBUG"],
)

setup(ext_modules=cythonize([ext], language_level=3))
