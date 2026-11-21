#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for json in 1 0; do
    prefix="$tmp/install-$json"
    make -j2 install LITE3_JSON_LIBRARIES="$json" \
        OBJ_DIR="$tmp/obj-$json" BUILD_DIR="$tmp/build-$json" PREFIX="$prefix"

    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    pkg-config --validate lite3
    test "$(pkg-config --modversion lite3)" = "$(sed -n 's/^Version: //p' lite3.pc)"

    if [ "$json" = 1 ]; then
        consumer=tests/examples/buffer_api_07_json_conversion.c
    else
        consumer=tests/nested_generations.c
    fi
    # The staging directory has no whitespace; pkg-config emits separate flags.
    cc "$consumer" $(pkg-config --cflags --libs --static lite3) -o "$tmp/consumer"
    "$tmp/consumer"
done
