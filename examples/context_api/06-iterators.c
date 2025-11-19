/*
    Lite³: A JSON-Compatible Zero-Copy Serialization Format

    Copyright © 2025 Elias de Jong <elias@fastserial.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

      __ __________________        ____
    _  ___ ___/ /___(_)_/ /_______|_  /
     _  _____/ / __/ /_  __/  _ \_/_ < 
      ___ __/ /___/ / / /_ /  __/____/ 
           /_____/_/  \__/ \___/       
*/
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "lite3_context_api.h"


#define NAME_COUNT 6

const char names[NAME_COUNT][10] = {
        "Boris",
        "John",
        "Olivia",
        "Tanya",
        "Paul",
        "Sarah",
};

int main() {
        lite3_ctx *ctx = lite3_ctx_create();
        if (!ctx) {
                perror("Failed to create lite3_ctx *ctx");
                return 1;
        }

        // Build array
        if (lite3_ctx_init_arr(ctx) < 0) {
                perror("Failed to initialize array");
                return 1;
        }
        for (int i = 0; i < NAME_COUNT; i++) {
                size_t obj_ofs;
                if (lite3_ctx_arr_append_obj(ctx, 0, &obj_ofs)                  < 0
                        || lite3_ctx_set_i64(ctx, obj_ofs, "id", (int64_t)i)    < 0
                        || lite3_ctx_set_bool(ctx, obj_ofs, "vip_member", false)< 0
                        || lite3_ctx_set_null(ctx, obj_ofs, "benefits")         < 0
                        || lite3_ctx_set_str(ctx, obj_ofs, "name", names[i])    < 0) {
                        perror("Failed to build array");
                        return 1;
                }
        }
        if (lite3_ctx_json_print(ctx, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        // Iterate over array objects
        lite3_iter iter;
        if (lite3_ctx_iter_create(ctx, 0, &iter) < 0) {
                perror("Failed to create iterator");
                return 1;
        }
        size_t val_ofs;
        while (lite3_ctx_iter_next(ctx, &iter, NULL, &val_ofs) == LITE3_ITER_ITEM) {
                int64_t id;
                bool vip_member;
                bool benefits = !lite3_ctx_is_null(ctx, val_ofs, "benefits");
                lite3_str name;
                if (lite3_ctx_get_i64(ctx, val_ofs, "id", &id)                          < 0
                        || lite3_ctx_get_bool(ctx, val_ofs, "vip_member", &vip_member)  < 0
                        || lite3_ctx_get_str(ctx, val_ofs, "name", &name)               < 0) {
                        perror("Failed to get object");
                        return 1;
                }
                printf("id: %li\tname: %s\tvip_member: %s\tbenefits: %s\n",
                        id,
                        LITE3_STR(ctx->buf, name),
                        vip_member ? "true" : "false",
                        benefits ? "yes" : "no"
                );
        }

        // Iterate over object key-value pairs
        lite3_iter iter_2;
        if (lite3_ctx_iter_create(ctx, val_ofs, &iter_2) < 0) {
                perror("Failed to create iterator");
                return 1;
        }
        printf("\nObject keys:\n");
        lite3_str key;
        size_t val_ofs_2;
        while (lite3_ctx_iter_next(ctx, &iter_2, &key, &val_ofs_2) == LITE3_ITER_ITEM) {

                lite3_val *val = (lite3_val *)(ctx->buf + val_ofs_2);
                printf("key: %s\tvalue: ", LITE3_STR(ctx->buf, key));

                switch (val->type) {
                case LITE3_TYPE_I64:
                        printf("%li\n", lite3_val_i64(val));
                        break;
                case LITE3_TYPE_BOOL:
                        printf("%s\n", lite3_val_bool(val) ? "true" : "false");
                        break;
                case LITE3_TYPE_NULL:
                        printf("null\n");
                        break;
                case LITE3_TYPE_STRING:
                        printf("%s\n", lite3_val_str(val));
                        break;
                default:
                        fprintf(stderr, "Invalid object value type\n");
                        return 1;
                }
        }

        lite3_ctx_destroy(ctx);
        return 0;
}