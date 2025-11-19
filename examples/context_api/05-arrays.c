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

#include "lite3_context_api.h"


int main() {
        lite3_ctx *ctx = lite3_ctx_create();
        if (!ctx) {
                perror("Failed to create lite3_ctx *ctx");
                return 1;
        }

        // Build message
        if (lite3_ctx_init_arr(ctx)                             < 0
                || lite3_ctx_arr_append_str(ctx, 0, "zebra")    < 0
                || lite3_ctx_arr_append_str(ctx, 0, "giraffe")  < 0
                || lite3_ctx_arr_append_str(ctx, 0, "buffalo")  < 0
                || lite3_ctx_arr_append_str(ctx, 0, "lion")     < 0
                || lite3_ctx_arr_append_str(ctx, 0, "rhino")    < 0
                || lite3_ctx_arr_append_str(ctx, 0, "elephant") < 0) {
                perror("Failed to build message");
                return 1;
        }
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        lite3_str element_2;
        if (lite3_ctx_arr_get_str(ctx, 0, 2, &element_2) < 0) {
                perror("Failed to get element");
                return 1;
        }
        printf("Element at index 2: %s\n", LITE3_STR(ctx->buf, element_2));

        uint32_t element_count;
        if (lite3_ctx_count(ctx, 0, &element_count) < 0) {
                perror("Failed to get element count");
                return 1;
        }
        printf("Element count: %u\n", element_count);

        lite3_str last_element;
        if (lite3_ctx_arr_get_str(ctx, 0, element_count - 1, &last_element) < 0) {
                perror("Failed to get element");
                return 1;
        }
        printf("Last element: %s\n", LITE3_STR(ctx->buf, last_element));

        printf("\nOverwriting index 2 with \"gnu\"\n");
        if (lite3_ctx_arr_set_str(ctx, 0, 2, "gnu") < 0) {
                perror("Failed to set element");
                return 1;
        }
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }

        printf("\nOverwriting index 3 with \"springbok\"\n");
        if (lite3_ctx_arr_set_str(ctx, 0, 3, "springbok") < 0) {
                perror("Failed to set element");
                return 1;
        }
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }

        lite3_ctx_destroy(ctx);
        return 0;
}