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


int main() {
        lite3_ctx *ctx = lite3_ctx_create();
        if (!ctx) {
                perror("Failed to create lite3_ctx *ctx");
                return 1;
        }

        // Build message
        if (lite3_ctx_init_obj(ctx)                                                             < 0
                || lite3_ctx_set_str(ctx, 0, "title", "C Programming Language, 2nd Edition")    < 0
                || lite3_ctx_set_str(ctx, 0, "language", "en")                                  < 0
                || lite3_ctx_set_f64(ctx, 0, "price_usd", 60.30)                                < 0
                || lite3_ctx_set_i64(ctx, 0, "pages", 272)                                      < 0
                || lite3_ctx_set_bool(ctx, 0, "in_stock", true)                                 < 0
                || lite3_ctx_set_null(ctx, 0, "reviews")                                        < 0) {
                perror("Failed to build message");
                return 1;
        }
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        lite3_str title, language;
        double price_usd;
        int64_t pages;
        bool in_stock;
        if (lite3_ctx_get_str(ctx, 0, "title", &title)                  < 0
                || lite3_ctx_get_str(ctx, 0, "language", &language)     < 0
                || lite3_ctx_get_f64(ctx, 0, "price_usd", &price_usd)   < 0
                || lite3_ctx_get_i64(ctx, 0, "pages", &pages)           < 0
                || lite3_ctx_get_bool(ctx, 0, "in_stock", &in_stock)    < 0) {
                perror("Failed to read message");
                return 1;
        }
        printf("\ntitle: %s\n", LITE3_STR(ctx->buf, title));
        printf("language: %s\n", LITE3_STR(ctx->buf, language));
        printf("price_usd: %f\n", price_usd);
        printf("pages: %li\n", pages);
        printf("in_stock: %s\n\n", in_stock ? "true" : "false");
        
        if (lite3_ctx_is_null(ctx, 0, "reviews")) {
                printf("No reviews to display.\n");
        }

        printf("\nTitle field exists: %s\n", lite3_ctx_exists(ctx, 0, "title") ? "true" : "false");
        printf("Price field exists: %s\n", lite3_ctx_exists(ctx, 0, "price_usd") ? "true" : "false");
        printf("ISBN field exists: %s\n", lite3_ctx_exists(ctx, 0, "isbn") ? "true" : "false");

        enum lite3_type title_type = lite3_ctx_get_type(ctx, 0, "title");
        printf("\nTitle is string type: %s\n", title_type == LITE3_TYPE_STRING ? "true" : "false");
        printf("Title is integer type: %s\n", title_type == LITE3_TYPE_I64 ? "true" : "false");

        lite3_val *price_val;
        if (lite3_ctx_get(ctx, 0, "price_usd", &price_val) < 0) {
                perror("Failed to get price_usd");
                return 1;
        }
        printf("\nPrice is string type: %s\n", lite3_val_is_str(price_val) ? "true" : "false");
        printf("Price is double type: %s\n", lite3_val_is_f64(price_val) ? "true" : "false");
        if (price_val->type == LITE3_TYPE_F64) {
                printf("price_val value: %f\n", lite3_val_f64(price_val));
                printf("price_val type size: %zu\n", lite3_val_type_size(price_val));
        }
        
        uint32_t entry_count;
        if (lite3_ctx_count(ctx, 0, &entry_count) < 0) {
                perror("Failed to get entry count");
                return 1;
        }
        printf("\nObject entries: %u\n", entry_count);

        lite3_ctx_destroy(ctx);
        return 0;
}