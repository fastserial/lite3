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
        if (lite3_ctx_init_obj(ctx)                                     < 0
                || lite3_ctx_set_str(ctx, 0, "event", "lap_complete")   < 0
                || lite3_ctx_set_i64(ctx, 0, "lap", 55)                 < 0
                || lite3_ctx_set_f64(ctx, 0, "time_sec", 88.427)        < 0) {
                perror("Failed to build message");
                return 1;
        }
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        printf("\nUpdating lap count\n");
        if (lite3_ctx_set_i64(ctx, 0, "lap", 56) < 0) {
                perror("Failed to update lap count");
                return 1;
        }
        printf("Data to send:\n");
        printf("buflen: %zu\n", ctx->buflen);
        if (lite3_ctx_json_print(ctx, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }

        // Transmit data / copy to new context
        lite3_ctx *rx = lite3_ctx_create_from_buf(ctx->buf, ctx->buflen);
        if (!rx) {
                perror("Failed create lite3_ctx *rx");
                return 1;
        }

        // Mutate (zero-copy, no parsing)
        printf("\nVerifying fastest lap\n");
        if (lite3_ctx_set_str(rx, 0, "verified", "race_control")        < 0
                || lite3_ctx_set_bool(rx, 0, "fastest_lap", true)       < 0) {
                perror("Failed to verify lap");
                return 1;
        }
        printf("Modified data:\n");
        printf("rx_buflen: %zu\n", rx->buflen);
        if (lite3_ctx_json_print(rx, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }

        // Ready to send:
        // send(sock, rx->buf, rx->buflen, 0);

        lite3_ctx_destroy(rx);
        lite3_ctx_destroy(ctx);
        return 0;
}