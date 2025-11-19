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

#include "lite3.h"


static unsigned char buf[1024], rx[1024];

int main() {
        size_t buflen = 0;
        size_t bufsz = sizeof(buf);

        // Build message
        if (lite3_init_obj(buf, &buflen, bufsz)                                         < 0
                || lite3_set_str(buf, &buflen, 0, bufsz, "event", "lap_complete")       < 0
                || lite3_set_i64(buf, &buflen, 0, bufsz, "lap", 55)                     < 0
                || lite3_set_f64(buf, &buflen, 0, bufsz, "time_sec", 88.427)            < 0) {
                perror("Failed to build message");
                return 1;
        }
        printf("buflen: %zu\n", buflen);
        if (lite3_json_print(buf, buflen, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        printf("\nUpdating lap count\n");
        if (lite3_set_i64(buf, &buflen, 0, bufsz, "lap", 56) < 0) {
                perror("Failed to update lap count");
                return 1;
        }
        
        printf("Data to send:\n");
        printf("buflen: %zu\n", buflen);
        if (lite3_json_print(buf, buflen, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }
        
        // Transmit
        size_t rx_buflen = buflen;
        size_t rx_bufsz = sizeof(rx);
        memcpy(rx, buf, buflen);
        
        // Mutate (zero-copy, no parsing)
        printf("\nVerifying fastest lap\n");
        if (lite3_set_str(rx, &rx_buflen, 0, rx_bufsz, "verified", "race_control")      < 0
                || lite3_set_bool(rx, &rx_buflen, 0, rx_bufsz, "fastest_lap", true)     < 0) {
                perror("Failed to verify lap");
                return 1;
        }

        printf("Modified data:\n");
        printf("rx_buflen: %zu\n", rx_buflen);
        if (lite3_json_print(rx, rx_buflen, 0) < 0) {
                perror("Failed to print JSON");
                return 1;
        }

        // Ready to send:
        // send(sock, rx, rx_buflen, 0);

        return 0;
}